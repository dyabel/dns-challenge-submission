import sys
sys.path.append("/data/dy/inteldns-master/inteldns-master")
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import IPython.display as ipd

from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from snr import si_snr
from dnsmos import DNSMOS
import soundfile as sf
from pypesq import pesq
from pystoi import stoi


# score = pesq(ref, deg, sr)
# from train_sdnn_s4 import collate_fn, stft_splitter, stft_mixer, nop_stats, Network
import pysepm
import os
from spikingjelly.clock_driven import neuron, encoding, functional,layer
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
fs = 16000
exp = 'lif'
if exp == 'lif':
    from train_sdnn_s4_snn import collate_fn, stft_splitter, stft_mixer, nop_stats, Network
    trained_folder = 'checkpoints/Traineds4_snn_plif_final'
    ckpt_path = '/ckpt_epoch49.pt'
    b = 128
    
    
# trained_folder = 'checkpoints/Traineds4_b256'
args = yaml.safe_load(open(trained_folder + '/args.txt', 'rt'))
if 'out_delay' not in args.keys():
    args['out_delay'] = 0
if 'n_fft' not in args.keys():
    args['n_fft'] = 512
device = torch.device('cuda:0')
root = args['path']
out_delay = args['out_delay']
n_fft = args['n_fft']
win_length = n_fft
hop_length = n_fft // 4
stats = slayer.utils.LearningStats(accuracy_str='SI-SNR', accuracy_unit='dB')

# train_set = DNSAudio(root=root + 'training_set/')
validation_set = DNSAudio(root=root + 'validation_set/')
noisy, clean, noise, metadata = validation_set[0]

# train_loader = DataLoader(train_set,
#                           batch_size=32,
#                           shuffle=True,
#                           collate_fn=collate_fn,
#                           num_workers=4,
#                           pin_memory=True)
validation_loader = DataLoader(validation_set,
                               batch_size=b,
                               shuffle=True,
                               collate_fn=collate_fn,
                               num_workers=4,
                               pin_memory=False)

# print(args)
args['lr'] = 0.01
net = Network(args['threshold'],
                   args['tau_grad'],
                   args['scale_grad'],
                   args['dmax'],
                   args['out_delay'], dotdict(args)).to(device)
total = sum([param.nelement() for param in net.parameters()])
print('parameter num:', total)

# noisy, clean, noise, metadata = train_set[0]
# noisy = torch.unsqueeze(torch.FloatTensor(noisy), dim=0).to(device)
# noisy_abs, noisy_arg = stft_splitter(noisy, n_fft)
# net(noisy_abs)
net.load_state_dict(torch.load(trained_folder + ckpt_path)['model'])
dnsmos = DNSMOS()

dnsmos_noisy = np.zeros(3)
dnsmos_clean = np.zeros(3)
dnsmos_noise = np.zeros(3)
dnsmos_cleaned  = np.zeros(3)
pesqs = 0
stois = 0
valid_event_counts = []

t_st = datetime.now()
for i, (noisy, clean, noise) in enumerate(validation_loader):
    net.eval()
    with torch.no_grad():
        noisy = noisy.to(device)
        clean = clean.to(device)

        noisy_abs, noisy_arg = stft_splitter(noisy, n_fft)
        clean_abs, clean_arg = stft_splitter(clean, n_fft)

        denoised_abs = net(noisy_abs)
        # valid_event_counts.append(count.cpu().data.numpy())
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        clean_abs = slayer.axon.delay(clean_abs, out_delay)
        clean = slayer.axon.delay(clean, win_length * out_delay)

        loss = F.mse_loss(denoised_abs, clean_abs)
        clean_rec = stft_mixer(denoised_abs, noisy_arg, n_fft)
        score = si_snr(clean_rec, clean)
        functional.reset_net(net)
        # if i % 100 == 0:
        # pesqs += np.sum([pesq(clean1, clean_rec1, fs) for clean1, clean_rec1 in zip(clean.detach().cpu().numpy(), clean_rec.detach().cpu().numpy())])
        # pesqs += np.sum([pysepm.pesq(clean1, clean_rec1, fs)[1] for clean1, clean_rec1 in zip(clean.detach().cpu().numpy(), clean_rec.detach().cpu().numpy())])
            # fwsnrseg += np.sum([pysepm.fwSNRseg(clean1, clean_rec1, fs) for clean1, clean_rec1 in zip(clean.detach().cpu().numpy(), clean_rec.detach().cpu().numpy())])
        # stois += np.sum([stoi(clean1, clean_rec1, fs) for clean1, clean_rec1 in zip(clean.detach().cpu().numpy(), clean_rec.detach().cpu().numpy())])
        # stois += np.sum([pysepm.stoi(clean1, clean_rec1, fs) for clean1, clean_rec1 in zip(clean.detach().cpu().numpy(), clean_rec.detach().cpu().numpy())])
            # print('pesq:', pesq, file=open(os.path.join(trained_folder, 'output.txt'), 'w'))
            # print('fwsnrseg:', fwsnrseg, file=open(os.path.join(trained_folder, 'output.txt'), 'w'))
            # print('stoi:', stoi, file=open(os.path.join(trained_folder, 'output.txt'), 'w'))

        dnsmos_noisy += np.sum(dnsmos(noisy.cpu().data.numpy()), axis=0)
        dnsmos_clean += np.sum(dnsmos(clean.cpu().data.numpy()), axis=0)
        dnsmos_noise += np.sum(dnsmos(noise.cpu().data.numpy()), axis=0)
        dnsmos_cleaned += np.sum(dnsmos(clean_rec.cpu().data.numpy()), axis=0)

        stats.validation.correct_samples += torch.sum(score).item()
        stats.validation.loss_sum += loss.item()
        stats.validation.num_samples += noisy.shape[0]

        processed = i * validation_loader.batch_size
        total = len(validation_loader.dataset)
        time_elapsed = (datetime.now() - t_st).total_seconds()
        samples_sec = time_elapsed / (i + 1) / validation_loader.batch_size
        header_list = [f'Valid: [{processed}/{total} '
                        f'({100.0 * processed / total:.0f}%)]']
        # header_list.append(f'Event rate: {[c.item() for c in count]}')
        print(f'\r{header_list[0]}', end='')

dnsmos_clean /= len(validation_loader.dataset)
dnsmos_noisy /= len(validation_loader.dataset)
dnsmos_noise /= len(validation_loader.dataset)
dnsmos_cleaned /= len(validation_loader.dataset)
# pesqs /= len(validation_loader.dataset)
# stois /= len(validation_loader.dataset)

print()
stats.print(0, i, samples_sec, header=header_list)
print('Avg DNSMOS clean   [ovrl, sig, bak]: ', dnsmos_clean)
print('Avg DNSMOS noisy   [ovrl, sig, bak]: ', dnsmos_noisy)
print('Avg DNSMOS noise   [ovrl, sig, bak]: ', dnsmos_noise)
print('Avg DNSMOS cleaned [ovrl, sig, bak]: ', dnsmos_cleaned)
print('Avg pesq:', pesqs)
print('Avg stoi:', stois)

# mean_events = np.mean(valid_event_counts, axis=0)

neuronops = []
# for block in net.blocks[:-1]:
#     neuronops.append(np.prod(block.neuron.shape))

synops = []
# for events, block in zip(mean_events, net.blocks[1:]):
#     synops.append(events * np.prod(block.synapse.shape))
# print(f'SynOPS: {synops}')
# print(f'Total SynOPS: {sum(synops)} per time-step')
# print(f'Total NeuronOPS: {sum(neuronops)} per time-step')
print(f'Time-step per sample: {noisy_abs.shape[-1]}')
dt = hop_length / metadata['fs']
buffer_latency = dt
print(f'Buffer latency: {buffer_latency * 1000} ms')

t_st = datetime.now()
for i in range(noisy.shape[0]):
    audio = noisy[i].cpu().data.numpy()
    stft = librosa.stft(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    istft = librosa.istft(stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

time_elapsed = (datetime.now() - t_st).total_seconds()

enc_dec_latency = time_elapsed / noisy.shape[0] / 16000 / 30 * hop_length
print(f'STFT + ISTFT latency: {enc_dec_latency * 1000} ms')

dns_delays = []
max_len = 50000  # Only evaluate for first clip of audio
for i in range(noisy.shape[0]):
    delay = np.argmax(np.correlate(noisy[i, :max_len].cpu().data.numpy(),
                                   clean_rec[i, :max_len].cpu().data.numpy(),
                                   'full')) - max_len + 1
    dns_delays.append(delay)
dns_latency = np.mean(dns_delays) / metadata['fs']
print(f'N-DNS latency: {dns_latency * 1000} ms')

base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                        accuracy_unit='dB')
nop_stats(validation_loader, base_stats, base_stats.validation, print=False)

si_snr_i = stats.validation.accuracy - base_stats.validation.accuracy
print(f'SI-SNR  (validation set): {stats.validation.accuracy: .2f} dB')
print(f'SI-SNRi (validation set): {si_snr_i: .2f} dB')

latency = buffer_latency + enc_dec_latency + dns_latency
effective_synops_rate = (sum(synops) + 10 * sum(neuronops)) / dt
synops_delay_product = effective_synops_rate * latency

print(f'Solution Latency                 : {latency * 1000: .3f} ms')
print(f'Power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s')
print(f'PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops')