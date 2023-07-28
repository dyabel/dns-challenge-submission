import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .conv_stft import ConviSTFT, ConvSTFT
from .unet import UNet
from . import complex_nn
from s4.s4_model_complex import ComplexS4Model

class Encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 complex=False,
                 padding_mode='zeros'):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_r = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())
        self.fc_i = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        x_r = self.avg_pool(x[:, :, :, :, 0]).view(b, c)
        x_i = self.avg_pool(x[:, :, :, :, 1]).view(b, c)
        y_r = self.fc_r(x_r).view(b, c, 1, 1, 1) - self.fc_i(x_i).view(
            b, c, 1, 1, 1)
        y_i = self.fc_r(x_i).view(b, c, 1, 1, 1) + self.fc_i(x_r).view(
            b, c, 1, 1, 1)
        y = torch.cat([y_r, y_i], 4)
        return x * y

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class FRCRN(nn.Module):
    r""" Frequency Recurrent CRN """

    def __init__(self,
                 complex,
                 model_complexity,
                 model_depth,
                 log_amp,
                 padding_mode,
                 win_len=640,
                 win_inc=320,
                 fft_len=640,
                 win_type='hann',
                 args=None,
                 **kwargs):
        r"""
        Args:
            complex: Whether to use complex networks.
            model_complexity: define the model complexity with the number of layers
            model_depth: Only two options are available : 10, 20
            log_amp: Whether to use log amplitude to estimate signals
            padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
            win_len: length of window used for defining one frame of sample points
            win_inc: length of window shifting (equivalent to hop_size)
            fft_len: number of Short Time Fourier Transform (STFT) points
            win_type: windowing type used in STFT, eg. 'hanning', 'hamming'
        """
        super().__init__()
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True
        self.stft = ConvSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        self.istft = ConviSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        
        self.s4_model1 = ComplexS4Model(
                        d_input=321,
                        d_output=321,
                        d_model=256,
                        n_layers=4,
                        dropout=0.2,
                        prenorm=False,
                        args=args) # Build Neural Network

        # self.s4_model2 = ComplexS4Model(
        #                 d_input=321,
        #                 d_output=321,
        #                 d_model=256,
        #                 n_layers=4,
        #                 dropout=0.2,
        #                 prenorm=False,
        #                 args=args) # Build Neural Network

       

    def forward(self, inputs):
        out_list = []
        # [B, D*2, T]
        cmp_spec = self.stft(inputs)
        # [B, 1, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)

        # to [B, 2, D, T] real_part/imag_part
        cmp_spec = torch.cat([
            cmp_spec[:, :, :self.feat_dim, :],
            cmp_spec[:, :, self.feat_dim:, :],
        ], 1)

        # [B, 2, D, T]
        # cmp_spec = torch.unsqueeze(cmp_spec, 4)
        # [B, 1, D, T, 2]
        #[B, 2, D, T] -> [B, T, D, 2]
        cmp_spec = torch.transpose(cmp_spec, 1, 3)
        s4_out1 = self.s4_model1(cmp_spec)
        cmp_mask1 = torch.tanh(s4_out1).unsqueeze(1).transpose(2, 3)
        # s4_out2 = self.s4_model2(s4_out1)
        # cmp_mask2 = torch.tanh(s4_out2).unsqueeze(1).transpose(2, 3)
        cmp_spec = cmp_spec.unsqueeze(1).transpose(2, 3)
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask1)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        # cmp_mask2 = cmp_mask2 + cmp_mask1
        # est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        # out_list.append(est_spec)
        # out_list.append(est_wav)
        # out_list.append(est_mask)
        return out_list

    def apply_mask(self, cmp_spec, cmp_mask):
        est_spec = torch.cat([
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 0]
            - cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 1],
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 1]
            + cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 0]
        ], 1)
        est_spec = torch.cat([est_spec[:, 0, :, :], est_spec[:, 1, :, :]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:, :, :, 0], cmp_mask[:, :, :, 1]], 1)

        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, cmp_mask

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, noisy, labels, out_list, mode='Mix'):
        if mode == 'SiSNR':
            count = 0
            while count < len(out_list):
                est_spec = out_list[count]
                count = count + 1
                est_wav = out_list[count]
                count = count + 1
                est_mask = out_list[count]
                count = count + 1
                if count != 3:
                    loss = self.loss_1layer(noisy, est_spec, est_wav, labels,
                                            est_mask, mode)
            return dict(sisnr=loss)

        elif mode == 'Mix':
            count = 0
            while count < len(out_list):
                est_spec = out_list[count]
                count = count + 1
                est_wav = out_list[count]
                count = count + 1
                est_mask = out_list[count]
                count = count + 1
                if count != 6:
                    amp_loss, phase_loss, SiSNR_loss = self.loss_1layer(
                        noisy, est_spec, est_wav, labels, est_mask, mode)
                    loss = amp_loss + phase_loss + SiSNR_loss
            return dict(loss=loss, amp_loss=amp_loss, phase_loss=phase_loss)

    def loss_1layer(self, noisy, est, est_wav, labels, cmp_mask, mode='Mix'):
        r""" Compute the loss by mode
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        """
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            return -si_snr(est_wav, labels)
        elif mode == 'Mix':

            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            SiSNR_loss = -si_snr(est_wav, labels)

            b, d, t = est.size()
            S = self.stft(labels)
            Sr = S[:, :self.feat_dim, :]
            Si = S[:, self.feat_dim:, :]
            Y = self.stft(noisy)
            Yr = Y[:, :self.feat_dim, :]
            Yi = Y[:, self.feat_dim:, :]
            Y_pow = Yr**2 + Yi**2
            gth_mask = torch.cat([(Sr * Yr + Si * Yi) / (Y_pow + 1e-8),
                                  (Si * Yr - Sr * Yi) / (Y_pow + 1e-8)], 1)
            gth_mask[gth_mask > 2] = 1
            gth_mask[gth_mask < -2] = -1
            amp_loss = F.mse_loss(gth_mask[:, :self.feat_dim, :],
                                  cmp_mask[:, :self.feat_dim, :]) * d
            phase_loss = F.mse_loss(gth_mask[:, self.feat_dim:, :],
                                    cmp_mask[:, self.feat_dim:, :]) * d
            return amp_loss, phase_loss, SiSNR_loss


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)
