import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = f'./ckpt'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# Loading dataset
hf_ds = MsDataset.load(
    'ICASSP_2021_DNS_Challenge', split='test').to_hf_dataset()
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    batched=True,
    batch_size=36)
mapped_ds = mapped_ds.train_test_split(test_size=150)
# Use below code for real large data training
# hf_ds = load_dataset(
#     '/mnt/data/projects/neuralmorphic-computing/audio-denoise/IntelNeuromorphicDNSChallenge/data/datasets_fullband/',
#     'train',
#     split='train')
# mapped_ds = hf_ds.map(
#     to_segment,
#     remove_columns=['duration'],
#     num_proc=8,
#     batched=True,
#     batch_size=36)
# mapped_ds = mapped_ds.train_test_split(test_size=3000)
# End of comment

mapped_ds = mapped_ds.shuffle()
dataset = MsDataset.from_hf_dataset(mapped_ds)

kwargs = dict(
    model='damo/speech_frcrn_ans_cirm_16k',
    # model_revision='beta',
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    work_dir=tmp_dir)
trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)
trainer.train()