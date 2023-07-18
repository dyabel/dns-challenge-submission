# import os

# from datasets import load_dataset

# from modelscope.metainfo import Trainers
# from modelscope.msdatasets import MsDataset
# from modelscope.trainers import build_trainer
# from modelscope.utils.audio.audio_utils import to_segment

# tmp_dir = './checkpoint'
# if not os.path.exists(tmp_dir):
#     os.makedirs(tmp_dir)

# hf_ds = load_dataset(
#     '/your_local_path/ICASSP_2021_DNS_Challenge',
#     'train',
#     split='train')
# mapped_ds = hf_ds.map(
#     to_segment,
#     remove_columns=['duration'],
#     num_proc=8,
#     batched=True,
#     batch_size=36)
# mapped_ds = mapped_ds.train_test_split(test_size=3000)
# mapped_ds = mapped_ds.shuffle()
# dataset = MsDataset.from_hf_dataset(mapped_ds)

# kwargs = dict(
#     model='your_local_path/speech_frcrn_ans_cirm_16k',
#     train_dataset=dataset['train'],
#     eval_dataset=dataset['test'],
#     work_dir=tmp_dir)
# trainer = build_trainer(
#     Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)
# trainer.train()


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav',
    output_path='output.wav')