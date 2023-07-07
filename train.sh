#! /bin/bash
for rnn_size in 64 256
do
CUDA_VISIBLE_DEVICES=1 python baseline_solution/sdnn_delays/train_sdnn_rnn.py -path /mnt/data/projects/neuralmorphic-computing/audio-denoise/IntelNeuromorphicDNSChallenge/data/datasets_fullband/ --step retrain --thx 0.1 --thh 0.1 -b 32 --rnn_size ${rnn_size} --fc_extra_size 256 -exp rnn_${rnn_size}
done
