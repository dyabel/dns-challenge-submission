#! /bin/bash
for rnn_size in 64
do
CUDA_VISIBLE_DEVICES=0 python baseline_solution/sdnn_delays/train_sdnn_rnn.py -path /home/dy/datasets/IntelDNS/ --step retrain --thx 0.1 --thh 0.1 -b 32 --rnn_size ${rnn_size} --fc_extra_size 256 -exp rnn_${rnn_size}
done
