#!/bin/bash
for nf_layers in 20 40 50
do
    for z_dim in 3 5 10
    do
        echo "Trying  --nf_layers=$nf_layers --z_dim=$z_dim"
        time python -u main.py --nf_layers=$nf_layers --z_dim=$z_dim  --dataset='mypkg_DSN_1k' --max_epoch=20 --batch_size=1400 --test_batch_size=1400 2>&1 | tee run_z${z_dim}_nf${nf_layers}.txt
    done
done
