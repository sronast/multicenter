#!/bin/bash

python train.py --run_name 'train_scratch_no_crop_lr_step' \
                --project 'ETIS' \
                --device 1 \
                --lr 0.0002 \
                --total_iters 20000 \
                --eval_frq 200 \
                --print_frq 100 \
                --enc_weight '' \
                --dec_weight '' \
                --save_frq 2000 \
                # --data 'polyp' \
