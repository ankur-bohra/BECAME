#!/usr/bin/env bash
# filepath: /mnt/c/Users/ankur/Programs/BML/BECAME/NSCL-based/scripts/run_some_sgd.sh
GPUID=0
REPEAT=1

CUDA_VISIBLE_DEVICES=$1 python -u main_swag.py --gpuid $GPUID --reg_coef 100 \
    --model_lr 0.1 \
    --head_lr 0.1 \
    --svd_lr 0.01 \
    --bn_lr 0.01 \
    --svd_thres 10 --model_weight_decay 5e-4 \
    --agent_type svd_based --agent_name svd_based \
    --dataset CIFAR100 --repeat $REPEAT \
    --model_optimizer SGDSVD --model_name resnet18 --model_type resnet \
    --schedule 30 60 80 --force_out_dim 0 \
    --first_split_size 10 --other_split_size 10 --batch_size 32 \
    --merge_list 80 --early_stop --patience 3 \
    --decay_time 2 --gamma 0.5 --momentum 0.9 --nesterov \
    --merge_method swag --swag_start 40 --swag_collect_freq 1