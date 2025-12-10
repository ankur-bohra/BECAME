#!/usr/bin/env bash
# difference: the way to update covariance
GPUID=$1
REPEAT=$2

############################################ CIFAR100-10
# For Ours
CUDA_VISIBLE_DEVICES=$1 python -u main_swag.py \
    --gpuid $GPUID \
    --reg_coef 100 \
    --model_lr 0.1 \
    --head_lr 0.1 \
    --svd_lr 0.01 \
    --bn_lr 0.001 \
    --svd_thres 10 \
    --model_weight_decay 5e-4 \
    --agent_type svd_based \
    --agent_name svd_based \
    --dataset CIFAR100 \
    --repeat $REPEAT \
    --model_optimizer SGDSVD \
    --model_name resnet18 \
    --model_type resnet \
    --schedule 30 60 80 \
    --force_out_dim 0 \
    --first_split_size 10 \
    --other_split_size 10 \
    --batch_size 32 \
    --merge_list 80 \
    --merge_method swag-diag \
    --early_stop \
    --patience 3 \
    --decay_time 2 \
    --gamma 0.5 \
    --momentum 0.9 \
    --nesterov \
    --swag_start 15 \
    --swag_collect_freq 1