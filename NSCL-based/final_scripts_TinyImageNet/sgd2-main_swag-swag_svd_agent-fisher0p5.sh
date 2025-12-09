#!/usr/bin/env bash
# difference: the way to update covariance
GPUID=$1
REPEAT=$2

############################################ CIFAR100-10
# For Ours
CUDA_VISIBLE_DEVICES=$1 python -u main_swag.py \
    --gpuid $GPUID \
    --dataroot './data/tiny-imagenet-200' \
    --reg_coef 100 \
    --model_lr 0.05 \
    --head_lr 0.1 \
    --svd_lr 0.01 \
    --bn_lr 0.001 \
    --svd_thres 10 \
    --model_weight_decay 5e-4 \
    --agent_type svd_based \
    --agent_name svd_based \
    --dataset TinyImageNet \
    --repeat $REPEAT \
    --model_optimizer SGDSVD \
    --model_name resnet18 \
    --model_type resnet \
    --schedule 30 60 80 \
    --force_out_dim 0 \
    --first_split_size 8 \
    --other_split_size 8 \
    --batch_size 16 \
    --merge_list 80 \
    --merge_method fisher \
    --early_stop \
    --patience 3 \
    --decay_time 2 \
    --gamma 0.5 \
    --momentum 0.9 \
    --nesterov