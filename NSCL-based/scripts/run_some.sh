#!/usr/bin/env bash
# difference: the way to update covariance
GPUID=0
REPEAT=1

############################################ CIFAR100-10
# For Ours
CUDA_VISIBLE_DEVICES=$1 python -u main.py --gpuid $GPUID --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --repeat $REPEAT  --model_optimizer Adam --model_name resnet18 --model_type resnet --schedule 30 60 80 --force_out_dim 0 --first_split_size 10 --other_split_size 10  --batch_size 32 --merge_list 80 --fisher_m  --early_stop --patience 3 --decay_time 2 --lr_decay 0.5 --num_tasks 1