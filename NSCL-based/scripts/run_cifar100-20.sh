#!/usr/bin/env bash
# difference: the way to update covariance
GPUID=$1
REPEAT=5

# ############################################ CIFAR100-10
# # For baseline
# CUDA_VISIBLE_DEVICES=$1 python -u main.py --gpuid $GPUID --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --repeat $REPEAT  --model_optimizer Adam --model_name resnet18 --model_type resnet --schedule 30 60 80 --force_out_dim 0 --first_split_size 10 --other_split_size 10  --batch_size 32 
# # For Ours
# CUDA_VISIBLE_DEVICES=$1 python -u main.py --gpuid $GPUID --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --repeat $REPEAT  --model_optimizer Adam --model_name resnet18 --model_type resnet --schedule 30 60 80 --force_out_dim 0 --first_split_size 10 --other_split_size 10  --batch_size 32 --merge_list 80 --fisher_m  --early_stop --patience 3 --decay_time 2 --lr_decay 0.5 

# ############################################ CIFA100-20
# For baseline
# python -u main.py --schedule 30 60 80 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 30 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0  --first_split_size 5 --other_split_size 5  --batch_size 16 --model_name resnet18 --model_type resnet
# For Ours
python -u main.py --schedule 30 60 80 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 30 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0  --first_split_size 5 --other_split_size 5  --batch_size 16 --model_name resnet18 --model_type resnet --merge_list 80 --fisher_m  --early_stop --patience 3 --decay_time 2 --lr_decay 0.5 

# # ############################################ TinyImageNet
# # For baseline
# CUDA_VISIBLE_DEVICES=$1 python -u main.py --gpuid $1 --dataroot './data/tiny-imagenet-200' --reg_coef 100 --model_lr 5e-5 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --gamma 0.5  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset TinyImageNet --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0 --first_split_size 8 --other_split_size 8  --batch_size 16 --model_name resnet18 --model_type resnet --schedule 30 60 80
# # For Ours
# CUDA_VISIBLE_DEVICES=$1 python -u main.py --gpuid $1 --dataroot './data/tiny-imagenet-200' --reg_coef 100 --model_lr 5e-5 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --gamma 0.5  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset TinyImageNet --repeat $REPEAT  --model_optimizer Adam --force_out_dim 0 --first_split_size 8 --other_split_size 8  --batch_size 16 --model_name resnet18 --model_type resnet --schedule 30 60 80 --merge_list 80 --fisher_m  --early_stop --patience 3 --decay_time 2 --lr_decay 0.5 
