import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict

import torch.utils
import torch.utils.data
from dataloaders.datasetGen import SplitGen, PermutedGen
from utils.utils import factory
import random
from torch.autograd import Variable
import copy
from datetime import datetime

AK_Star = {
    'CIFAR100_10_10': np.array([88.2, 87.5, 89.5, 89.9, 89.6, 92.0, 89.0, 88.7, 91.0, 94.5]),
    'CIFAR100_5_5': np.array([88.6, 90.4, 92.8, 90.8, 95.8, 90.8, 93.8, 94.6, 90.6, 95.6, 96.4, 96.4, 95.6, 93.4, 89.2, 94.0, 95.4, 93.8, 96.2, 97.2]),
    'TinyImageNet_8_8': np.array([71.5, 83.5, 84.0, 61.25, 78.0, 83.25, 74.25, 87.25, 79.5, 79.5, 82.75, 86.75, 82.25, 85.75, 86.0, 84.5, 78.25, 78.25, 86.5, 84.25, 91.0, 83.5, 73.0, 82.25, 81.25])
}
PATIENCE = 5


def run(args, results_dir):

    # Prepare dataloaders
    train_dataset, val_dataset = factory(
        'dataloaders', 'base', args.dataset)(args.dataroot, args.train_aug)
    if args.n_permutation > 0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)

    # Prepare the Agent (model)
    dataset_name = args.dataset + \
        '_{}'.format(args.first_split_size) + \
        '_{}'.format(args.other_split_size)
    exp_name = f'MM_{args.model_name}_{args.dataset.lower()}-{args.first_split_size}-{args.other_split_size}'
    if args.is_test:
        exp_name = 'test_' + exp_name
    if args.suffix:
        exp_name += f'_{args.suffix}'
    agent_config = {'model_lr': args.model_lr, 'momentum': args.momentum, 'model_weight_decay': args.model_weight_decay, 'nesterov': args.nesterov,
                    'schedule': args.schedule,
                    'model_type': args.model_type, 'model_name': args.model_name, 'model_weights': args.model_weights,
                    'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'model_optimizer': args.model_optimizer,
                    'print_freq': args.print_freq,
                    'gpu': True if args.gpuid[0] >= 0 else False,
                    'with_head': args.with_head,
                    'reset_model_opt': args.reset_model_opt,
                    'reg_coef': args.reg_coef,
                    'head_lr': args.head_lr,
                    'svd_lr': args.svd_lr,
                    'bn_lr': args.bn_lr,
                    'svd_thres': args.svd_thres,
                    'gamma': args.gamma,
                    'dataset_name': dataset_name,
                    'batch_size': args.batch_size,
                    'exp_name': exp_name,
                    'merge_list': args.merge_list,
                    'distill_ep': args.distill,
                    'multi_task': args.multi_task,
                    'class_icr': args.incremental_class,
                    'fisher_m': args.fisher_m,
                    'do_analysis': args.do_analysis,
                    'val_freq': args.val_freq,
                    'connector': args.connector_base,
                    'early_stop': args.early_stop,
                    'patience': args.patience,
                    'lr_decay': args.lr_decay,
                    'decay_time': args.decay_time,
                    }
    
    agent = factory('svd_agent', args.agent_type,
                    args.agent_name)(agent_config)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    
    # Limit number of tasks if specified
    if args.num_tasks > 0:
        task_names = task_names[:args.num_tasks]
    
    agent.task_names = task_names
    print('Task order:', task_names)

    acc_table_np = np.zeros((len(task_names), len(task_names)))
    
    for i in range(len(task_names)):
        train_name = task_names[i]
        print('======================', train_name,
              '=======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)

        agent.val_loader_buffer.append(val_loader)
        agent.trained_tasks.append(train_name)

        if args.incremental_class:
            agent.add_valid_output_dim(task_output_space[train_name])

        if i > 0:
            agent.pre_model = copy.deepcopy(agent.model).cuda()

            # stage 1: stability container
            agent.model_optimizer.switch = True
            agent.train_task(train_loader, val_loader, args.schedule[-1], early_stop=args.early_stop)
            agent.m_model = copy.deepcopy(agent.model.state_dict())
            
            # stage 1.2 for Connector
            if args.connector_base:
                agent.model.load_state_dict(agent.pre_model.state_dict())
                print('Classifier Optimizer is reset!')
                agent.init_model_optimizer()
                agent.model_optimizer.switch = False
                agent.is_distill = True  
                agent.fisher_m = False
                agent.merge_list = args.schedule[-1:]
                agent.train_task(train_loader, val_loader, args.schedule[-1])
                agent.m_model = copy.deepcopy(agent.model.state_dict())
                agent.is_distill = False
                agent.fisher_m = args.fisher_m
                agent.merge_list = args.merge_list

            # stage 2: stability-plasticity trade-off
            s2_epochs = 0
            if args.merge_list[-1] > 0:
                s2_epochs = args.merge_list[-1]
            if args.distill>0:
                s2_epochs = args.distill if s2_epochs==0 else s2_epochs
                agent.is_distill = True
            if s2_epochs > 0:
                print('Classifier Optimizer is reset!')
                agent.init_model_optimizer()
                agent.model_optimizer.switch = False
                agent.train_task(train_loader, val_loader, s2_epochs, early_stop=args.early_stop)
        else:
            agent.train_task(train_loader, val_loader, args.schedule[-1], early_stop=args.early_stop)
            
        # update fisher matrix
        agent.update_fisher_matrix_diag(train_loader)

        # AfterTrain
        agent.after_train(train_loader)
        
        torch.cuda.empty_cache()
            
        for j in range(i + 1):
                
            val_name = task_names[j]

            # print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[
                val_name]
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
            acc_table_np[i, j], _ = agent.validation(val_loader)
            print('Accuracy on task {} after learning task {}: {:.2f}%'.format(val_name, task_names[i], acc_table_np[i, j]*100))

            # print("**************************************************")
            
    np.save(os.path.join(results_dir, 'tables', 'acc_table.npy'), acc_table_np)
    
    # save the model parameters
    torch.save(agent.model.state_dict(), os.path.join(results_dir, 'state_dict.pth'))
        
    return acc_table_np, task_names


def str2bool(v):
    return v.lower() in ('true', '1')

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='resnet',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help="The name of actual model for the backbone")

    parser.add_argument('--force_out_dim', type=int, default=0,
                        help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str,
                        default='svd_based', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str,
                        default='svd_based', help="The class name of agent")

    parser.add_argument('--model_optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")

    parser.add_argument('--dataroot', type=str, default='./data/',
                        help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        help="CIFAR10|CIFAR100|TinyImageNet|MiniImageNet")
    parser.add_argument('--n_permutation', type=int, default=0,
                        help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=10)
    parser.add_argument('--other_split_size', type=int, default=10)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")  # class:we need to know specific class,other:no need to know specific class
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_false',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--workers', type=int, default=0,
                        help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--model_lr', type=float,
                        default=0.0005, help="Classifier Learning rate")
    parser.add_argument('--head_lr', type=float,
                        default=0.0005, help="Classifier Learning rate")
    parser.add_argument('--svd_lr', type=float, default=0.0005,
                        help="Classifier Learning rate")
    parser.add_argument('--bn_lr', type=float, default=0.0005,
                        help="Classifier Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Learning rate decay")
    parser.add_argument('--svd_thres', type=float,
                        default=1.0, help='reserve eigenvector')

    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--nesterov', default=False, action='store_true', 
                        help="Use Nesterov momentum for SGD")

    parser.add_argument('--model_weight_decay',
                        type=float, default=1e-5)  # 1e-4

    parser.add_argument('--schedule', nargs="+", type=int, default=[1],
                        help="epoch ")

    parser.add_argument('--print_freq', type=float, default=10,
                        help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")

    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1,
                        help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")

    parser.add_argument('--with_head', dest='with_head', default=False, action='store_true',
                        help="whether constraining head")
    parser.add_argument('--reset_model_opt', dest='reset_model_opt', default=True, action='store_true',
                        help="whether reset optimizer for model at the start of training each tasks")
    parser.add_argument('--reg_coef', type=float, default=100,
                        help="The coefficient for ewc reg")
    
    parser.add_argument('--merge_list', nargs="+", type=int, default=[-1])
    parser.add_argument('--is_test', default=False, action='store_true',
                        help="whether to run only for test code")
    parser.add_argument('--distill', type=int, default=0)
    parser.add_argument('--multi_task', default=False, action='store_true',
                        help='whether to use multi-task learning, training all the tasks at the same time')
    parser.add_argument('--suffix', type=str, default='', help='the suffix for the results dir')
    parser.add_argument('--fisher_m', default=False, action='store_true', help='whether to use fisher matrix')
    parser.add_argument('--do_analysis', default=False, action='store_true', help='whether to do merge analysis')
    parser.add_argument('--val_freq', type=int, default=10, help='the frequency of validation')
    parser.add_argument('--connector_base', default=False, action='store_true', help='whether to use connector')
    
    parser.add_argument('--patience', type=int, default=PATIENCE, help='the patience for early stopping')
    parser.add_argument('--early_stop', default=False, action='store_true', help='whether to use early stopping')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--decay_time', type=int, default=3)
    parser.add_argument('--num_tasks', type=int, default=-1,
                        help="Number of tasks to run. -1 means all tasks.")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    avg_final_acc = np.zeros(args.repeat)
    final_bwt = np.zeros(args.repeat)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid[0])
    
    dataset_name = args.dataset + \
        '_{}'.format(args.first_split_size) + \
        '_{}'.format(args.other_split_size)
    ak_star = AK_Star.get(dataset_name)

    for r in range(args.repeat):
        results_dir = os.path.join(f'{args.model_name}_{args.dataset.lower()}-{args.first_split_size}-{args.other_split_size}', f"{datetime.now().strftime('%Y%m%d%H%M%S')}-s{r}")
        if args.is_test :
            results_dir = os.path.join('results', 'test', results_dir)
        else:
            results_dir = os.path.join('results', results_dir)
        print(f"Results will be saved in {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            os.makedirs(os.path.join(results_dir, 'tables'))
            
        # save all args into a yml file
        with open(os.path.join(results_dir, 'args.yml'), 'w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
        
        # Seed
        SEED = r
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        
        # enable the model could have same output for specific data and params
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        # Run the experiment
        acc_table_np, task_names = run(args, results_dir)
        
        # Calculate average performance across tasks
        # Customize this part for a different performance metric
        avg_acc_history = np.zeros(len(task_names))
        bwt_history = np.zeros(len(task_names))
        for i in range(len(task_names)):
            train_name = task_names[i]
            cls_acc_sum = 0
            backward_transfer = 0
            avg_acc_history[i] = acc_table_np[i, :i+1].mean()
            bwt_history[i] = np.array([acc_table_np[i, j] - acc_table_np[j, j] for j in range(i)]).mean()
            print('Task', train_name, 'average acc:', avg_acc_history[i])
            print('Task', train_name, 'backward transfer:', bwt_history[i])

        # Gather the final avg accuracy
        avg_final_acc[r] = avg_acc_history[-1]
        final_bwt[r] = bwt_history[-1]
    
        # results save 
        cur_r_bwt = np.array([acc_table_np[-1, i] - acc_table_np[i, i] for i in range(len(task_names)-1)])        
                        
        # Print the summary so far
        print('===Summary of experiment repeats:',
              r + 1, '/', args.repeat, '===')
        print('The last avg acc of all repeats:', avg_final_acc)
        print('The last bwt of all repeats:', final_bwt)
        print(f'acc mean: {avg_final_acc.mean():.2f}, acc std: {avg_final_acc.std():.2f}')
        print(f'bwt mean: {final_bwt.mean():.2f}, bwt std: {final_bwt.std():.2f}')