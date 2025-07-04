import time
import torch
import torch.nn as nn
from types import MethodType
from tensorboardX import SummaryWriter
from datetime import datetime


from utils.metric import accumulate_acc, AverageMeter, Timer
from utils.utils import count_parameter, factory
import optim

import copy
import numpy as np
from collections import OrderedDict
import os

import re
from tqdm import tqdm

class Agent(nn.Module):
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super().__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        self.log(agent_config)
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(
            self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
        self.num_task = len(self.config['out_dim']) if len(
            self.config['out_dim']) > 1 else None
        self.model = self.create_model()
        self.pre_model = self.create_model()

        self.criterion_fn = nn.CrossEntropyLoss()

        # Default: 'ALL' means all output nodes are active # Set a interger here for the incremental class scenario
        self.valid_out_dim = 'ALL'

        self.clf_param_num = count_parameter(self.model)
        self.task_count = 0
        self.reg_step = 0
        self.summarywritter = SummaryWriter(
            './log/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if self.config['gpu']:
            self.model = self.model.cuda()
            self.criterion_fn = self.criterion_fn.cuda()
        self.log('#param of model:{}'.format(self.clf_param_num))

        self.reset_model_optimizer = agent_config['reset_model_opt']
        self.dataset_name = agent_config['dataset_name']
        
        self.val_loader_buffer = []
        self.merge_list = agent_config['merge_list']
        self.m_model = None
        self.t = 0
        self.epoch_cnt = 0
        self.is_distill = False
        self.trained_tasks = []
        self.device = 'cuda' if self.config['gpu'] else 'cpu'
        self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.named_parameters() if p.requires_grad and not re.match(r'^last', n)}
        self.fisher_m = self.config['fisher_m']
        self.aft_one_acc = []


    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = factory('models', cfg['model_type'], cfg['model_name'])()
        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features  # input_dim

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim, bias=True)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            # all
            outputs['ALL'] = torch.cat([outputs[task] for task in outputs.keys()], dim=1)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def init_model_optimizer(self):
        model_optimizer_arg = {'params': self.model.parameters(),
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.1)

    def train_task(self, train_loader, val_loader=None):
        raise NotImplementedError

    def train_epoch(self, train_loader, epoch, count_cls_step):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        end = time.time()
        for i, (inputs, target, task) in enumerate(train_loader):
            # print("*"*100)
            # print(inputs.mean())
            count_cls_step += 1
            data_time.update(time.time() - end)  # measure data loading time

            if self.config['gpu']:
                inputs = inputs.cuda()
                target = target.cuda()
            output = self.model.forward(inputs)
            loss = self.criterion(output, target, task)
            
            # loss for knowledge distillation
            # only distill when self.is_distill>0
            if self.task_count > 0 and self.model_optimizer.switch == False and self.is_distill:
                fa = self.model.features_last(inputs)
                fb = self.pre_model.features_last(inputs)
                loss += (fa - fb).pow(2).mean()

            acc = accumulate_acc(output, target, task, acc)

            self.model_optimizer.zero_grad()
            self.model_scheduler.step(epoch)

            loss.backward()
            self.model_optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss, inputs.size(0))

        return losses.avg, acc.avg

    def train_model(self, train_loader, val_loader=None, epochs=1, early_stop=False):
        count_cls_step = 0
        cfg = self.config
        self.t = len(self.val_loader_buffer)
        
        # record for early stop
        if early_stop:
            best_train_loss = 1e10
            patience = cfg["patience"]
            decay_time = cfg["decay_time"]
        
        stop_n_merge = False

        for epoch in tqdm(range(epochs)):

            self.model.train()
            losses, acc = self.train_epoch(train_loader, epoch, count_cls_step)
            
            if early_stop:
                if losses < best_train_loss:
                    best_train_loss = losses
                    patience = cfg["patience"]
                else:
                    patience -= 1
                    if patience == 0:
                        if decay_time > 0:
                            # adjust the learning rate
                            for param_group in self.model_optimizer.param_groups:
                                param_group['lr'] = param_group['lr'] * cfg['lr_decay']
                            print('Learning rate decay by favtor:', cfg['lr_decay'])
                            decay_time -= 1
                            patience = cfg["patience"]
                        else:
                            stop_n_merge = True
                        
            
            # merge the old model during training
            if self.t>1 and self.model_optimizer.switch==False and (epoch+1 in self.merge_list or stop_n_merge):
                self.merge_n_analysis(train_loader, epoch)
                
            if epoch==0 and self.model_optimizer.switch==True and val_loader!=None:
                acc, _ = self.validation(val_loader)
                self.aft_one_acc.append(acc)
            
            if stop_n_merge:
                print('Stop training!')
                break
            
            self.epoch_cnt += 1
            

    def validation(self, dataloader, is_print=True):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        val_acc = AverageMeter()
        losses = AverageMeter()
        batch_timer.tic()

        # self.hypermodel.eval()
        self.model.eval()

        for i, (inputs, target, task) in enumerate(dataloader):

            if self.config['gpu']:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()

                    output = self.model.forward(inputs)
                    loss = self.criterion(
                        output, target, task, regularization=False)
            losses.update(loss, inputs.size(0))
            for t in output.keys():
                output[t] = output[t].detach()
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            val_acc = accumulate_acc(output, target, task, val_acc)

        return val_acc.avg, losses.avg

    def criterion(self, preds, targets, tasks, regularization=True):
        loss = self.cross_entropy(preds, targets, tasks)
        if regularization and len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = self.reg_loss()
            loss += self.config['reg_coef'] * reg_loss
        return loss

    def cross_entropy(self, preds, targets, tasks):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                # The index of inputs that matched specific task
                inds = [i for i in range(len(tasks)) if tasks[i] == t]
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    # restore the loss from average
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)
                
            # Average the total loss by the mini-batch size
            loss /= len(targets)
        else:
            targets += np.sum(self.config['out_dim'][:self.task_count])
            if isinstance(self.valid_out_dim,
                          int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['ALL'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:',
                 self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:',
                 self.valid_out_dim)
        return self.valid_out_dim    
    

    # threshold compare
    def thres_compare(self, last_loss, losses, base_slope, cfg):
        thres = (losses-last_loss)/(pow(self.t-1, cfg['dm_pow'])*base_slope)
        print('Thres:', thres, 'Base slope:', base_slope, 'diff:', losses-last_loss, 'coef:', pow(self.t-1, cfg['dm_pow']))
        if thres >= cfg['dm_thres_min']:
            return True
        else:
            return False
        

    # model merge
    def merge_n_analysis(self, train_loader, epoch):
        old_model = self.m_model
        cur_model = self.model.state_dict()
        
        ans = old_model
        beta = 1/self.t
        # fisher matrix merge or not
        if not self.fisher_m:
            for k in old_model.keys():
                ans[k] = old_model[k]*(1-beta) + cur_model[k]*beta
        else:
            curr_fisher = self.compute_fisher_matrix_diag(train_loader) 
            # calculate fisher merge coef
            mole_sum = 0
            demo_sum = 0
            for k in self.fisher.keys():
                mole_sum += (curr_fisher[k]*(cur_model[k]-old_model[k]).pow(2)).sum().item()
                demo_sum += ((self.fisher[k]+curr_fisher[k])*(cur_model[k]-old_model[k]).pow(2)).sum().item()
            coef = mole_sum/demo_sum
            # merge model
            for k, p in old_model.items():
                ans[k] = old_model[k]*(1-coef) + cur_model[k]*coef
                        
        self.model.load_state_dict(ans)
             
                
    # test without task id
    def agnostic_validation(self, dataloader):
        # This function doesn't distinguish tasks.
        val_acc = AverageMeter()
        self.model.eval()

        for i, (inputs, target, task) in enumerate(dataloader):

            if self.config['gpu']:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()
                    output = self.model.forward(inputs)
            for t in output.keys():
                output[t] = output[t].detach()
            target_offest = True
            for k, t in enumerate(self.trained_tasks):
                if k == 0:
                    agnostic_output = output[t]
                else:
                    agnostic_output = torch.cat((agnostic_output, output[t]), dim=1)
                if target_offest:
                    if t == task[0]:
                        target_offest = False
                        continue
                    target = target + self.config['out_dim'][t]
            for t in self.trained_tasks:
                output[t] = agnostic_output
            val_acc = accumulate_acc(output, target, task, val_acc)
        print(f' * Val Acc CIL {val_acc.avg:.3f}')
        return val_acc.avg, None
    
    
    # [FM] calculate the fisher information matrix
    def compute_fisher_matrix_diag(self, train_loader):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.named_parameters()
                  if p.requires_grad and not re.match(r'^last', n)}
        # Do forward and backward pass to compute the fisher information
        self.model.train()
        for i, (inputs, target, task) in enumerate(train_loader):
            if self.config['gpu']:
                inputs = inputs.cuda()
                target = target.cuda()
            output = self.model.forward(inputs)
            pred = output[self.trained_tasks[-1]].argmax(dim=1).flatten()
            loss = self.criterion(output, pred, task)
            self.model_optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in self.fisher.keys():
                    fisher[n] += p.grad.pow(2) * len(target)
        n_samples = len(train_loader.dataset)
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher
    
    # [FM]update fisher matrix
    def update_fisher_matrix_diag(self, train_loader):
        t = self.t
        """Runs after training all the epochs of the task (after the train session)"""
        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(train_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            self.fisher[n] += curr_fisher[n]
        print(f'Update Fisher Matrix for task {self.trained_tasks[-1]}')
        return curr_fisher