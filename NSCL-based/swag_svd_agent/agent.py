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

from .swag import SWAG
from .swag import utils

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

        # self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.named_parameters() if p.requires_grad and not re.match(r'^last', n)}
        # self.fisher_m = self.config['fisher_m']
        self.merge_method = self.config['merge_method']
        assert self.merge_method in ['fisher', 'weighted_average', 'swag'], "merge method not supported!"
        if self.merge_method == 'fisher':
            self.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.named_parameters() if p.requires_grad and not re.match(r'^last', n) and self.model._parameters[n] is not None}
        elif self.merge_method == 'weighted_average':
            pass
        elif self.merge_method == 'swag':
            # raise Exception("SWAG not taking 'not last' parameters!!.")
            # Σ = 0.5 * (diag(var) + D^/sqrt(k-1) (D^)^T/sqrt(k-1))
            # Since D^ (D^)^T isn't diagonal, can't store separately for params
            # Find total number of parameters in model
            # n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # Woodbury identity: 
            # Σ^-1 = 2 * (diag(var)^-1 - diag(var)^-1 D^ (I + D^T diag(var)^-1 D^)^-1 D^T diag(var)^-1)
            #      = 2 * (inv_diag - D~ M_inv D~^T)
            # Add over tasks:
            # inv_diag -> sum over tasks
            # D~ M_inv D~^T -> can't store sum directly without storing full product
            # self.sum_inv_diag = torch.zeros(n_params).to(self.device)
            # self.low_rank_terms = []  # List of (D~, M_inv) tuples
            names = []
            for n, p in self.model.named_parameters():
                if p.requires_grad and not re.match(r'^last', n):  # Uncomment if filtering head
                    module_name, param_name = n.rsplit('.', 1)
                    module = self.model.get_submodule(module_name)
                    if module._parameters[param_name] is not None:
                        names.append(n)
            self.sum_inv_diag_map = {
                n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if n in names
            }
            # print("sum_inv_diag_map keys:", list(self.sum_inv_diag_map.keys()))
            self.low_rank_history = []
            self.swag_model = None
            self.current_inv_diag_map = None

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
    
    def create_swag_model(self, param_names=[]):
        """
        Follows the same structure as create_model but returns a SWAG model.
        - Creates the same base model architecture.
        - Doesn't load pre-trained weights.
        - Only tracks parameters specified in param_names.
        """
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model_class = factory('models', cfg['model_type'], cfg['model_name'])
        swag_model = SWAG(model_class, no_cov_mat=False, max_num_models=cfg['swag_max_num_models'], restrict_params=param_names)
        model = swag_model.base
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
        # if cfg['model_weights'] is not None:
        #     print('=> Load model weights:', cfg['model_weights'])
        #     model_state = torch.load(cfg['model_weights'],
        #                              map_location=lambda storage, loc: storage)  # Load to CPU.
        #     model.load_state_dict(model_state)
        #     print('=> Load Done')
        return swag_model

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
        #! Change schedule for SWAG?
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
        if self.merge_method == 'swag':
            # A new "path" is starting, buffer history should be forgotten
            self.swag_model = self.create_swag_model(param_names=list(self.sum_inv_diag_map.keys()))  # Equivalent to resetting buffers

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

            if (
                self.merge_method == 'swag'
                and epoch + 1 > self.config['swag_start']
                and (epoch + 1) % self.config['swag_collect_freq'] == 0
            ):
                self.swag_model.collect_model(self.model)
            
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
                if self.merge_method == 'swag':
                    # "path" has ended, take SWA mean
                    self.swag_model.sample(0.0)
                    self.swag_model.cuda()
                    utils.bn_update(train_loader, self.swag_model)  # Update BN stats for the averaged model
                    self.swag_model.cpu()
                    # Copy params to active model
                    for param in self.sum_inv_diag_map.keys():
                        module_name, param_name = param.rsplit('.', 1)
                        swag_module = self.swag_model.base.get_submodule(module_name)
                        model_module = self.model.get_submodule(module_name)
                        mean = swag_module.__getattr__("%s_mean" % param_name)
                        model_module.__getattr__("%s" % param_name).data.copy_(mean.data)
                    # Now, SWAG mean is available to merge in self.model, and SWAG buffers
                    # will be used from self.swag_model for precision

                    # merge_n_analysis will assume the current precision matrix has been
                    # computed. Precision must be updated here.
                    # If this early stop is so early that no SWAG models were collected,
                    # collect one model to fill buffers
                    if self.swag_model.n_models.item() == 0:
                        self.swag_model.collect_model(self.model)
                    self.update_precision()
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
        if self.merge_method == 'weighted_average':
            for k in old_model.keys():
                ans[k] = old_model[k]*(1-beta) + cur_model[k]*beta
        elif self.merge_method == 'fisher':
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
        elif self.merge_method == 'swag':
            # calculate merge coef
            # coeff = delta^T @ F_t @ delta / delta^T @ (F_t + precision) @ delta
            delta_map = {k: cur_model[k] - old_model[k] for k in self.sum_inv_diag_map}
            # At this point the precision matrix for this task should already have been
            # computed
            # print("Computing SWAG merge numerator...")
            # print("diag_map:", self.current_inv_diag_map)
            numerator = self.compute_precision_quadratic_form_woodbury(
                delta_theta_map=delta_map,
                diag_map=self.current_inv_diag_map,
                low_rank_list=[self.low_rank_history[-1]]
            )
            # print("Computing SWAG merge denominator...")
            denominator = self.compute_precision_quadratic_form_woodbury(
                delta_theta_map=delta_map,
                diag_map=self.sum_inv_diag_map,
                low_rank_list=self.low_rank_history
            )
            # if torch.isclose(torch.tensor(numerator), torch.tensor(0.0)):
            #     print("WARNING: Numerator close to 0 in SWAG merge coef calculation.")
            # if torch.isclose(torch.tensor(denominator), torch.tensor(0.0)):
            #     print("WARNING: Denominator close to 0 in SWAG merge coef calculation.")
            # if torch.isclose(torch.tensor(denominator), torch.tensor(0.0)) or torch.isclose(torch.tensor(numerator), torch.tensor(0.0)):
            #     exit(-1)
            coef = numerator / denominator
            for k in old_model.keys():
                ans[k] = old_model[k]*(1-coef) + cur_model[k]*coef
                if torch.isnan(ans[k]).any():
                    print(f"NaN detected in parameter {k} during SWAG merge!")
                    exit(-1)
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
    # Required for hessian of loss function
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
    # Used for approximation of precision matrix
    # def update_precision(self, train_loader):
    #     """Runs after training all the epochs of the task (after the train session)"""
    #     self.swag_model.collect_model(self.model)  # Update statistic buffers
    #     _, var_list, cov_mat_sqrt_list = self.swag_model.generate_mean_var_covar()
    #     var_diag = torch.cat([v.view(-1) for v in var_list])
    #     # Σ^-1 = 2 * (diag(var)^-1 - diag(var)^-1 D^ (I + D^T diag(var)^-1 D^)^-1 D^T diag(var)^-1)
    #     #      = 2 * (inv_diag - D~ M_inv D~^T)
    #     inv_diag = 1/var_diag
    #     self.sum_inv_diag += inv_diag
    #     D_hat = torch.cat(cov_mat_sqrt_list, dim=1).t()  # k x n_params -> n_params x k
    #     D_tilde = D_hat * inv_diag.unsqueeze(1)  # Scale rows by inv_diag
    #     M = torch.eye(D_tilde.size(1)).to(self.device) + D_hat.t() @ D_tilde
    #     M_inv = torch.linalg.inv(M)
    #     self.low_rank_terms.append((D_tilde, M_inv))
    
    def update_precision(self):
        print("Updating Precision Matrices...")
        # self.swag_model.collect_model(self.model)
        swag_buffers = {}
        for (buffer_name, buffer) in self.swag_model.base.named_buffers():
            stripped_name = buffer_name.replace('base.', '')
            swag_buffers[stripped_name] = buffer
        rank = 0
        for key in swag_buffers.keys():
            if re.match(r'.*_cov_mat_sqrt', key):
                rank = swag_buffers[key].size(0)
                break
        else:
            print("No covariance buffers found in SWAG model!")
            print("Buffers:", list(swag_buffers.keys()))
            exit(-1)
        if rank == 0:
            print("Covariance buffers were found but have rank 0!")
            print("Buffers:", list(swag_buffers.keys()))
            print("Sample buffer:", swag_buffers[key])
            exit(-1)
        M_global = torch.eye(rank, device=self.device)
        
        current_task_D_tilde_map = {}
        # Store current task's diagonal separately for current precision computation
        # The low rank history is already stored separately
        self.current_inv_diag_map = {}
        
        for name in self.sum_inv_diag_map.keys():
            # ... (Buffer retrieval logic same as previous) ...
            mean_key = f"{name}_mean"
            sq_mean_key = f"{name}_sq_mean"
            cov_key = f"{name}_cov_mat_sqrt"
            
            if cov_key in swag_buffers:
                # print(f"Precision processing parameter: {name}")
                D_hat = swag_buffers[cov_key].t().to(self.device)
                mean = swag_buffers[mean_key].to(self.device)
                sq_mean = swag_buffers[sq_mean_key].to(self.device)
                
                var = torch.clamp(sq_mean - mean.pow(2), min=self.swag_model.var_clamp)
                inv_diag = 1.0 / var
                
                # 1. Save to CURRENT map (for Numerator)
                self.current_inv_diag_map[name] = inv_diag.detach().clone()
                
                # 2. Add to TOTAL map (for Denominator)
                self.sum_inv_diag_map[name] += inv_diag.detach()  # Accumulated on GPU
                
                # ... (Compute D_tilde and M_global same as previous) ...
                D_tilde = D_hat * inv_diag.view(-1).unsqueeze(1)  # Flatten inv_diag
                current_task_D_tilde_map[name] = D_tilde.cpu()  # Low rank histories are stored on CPU
                
                M_part = torch.matmul(D_hat.t(), D_tilde)
                M_global += M_part

        M_inv = torch.linalg.inv(M_global).cpu()  # Low rank histories are stored on CPU
        self.low_rank_history.append((current_task_D_tilde_map, M_inv))
        
        print(f"Task precision stored.")

    def compute_precision_quadratic_form_woodbury(self, delta_theta_map, 
                                                  diag_map=None, 
                                                  low_rank_list=None):
        """
        Computes delta^T * Lambda * delta.
        
        Args:
            delta_theta_map: {param_key: tensor_delta}
            diag_map: Diagonal precision map to use. 
                      If None, uses self.sum_inv_diag_map (TOTAL history).
                      For numerator, pass self.current_inv_diag_map.
            low_rank_list: List of (D_tilde_map, M_inv) tuples.
                           If None, uses self.low_rank_history (TOTAL history).
                           For numerator, pass [self.low_rank_history[-1]].
        """
        if diag_map is None:
            diag_map = self.sum_inv_diag_map
        if low_rank_list is None:
            low_rank_list = self.low_rank_history

        # 1. Diagonal term
        scalar_diag_total = 0.0
        # print("diag_map keys:", list(diag_map.keys()))
        # print("delta_theta_map keys:", list(delta_theta_map.keys()))
        for key, inv_diag in diag_map.items():
            if key in delta_theta_map:
                delta = delta_theta_map[key].view(-1)
                # if torch.allclose(delta, torch.zeros_like(delta)):
                #     print(f"Warning: Zero delta for parameter {key} in precision quadratic form.")
                # delta^2 * diag
                contrib = (delta.pow(2) * inv_diag.view(-1)).sum().item()
                scalar_diag_total += contrib
                # print(f"Diag contrib for {key}: {contrib}")
            else:
                print(f"Warning: Parameter {key} not found in delta_theta_map for precision quadratic form.")
        # if torch.isclose(torch.tensor(scalar_diag_total), torch.tensor(0.0, device=self.device)):
        #     print("Warning: Diagonal term of precision quadratic form is zero.")
        #     exit(-1)

        # 2. Low-Rank terms
        scalar_low_rank_total = 0.0
        for (task_D_tilde_map, task_M_inv) in low_rank_list:            
            # Project delta onto this task's low-rank basis
            # v_global = D~^T * delta
            v_global = torch.zeros(task_M_inv.shape[0], device=self.device)
            for key, D_tilde_cpu in task_D_tilde_map.items():
                if key in delta_theta_map:
                    D_tilde_gpu = D_tilde_cpu.to(self.device)
                    delta = delta_theta_map[key].view(-1)
                    
                    v_part = torch.matmul(delta.view(1, -1), D_tilde_gpu).view(-1)
                    v_global += v_part
                else:
                    print(f"Warning: Parameter {key} not found in delta_theta_map for low-rank projection.")

            # Weighted Norm
            task_M_inv_gpu = task_M_inv.to(self.device)
            low_rank_part = torch.matmul(torch.matmul(v_global, task_M_inv_gpu), v_global)
            
            scalar_low_rank_total += low_rank_part

        return scalar_diag_total - scalar_low_rank_total