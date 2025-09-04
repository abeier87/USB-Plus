# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument



@IMB_ALGORITHMS.register('simpro')
class SIMPRO(ImbAlgorithmBase):
    """
        SIMPRO algorithm.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super(SIMPRO, self).__init__(args, net_builder, tb_log, logger, **kwargs)
        
        self.tau = args.tau
        self.ema_u = args.ema_u
        
        self.py_u = torch.ones(self.num_classes).cuda() / self.num_classes
        self.py_e = self.py_u.clone()
        
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(lb_class_dist)
        self.phi = torch.from_numpy(lb_class_dist / lb_class_dist.sum())
        self.phi_e = self.phi.clone()


    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")
            
            
            self.py_e = torch.zeros_like(self.py_e)
            if not self.py_e.is_cuda:
                self.py_e = self.py_e.cuda()
            if not self.py_u.is_cuda:
                self.py_u = self.py_u.cuda()
            if not self.phi.is_cuda:
                self.phi = self.phi.cuda()
            if not self.phi_e.is_cuda:
                self.phi_e = self.phi_e.cuda()
            if not self.lb_class_dist.is_cuda:
                self.lb_class_dist = self.lb_class_dist.cuda()

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.phi_e = (self.lb_class_dist + self.py_e) / (self.lb_class_dist.sum() + self.py_e.sum())
            self.phi = self.ema_u * self.phi + (1 - self.ema_u) * self.phi_e
            self.py_u = self.ema_u * self.py_u + (1 - self.ema_u) * self.py_e / self.py_e.sum()

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
        
    def train_step(self, *args, **kwargs):
        x_lb, y_lb, x_ulb_w, x_ulb_s = kwargs['x_lb'], kwargs['y_lb'], kwargs['x_ulb_w'], kwargs['x_ulb_s']
        
        num_lb = y_lb.shape[0]

        

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x = outputs['logits'][:num_lb]
                logits_u_w, logits_u_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_u_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_u_w = outs_x_ulb_w['logits']
  
            
            pseudo_label = self.compute_prob(logits_u_w.detach() + torch.log(self.py_u ** self.tau + 1e-12))

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.p_cutoff)

            self.py_e += torch.bincount(targets_u[mask], minlength=self.num_classes).float()

            Lx = self.ce_loss(logits_x + torch.log(self.phi ** self.tau + 1e-12), y_lb, reduction='mean')
            
            Lu = self.consistency_loss(logits_u_s + torch.log(self.phi ** self.tau + 1e-12), targets_u, 'ce', mask=mask)

            total_loss = Lx + Lu
            
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=Lx.item(), 
                                         unsup_loss=Lu.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())

        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tau', float, 2),
            SSL_Argument('--ema_u', float, 0.9),
        ]