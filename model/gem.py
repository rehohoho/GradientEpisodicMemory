# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import yaml
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import quadprog

from .common import MLP, ResNet18
from .agcn import AGCN

logger = logging.getLogger(__name__)

# Auxiliary functions useful for GEM's inner optimization.

''' remove loss subsetting, use weighted loss
def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
'''


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 input_shape,
                 n_outputs,
                 n_tasks,
                 loss_masks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = ('cifar100' in args.data_file)
        self.is_nturgbd = ('nturgbd60' in args.data_file)
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        elif self.is_nturgbd:
            model_args = yaml.load(args.model_args, Loader=yaml.FullLoader)
            self.net = AGCN(*model_args.values()) # requires model_args to follow defined sequence
        else:
            assert len(input_shape) == 1
            self.net = MLP([input_shape[0]] + [nh] * nl + [n_outputs])
        logger.info(self.net.__str__())

        self.ce_type = nn.CrossEntropyLoss
        self.loss_masks = loss_masks
        self.task_classes = {}
        for task, loss_mask in loss_masks.items():
            self.task_classes[task] = torch.squeeze(loss_mask.nonzero(as_tuple=False))
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, *list(input_shape))
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        ''' remove loss subsetting, use weighted loss
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        '''

    def forward(self, x, t):
        output = self.net(x)
        output *= self.loss_masks[t]
        ''' remove loss subsetting, use weighted loss
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        '''
        return output
    
    def compute_loss(self, pred, label, task_no):
        ''' Subsets relevant classes according to task and computes loss '''

        # (n_memories, inputs), (n_memories) -> (n_memories, outputs), (n_memories)
        ''' remove loss subsetting, use weighted loss
        offset1, offset2 = compute_offsets(past_task, self.nc_per_task, self.is_cifar)
        ptloss = self.ce(loss_input[:, offset1: offset2], loss_target - offset1)
        '''
        loss_input = self.forward(pred, task_no)
        loss_target = label
        task_classes = self.task_classes[task_no]
        gathered_loss_input = torch.index_select(loss_input, 1, task_classes)
        # logger.info(f'mapping for {loss_target} to its indices {gather_idx}')
        for i in range(len(task_classes)-1, -1, -1):
            loss_target[loss_target==task_classes[i]] = i
        # logger.info(f'done {loss_target}')
        logger.info(f'compute loss {gathered_loss_input} {loss_target}')

        return self.ce_type()(gathered_loss_input, loss_target)

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t) #[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 ...]
            self.old_task = t

        # Update ring buffer storing examples from current task
        # self.memory_data: (tasks x n_memories x input_size)
        # self.memory_labs: (tasks x n_memories)
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks, except current
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                ptloss = self.compute_loss(self.memory_data[past_task], self.memory_labs[past_task], past_task)
                logger.info(f'memory loss {ptloss}')
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()
        loss = self.compute_loss(x, y, t)
        logger.info(f'normal loss {loss}')
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
        self.opt.step()
