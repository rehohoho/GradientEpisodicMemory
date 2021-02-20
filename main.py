# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import datetime
import argparse
import random
import uuid
import time
import os

import numpy as np

import torch
from metrics.metrics import confusion_matrix
from feeders.continuum import load_datasets, Continuum

# continuum iterator #########################################################
# train handle ###############################################################


def eval_tasks(model, tasks, args):
    model.eval()
    result = []
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        rt = 0
        
        eval_bs = x.size(0)

        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)
            if b_from == b_to:
                if 'nturgbd60' not in args.data_file:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                else:
                    xb = torch.unsqueeze(x[b_from], dim=0)
                    yb = torch.LongTensor([y[b_to]])
            else:
                xb = x[b_from:b_to]
                yb = y[b_from:b_to]
            if args.cuda:
                xb = xb.cuda()
            _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
            rt += (pb == yb).float().sum()

        result.append(rt / x.size(0))

    return result


def life_experience(model, continuum, x_te, args):
    result_a = []
    result_t = []

    current_task = -1 # trigger first task change update
    time_start = time.time()
    print(f'Length of continuum: {continuum.length}')

    for (i, (task, classes, x, y)) in enumerate(continuum):
        # task - task number
        # classes - classes in same task
        # x - image: (b x 3072)
        # y - label: (b)``
        print(f'continuum idx {i}')
        if continuum.current == 20:
            break
        
        if task != current_task and args.model == 'gem':
            model.update_loss_mask(classes)

        # data sorted by task in data processing and built accordingly in continuum init
        if(((i % args.log_every) == 0) or (task != current_task)):
            result_a.append(eval_tasks(model, x_te, args))
            result_t.append(current_task)
            current_task = task

        if args.model != 'gem':
            v_x = x.view(x.size(0), -1)
        else:
            v_x = x
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        model.observe(v_x, task, v_y)

    result_a.append(eval_tasks(model, x_te, args))
    result_t.append(current_task)

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a), time_spent


def main(args):
    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    print('seeds initialised')

    # load data
    x_tr, x_te, input_shape, n_outputs, n_tasks = load_datasets(args)
    print('data loaded')

    # set up continuum
    continuum = Continuum(x_tr, args)
    print('continuum loaded')

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(input_shape, n_outputs, n_tasks, args)
    if args.cuda:
        model.cuda()
    print('model loaded')

    # run model on continuum
    result_t, result_a, spent_time = life_experience(
        model, continuum, x_te, args)
    print('training done')

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.data_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, model.state_dict(),
                stats, one_liner, args), fname + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--model_args', default=None,
                        help='a JSON string which specifies model arguments')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False

    # multimodal model has one extra layer
    if args.model == 'multimodal':
        args.n_layers -= 1

    main(args)
