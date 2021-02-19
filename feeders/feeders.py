"""
Data loading functions

Continuum expects data to be of the following format
x_tr and x_te Contents:
task x
    (task_start_class, task_end_class),
    (task_sample_no x sample_image_data), image data each 3072 floats (32 x 32 x 3)
    (task_sample_no), label data each 1 int
"""

import torch


def default_load(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)): # each task
        n_outputs = max(n_outputs, d_tr[i][2].max().item()) # maximum label
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


def load_nturgbd60(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    print('length of training data', len(d_tr))
    print('length of testing data', len(d_te))
    
    n_outputs = 0
    for datapoint in d_te:
        n_outputs = max(n_outputs, datapoint[2])
    print('max class', n_outputs + 1)
    
    return d_tr, d_te, 0, n_outputs + 1, len(d_tr)


def load_datasets(args):
    '''
    Returns: 
        training dataset: see description in feeder docstring
        testing dataset: see description in feeder docstring
        n_inputs: size of input = 3072 for cifar
        n_outputs: maximum label = 100 for cifar
        n_tasks: number of tasks = 20 for cifar
    '''
    if 'nturgbd60' in args.data_file:
        return load_nturgbd60(args)
    else:
        return default_load(args)
