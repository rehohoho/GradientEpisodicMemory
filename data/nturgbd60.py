'''
Splits nturgbd60 into superclasses suitable for gem

(touch mouth) 1 drink water, 2 eat meal, 3 brush teeth, 37 wipe face
(touch body) 14 put on jacket, 15 take off jacket, 45 chest pain, 46 back pain, 47 neck pain
(touch feet) 16 put on shoe, 17 take off shoe
(touch eye) 18 put on glasses, 19 take off glasses
(touch head) 20 put on a hat, 21 take off a hat, 4 brush hair, 38 salute, 44 headache
(hand front of body) 5 drop, 13 tear up paper, 29 play with phone/tablet
(sitting action) 11 reading, 12 writing, 30 type on a keyboard
(bend body) 6 pick up, 8 sit down, 9 stand up, 48 nausea/vomitting
(raise hand) 7 throw, 22 cheer up, 23 hand waving, 31 point to something, 32 taking a selfie, 54 point finger
(hand in middle of body) 10 clapping, 33 check time from watch, 34 rub two hands, 39 put palms together, 40 cross hands in front, 49 fan self
(move legs) 24 kicking something, 51 kicking
(pockets) 25 reach into pocket, 28 phone call, 56 touch pocket
(move head) 35 nod head, 36 shake head, 41 sneeze/cough
(movement) 26 hopping, 27 jump up, 42 staggering, 43 falling down, 59 walking towards, 60 walking apart
(mutual action hand) 50 punch/slap, 57 giving object, 58 shaking hands
(mutual action body) 52 pushing, 53 pat on back, 55 hugging
'''

import argparse
import os
import pickle
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_superclass_map(use_single_task=False):
    if use_single_task:
        superclasses = [[i for i in range(60)]]
    else:
        superclasses = [
            [0, 1, 2, 36],
            [13, 14, 44, 45, 46],
            [15, 16],
            [17, 18],
            [3, 19, 20, 37, 43],
            [4, 12, 28],
            [10, 11, 29],
            [5, 7, 8, 47],
            [6, 21, 22, 30, 31, 53],
            [9, 32, 33, 38, 39, 48],
            [23, 50],
            [24, 27, 55],
            [34, 35, 40],
            [25, 26, 41, 42, 58, 59],
            [49, 56, 57],
            [51, 52, 54]
        ]

    class_to_superclass = [0] * 60
    for i, classes in enumerate(superclasses):
        for _class in classes:
            class_to_superclass[_class] = i
    
    print(f'Using superclasses {superclasses}.\nclass_to_superclass map {class_to_superclass}')

    return superclasses, class_to_superclass


def load_pickle(file_path):
    try:
        with open(file_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(file_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    
    return sample_name, label


def build_task_list(data, labels, superclasses, class_to_superclass):
    '''
    Continuum expects data to be of the following format
    x_tr and x_te Contents:
    task x
        (task_start_class, task_end_class),
        (task_sample_no x sample_image_data), image data each 3072 floats (32 x 32 x 3)
        (task_sample_no), label data each 1 int
    '''
    data_list = [[] for _ in range(len(superclasses))]
    label_list = [[] for _ in range(len(superclasses))]
    
    for d, l in zip(data, labels):
        superclass = class_to_superclass[l]
        data_list[superclass].append(d)
        label_list[superclass].append(l)
    
    tasks = []
    for idx, superclasses in enumerate(superclasses):
        tasks.append([
            superclasses, 
            torch.from_numpy(np.array(data_list[idx])),
            torch.from_numpy(np.array(label_list[idx]))
        ])

    return tasks


def show_task_list_info(task_list):
    for idx, task in enumerate(task_list):
        print(f'task {idx}:')
        print(f'\tclasses: {task[0]}')
        print(f'\tnumber of samples: {len(task[1])}')
        if len(task[1]) > 0:
            print(f'\tsample shape: {task[1][0].shape}')
            print(f'\tsample label: {task[2][0]}')


def main(args):
    print('Starting...')
    _, l_tr = load_pickle(args.l_tr_path)
    _, l_te = load_pickle(args.l_te_path)
    print('Loading label data (pickle files) done.')

    d_tr = np.load(args.d_tr_path, mmap_mode='r')
    d_te = np.load(args.d_te_path, mmap_mode='r')
    print('Loading data (npy files) done.')

    # for testing uncomment
    # d_tr = d_tr[:100]
    # d_te = d_te[:100]
    # l_tr = l_tr[:100]
    # l_te = l_te[:100]

    superclasses, class_to_superclass = generate_superclass_map(args.use_single_task)
    tasks_tr = build_task_list(d_tr, l_tr, superclasses, class_to_superclass)
    tasks_te = build_task_list(d_te, l_te, superclasses, class_to_superclass)
    print('Loading task list done.')
    
    show_task_list_info(tasks_tr)
    
    torch.save([tasks_tr, tasks_te], args.o)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--d_tr_path', default='raw/train_data_joint.npy', help='nturgbd60 training data .npy')
    parser.add_argument('--l_tr_path', default='raw/train_label.pkl', help='nturgbd60 testing data .npy')
    parser.add_argument('--d_te_path', default='raw/val_data_joint.npy', help='nturgbd60 training labels .pkl')
    parser.add_argument('--l_te_path', default='raw/val_label.pkl', help='nturgbd60 testing labels .pkl')
    parser.add_argument('--use_single_task', action='store_true', help='use single task for all classes')
    parser.add_argument('--o', default='nturgbd60.pt', help='output file')
    args = parser.parse_args()

    main(args)
