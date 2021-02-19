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
(movement) 26 hopping, 27 jump up, 42 staggering, 43 falling down, 59 walking towards
(mutual action hand) 50 punch/slap, 57 giving object, 58 shaking hands
(mutual action body) 52 pushing, 53 pat on back, 55 hugging
'''

import argparse
import os
import pickle

import numpy as np
import torch

SUPERCLASSES = [
    [1, 2, 3, 37],
    [14, 15, 45, 46, 47],
    [16, 17],
    [18, 19],
    [4, 20, 21, 38, 44],
    [5, 13, 29],
    [11, 12, 30],
    [6, 8, 9, 48],
    [7, 22, 23, 31, 32, 54],
    [10, 33, 34, 39, 40, 49],
    [24, 51],
    [25, 28, 56],
    [35, 36, 41],
    [26, 27, 42, 43, 59],
    [50, 57, 58],
    [52, 53, 55]
]

CLASS_TO_SUPERCLASS = [0] * 60
for i, classes in enumerate(SUPERCLASSES):
    for _class in classes:
        CLASS_TO_SUPERCLASS[_class] = i


def load_pickle(file_path):
    try:
        with open(file_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(file_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    
    return sample_name, label


def build_task_list(data, labels):
    tasks = []
    for d, l in zip(data, labels):
        superclass = CLASS_TO_SUPERCLASS[l]
        tasks.append([SUPERCLASSES[superclass], d, l])
    return tasks


def main(args):
    _, l_tr = load_pickle(args.l_tr_path)
    _, l_te = load_pickle(args.l_te_path)

    d_tr = np.load(args.d_tr_path, mmap_mode='r')
    d_te = np.load(args.d_te_path, mmap_mode='r')

    # for testing uncomment
    d_tr = d_tr[:100]
    d_te = d_te[:100]
    l_tr = l_tr[:100]
    l_te = l_te[:100]

    tasks_tr = build_task_list(d_tr, l_tr)
    tasks_te = build_task_list(d_te, l_te)
    
    print('sample idx 0', tasks_tr[0][0])
    print('sample idx 1 shape', tasks_tr[0][1].shape)
    print('sample idx 2', tasks_tr[0][2])
    print('length of training data', len(tasks_tr))
    print('length of testing data', len(tasks_te))
    
    torch.save([tasks_tr, tasks_te], args.o)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--d_tr_path', default='raw/train_data_joint.npy', help='nturgbd60 training data .npy')
    parser.add_argument('--l_tr_path', default='raw/train_label.pkl', help='nturgbd60 testing data .npy')
    parser.add_argument('--d_te_path', default='raw/val_data_joint.npy', help='nturgbd60 training labels .pkl')
    parser.add_argument('--l_te_path', default='raw/val_label.pkl', help='nturgbd60 testing labels .pkl')
    parser.add_argument('--o', default='nturgbd60.pt', help='output file')
    args = parser.parse_args()

    main(args)
