import argparse

import numpy as np
import torch
from tqdm import tqdm


def main(args):

    r1 = torch.load(args.score1)
    r2 = torch.load(args.score2)
    _, d_te = torch.load(args.label_file)
    labels = d_te[0][2]
    
    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(labels))):
        
        l = labels[i]
        r = r1[i] + r2[i] * arg.alpha
        
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(acc, acc5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score1', default='experiment_results/results_nturgbd60_xsub_bone/gem_nturgbd60_xsub_bone.pt_score.pt',
        help='first pt file containing scores per sample.')
    parser.add_argument('--score2', default='experiment_results/results_nturgbd60_xsub_joint/gem_nturgbd60_xsub_joint.pt_score.pt',
        help='second pt file containing scores per sample.')
    parser.add_argument('--label_file', default='data/nturgbd60_xsub_bone.pt',
        help='file containing labels')
    parser.add_argument('--alpha', default=1, help='weighted summation')
    arg = parser.parse_args()

    main(arg)