'''
Data loading functions and
Continuum class

Continuum expects data to be of the following format
x_tr and x_te Contents:
task x
    (task_start_class, task_end_class),
    (task_sample_no x sample_image_data), image data each 3072 floats (32 x 32 x 3)
    (task_sample_no), label data each 1 int
'''

import random
import torch


def load_datasets(args):
    '''
    Returns: 
        training dataset: see description in feeder docstring
        testing dataset: see description in feeder docstring
        n_inputs: size of input = 3072 for cifar
        n_outputs: maximum label = 100 for cifar
        n_tasks: number of tasks = 20 for cifar
    '''
    
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)): # each task
        if len(d_tr[i][2]) > 0:
            n_outputs = max(n_outputs, d_tr[i][2].max().item()) # maximum label
        if len(d_te[i][2]) > 0:
            n_outputs = max(n_outputs, d_te[i][2].max().item())
    print(f'number of tasks {len(d_tr)}')
    print(f'max class {n_outputs + 1}')
    print(f'n_inputs {n_inputs}')
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:
    '''
    1. Shuffle tasks if specified in args
    2. Gets permutation of samples (all or specified) for each task
    3. Store epoch permutation of samples for each task according to (1) and (2)

    __next__:
        returns 
    '''

    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []
        # stores permutation of sample by task
        # e.g. [1, 5, 3], [4, 1, 2] ...

        for t in range(n_tasks):
            N = data[t][1].size(0)
            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)

            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []
        # stores permutation of sample by permuation of task
        # e.g. [0, 1], [0, 5], [0, 3], [0, 1], [0, 5], [0, 3], [1, 4] ...

        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        '''
        Takes next task
        Adds data units indices while the next sample is the same task, 
            until batch or end of data
        Reads data tensor based on data units collected
        
        x_tr and x_te Contents:
            (start_class, end_class),
            (sample_no x image_data), image data each 3072 floats (32 x 32 x 3)
            (sample_no), label data each 1 int

        Returns (training dataset), (testing dataset) for task
        '''
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return ti, self.data[ti][0], self.data[ti][1][j], self.data[ti][2][j]
