import random
import torch


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
            return self.data[ti][1][j], ti, self.data[ti][2][j]
