# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import logging

import torch

logger = logging.getLogger(__name__)


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(result_t, result_a, fname=None):
    nt, changes = task_changes(result_t)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[changes]

    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'w')

        logger.info(' '.join(['%.4f' % r for r in baseline]))
        logger.info('|')
        for row in range(result.size(0)):
            logger.info(' '.join(['%.4f' % r for r in result[row]]))
        logger.info('')
        # logger.info('Diagonal Accuracy: %.4f' % acc.mean())
        logger.info('Final Accuracy: %.4f' % fin.mean())
        logger.info('Backward: %.4f' % bwt.mean())
        logger.info('Forward:  %.4f' % fwt.mean())
        f.close()
    
    else:
        logger.info(' '.join(['%.4f' % r for r in baseline]))
        logger.info('|')
        for row in range(result.size(0)):
            logger.info(' '.join(['%.4f' % r for r in result[row]]))
        logger.info('')
        # logger.info('Diagonal Accuracy: %.4f' % acc.mean())
        logger.info('Final Accuracy: %.4f' % fin.mean())
        logger.info('Backward: %.4f' % bwt.mean())
        logger.info('Forward:  %.4f' % fwt.mean())

    stats = []
    # stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats
