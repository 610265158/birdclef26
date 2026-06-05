import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import sys

sys.path.append('.')
import numpy as np
import torch.nn as nn
from train_config import config as cfg

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from lib.utils.logger import logger

import warnings

warnings.filterwarnings('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ROCAUCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):

        self.y_true_11 = None
        self.y_pred_11 = None

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()

        y_pred = torch.sigmoid(y_pred).data.cpu().numpy()

        if self.y_true_11 is None:
            self.y_true_11 = y_true
            self.y_pred_11 = y_pred
        else:
            self.y_true_11 = np.concatenate((self.y_true_11, y_true), axis=0)
            self.y_pred_11 = np.concatenate((self.y_pred_11, y_pred), axis=0)

    @property
    def avg(self):
        non_zeroindx = np.sum(self.y_true_11, axis=0) > 0

        self.y_true_11 = self.y_true_11[:,non_zeroindx]
        self.y_pred_11 = self.y_pred_11[:,non_zeroindx]

        scpre = roc_auc_score(self.y_true_11, self.y_pred_11, average='macro')
        return scpre


if __name__ == '__main__':
    ROCAUC_score = ROCAUCMeter()

    y_true = np.random.randint(2, size=10000)
    y_prob = np.random.rand(10000)

    ROCAUC_score.update(y_true, y_prob)
    print(ROCAUC_score.avg)
