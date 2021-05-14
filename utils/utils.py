import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from opts.opts import args
import shutil, os

import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch

def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))

        print((grad_out[0]))
        print((grad_in[0]))

    return hook


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def log_add(log_a, log_b):
    return log_a + np.log(1 + np.exp(log_b - log_a))


def class_accuracy(prediction, label):
    cf = confusion_matrix(prediction, label)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt.astype(float)

    mean_cls_acc = cls_acc.mean()

    return cls_acc, mean_cls_acc

def cal_Lq(y_softmax, label, args):
    softmax = nn.Softmax(dim=1)
    y_softmax = softmax(y_softmax)
    #print(label.size())
    Lqq = 0.01
    Lqk = -1

    the_label = torch.LongTensor(np.array(range(label.size(0)))).cuda()
    y_label = y_softmax[the_label, label.data]
    y_loss = (1.0 - torch.pow(y_label, Lqq)) / Lqq
    #y_loss = - torch.log(y_label)

    # truncated Lq loss
    weight_var = (y_label > Lqk).float().detach()
    Lq = torch.sum(y_loss * weight_var) / torch.sum(weight_var)
    return Lq

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

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

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))

    decay = 0.5 ** (epoch // 10)
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)



def randSelectBatch(input, num):
    id_all = torch.randperm(input.size(0)).cuda()
    id = id_all[:num]
    return id, input[id]

def plot_confusion_matrix(path, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    num_classlabels = cm.sum(axis=1) # count the number of true labels for all the classes
    np.putmask(num_classlabels, num_classlabels == 0, 1) # avoid zero division

    if normalize:
        cm = cm.astype('float') / num_classlabels[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(13, 13))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    factor = 100 if normalize else 1
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*factor, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)