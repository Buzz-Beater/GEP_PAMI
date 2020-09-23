"""
Created on 10/30/18

@author: Baoxiong Jia

Description:

"""
import os
import shutil
import torch
import numpy as np
import sklearn.metrics

rgb_width = 1920
rgb_height = 1280
depth_width = 512
depth_height = 424


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


class MultiAverageMeter(AverageMeter):
    def __init__(self):
        super(MultiAverageMeter, self).__init__()
        self.reset()

    def reset(self):
        self.vals = {}
        self.avgs = {}
        self.sums = {}
        self.counts = {}
        self.val = 0
        self.avg = 0

    # Return avg precision for affordance that is not null
    def update(self, key, val, n=1):
        if key not in self.vals.keys():
            self.vals[key] = 0
            self.avgs[key] = 0
            self.sums[key] = 0
            self.counts[key] = 0
        self.vals[key] = val
        self.sums[key] += val * n
        self.counts[key] += n
        self.avgs[key] = self.sums[key] / self.counts[key]

        val = 0
        avg = 0
        count = 0
        for key in self.vals:
            if key is not 'null':
                val += self.vals[key]
                avg += self.avgs[key]
                count += 1
        if count != 0:
            self.val = val / count
            self.avg = avg / count
        else:
            self.val = -1
            self.avg = -1

class Logger(object):
    """record useful logging varaibles for training and validation"""
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.multi_losses = MultiAverageMeter()
        self.top1 = MultiAverageMeter()

def compute_accuracy(gt_results, results, labels='all', metric='micro'):
    if labels == 'all':
        labels_list = list(set(gt_results + results))
    else:
        labels_list = list(set(gt_results + results))
        labels_list.remove(0)
    results = sklearn.metrics.precision_recall_fscore_support(gt_results, results, labels=labels_list, average=metric)
    if metric == 'micro':
        return results[0]
    else:
        return results[0], results[1], results[2]

def save_checkpoint(state_dict, is_best, args, filename='checkpoint.pth'):
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    torch.save(state_dict, os.path.join(args.resume, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.resume, filename), os.path.join(args.resume, 'model_best.pth'))

def save_checkpoint_epoch(state_dict, epoch, args):
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    torch.save(state_dict, os.path.join(args.resume, 'checkpoint_{}.pth'.format(epoch)))

def load_checkpoint_epoch(args, model, epoch, optimizer=None, scheduler=None):
    file_name = os.path.join(args.resume, 'checkpoint_{}.pth'.format(epoch))
    print('Loading {}: {}'.format(file_name, os.path.isfile(file_name)))
    if os.path.isfile(file_name):
        checkpoint = torch.load(file_name)
        print('Best precision:{}'.format(checkpoint['best_prec']))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print('finished loading')


def load_checkpoint(args, model, optimizer=None, scheduler=None):
    print('Loading {}: {}'.format(os.path.join(args.resume, 'model_best.pth'), os.path.isfile(os.path.join(args.resume, 'model_best.pth'))))
    if os.path.isfile(os.path.join(args.resume, 'model_best.pth')):
        checkpoint = torch.load(os.path.join(args.resume, 'model_best.pth'))
        print('Best precision:{}'.format(checkpoint['best_prec']))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print('finished loading')



# TODO: Fix transform in both 3d and 2d
def transform(skeleton, mean_skeleton, dims, anchor_points=[5, 9, 1]):
    aligned_skeleton = skeleton
    return aligned_skeleton