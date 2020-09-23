"""
Created on 12/2/18

@author: Baoxiong Jia

Description:

"""
import sys
sys.path.append('/media/hdd/home/baoxiong/Projects/TPAMI2019/src')

import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import experiments.exp_config as exp_config
import utils.logutils as utils
# import models.BiLSTM as models
from models.LSTM_pred import LSTM_Pred

def loss_func(criterion, model_outputs, labels, total_lengths, args):
    loss = 0
    for i_batch in range(model_outputs.size()[1]):
        gt_pred_labels = list()
        seg_length = int(total_lengths[i_batch])
        current_label = int(labels[0, i_batch])
        for f in range(seg_length):
            if int(labels[f, i_batch]) != current_label:
                current_label = int(labels[f, i_batch])
                gt_pred_labels.extend([current_label for _ in range(f - len(gt_pred_labels) - 1)])
        gt_pred_labels.extend([int(labels[seg_length - 1, i_batch]) for _ in range(seg_length - len(gt_pred_labels))])
        gt_pred_labels = torch.LongTensor(gt_pred_labels)
        if args.cuda:
            gt_pred_labels = gt_pred_labels.cuda()
        loss += criterion(model_outputs[:seg_length, i_batch, :], gt_pred_labels)
    return loss

def train(data_loader, model, criterion, optimizer, epoch, args):
    logger = utils.Logger()
    model.train()
    start_time = time.time()
    task_acc_ratio = utils.AverageMeter()
    task_macro_prec = utils.AverageMeter()
    task_macro_rec = utils.AverageMeter()
    task_macro_f1 = utils.AverageMeter()
    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='Batch Loop Training')):
        logger.data_time.update(time.time() - start_time)
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit
        if args.cuda:
            features_batch = features_batch.cuda()
            labels_batch = labels_batch.cuda()

        video_length = torch.sum(total_lengths).item()
        output = model(features_batch)
        _, pred_labels = torch.max(output, dim=-1)
        loss = loss_func(criterion, output, labels_batch, total_lengths, args)
        logger.losses.update(loss.item(), video_length)
        gt_pred_labels = get_gt_pred(output, labels_batch, total_lengths)
        pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()

        micro_prec = utils.compute_accuracy(gt_pred_labels, pred_labels)
        macro_prec, macro_rec, macro_f1 = utils.compute_accuracy(gt_pred_labels, pred_labels, metric='macro')
        task_acc_ratio.update(micro_prec, video_length)
        task_macro_prec.update(macro_prec, video_length)
        task_macro_rec.update(macro_rec, video_length)
        task_macro_f1.update(macro_f1, video_length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.batch_time.update(time.time() - start_time)
        start_time = time.time()

        if (batch_idx + 1) % args.log_interval == 0:
            tqdm.write('Task {} Epoch: [{}][{}/{}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                        'Prec {prec.val:.4f} ({prec.avg:.4f})\t'
                        'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
                        'F1 {f1.val:.4f} ({f1.avg:.4f})'.format(
                args.task, epoch, batch_idx, len(data_loader), batch_time=logger.batch_time,
                data_time=logger.data_time, loss=logger.losses, top1=task_acc_ratio,
                prec=task_macro_prec, recall=task_macro_rec, f1=task_macro_f1))

def get_gt_pred(model_outputs, labels, total_lengths):
    all_gt_pred_labels = list()
    for i_batch in range(model_outputs.size()[1]):
        gt_pred_labels = list()
        seg_length = int(total_lengths[i_batch])
        current_label = int(labels[0, i_batch])
        for f in range(seg_length):
            if int(labels[f, i_batch]) != current_label:
                current_label = int(labels[f, i_batch])
                gt_pred_labels.extend([current_label for _ in range(f-len(gt_pred_labels)-1)])
        gt_pred_labels.extend([int(labels[seg_length-1, i_batch]) for _ in range(seg_length-len(gt_pred_labels))])
        all_gt_pred_labels.extend(gt_pred_labels)
    return all_gt_pred_labels

def validate(data_loader, model, args, test=False):
    task_acc_ratio = utils.AverageMeter()
    task_macro_prec = utils.AverageMeter()
    task_macro_rec = utils.AverageMeter()
    task_macro_f1 = utils.AverageMeter()
    all_labels = list()
    all_gt_labels = list()
    model.eval()
    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='Batch Loop Validation' if not test else 'Batch Loop Testing')):
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit
        batch_num = features_batch.size(1)
        if args.cuda:
            features_batch = features_batch.cuda()
            labels_batch = labels_batch.cuda()

        video_length = torch.sum(total_lengths).item()
        output = model(features_batch)
        _, pred_labels = torch.max(output, dim=-1)
        gt_pred_labels = get_gt_pred(output, labels_batch, total_lengths)
        pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()

        micro_prec = utils.compute_accuracy(gt_pred_labels, pred_labels)
        macro_prec, macro_rec, macro_f1 = utils.compute_accuracy(gt_pred_labels, pred_labels, metric='macro')
        task_acc_ratio.update(micro_prec, video_length)
        task_macro_prec.update(macro_prec, video_length)
        task_macro_rec.update(macro_rec, video_length)
        task_macro_f1.update(macro_f1, video_length)

        all_gt_labels.extend(gt_pred_labels)
        all_labels.extend(pred_labels)

        if not test:
            if (batch_idx + 1) % args.log_interval == 0:
                tqdm.write('Task {} {} Batch [{}/{}]\t'
                           'Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                           'Prec {prec.val:.4f} ({prec.avg:.4f})\t'
                           'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
                           'F1 {f1.val:.4f} ({f1.avg:.4f})'.format(
                    args.task, 'val' if not test else 'test', batch_idx, len(data_loader), top1=task_acc_ratio,
                    prec=task_macro_prec, recall=task_macro_rec, f1=task_macro_f1))

    micro_prec = utils.compute_accuracy(all_gt_labels, all_labels)
    macro_prec, macro_recall, macro_fscore = utils.compute_accuracy(all_gt_labels, all_labels, metric='weighted')
    tqdm.write('[Evaluation] Micro Prec: {}\t'
                   'Macro Precision: {}\t'
                   'Macro Recall: {}\t'
                   'Macro F-score: {}'.format(micro_prec, macro_prec, macro_recall, macro_fscore))
    return task_acc_ratio.avg


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    exp_info = exp_config.Experiment(args.dataset)
    paths = exp_info.paths
    args.paths = paths
    args.resume = os.path.join(paths.checkpoint_root,
                               'segment_prediction_{}_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.model, args.epochs,
                                                                      args.lr, args.batch_size, args.lr_decay,
                                                                          1 if not args.subsample else args.subsample,
                                                                        args.dropout_rate))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    feature_size, train_loader, val_loader, test_loader, _ = exp_info.get_dataset(args, save=True)
    label_num = exp_info.get_label_num(args)

    criterion = torch.nn.CrossEntropyLoss()
    hidden_size = 256
    hidden_layers = 2
    model = LSTM_Pred(feature_size, hidden_size, hidden_layers, label_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, args.lr_freq, args.lr_decay)
    model = torch.nn.DataParallel(model)
    if args.cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    if args.resume:
        utils.load_checkpoint(args, model, optimizer, scheduler)

    best_prec = 0.0

    if args.eval:
        validate(test_loader, model, args, test=True)
    else:
        for epoch in tqdm(range(args.start_epoch, args.epochs), desc='Epochs Loop'):
            train(train_loader, model, criterion, optimizer, epoch, args)
            prec = validate(val_loader, model, args)
            best_prec = max(prec, best_prec)
            is_best = (best_prec == prec)
            tqdm.write('Best precision: {:.03f}'.format(best_prec))
            if (epoch + 1) % args.save_interval == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best, args)

def parse_arguments():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Unsupported value encountered')
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
        return x

    tasks = ['affordance', 'activity']
    task = tasks[0]
    parser = argparse.ArgumentParser(description='Segmentation LSTM baseline')
    parser.add_argument('--dataset', default='VCLA_GAZE', type=str,
                        help='indicating which dataset to use')
    parser.add_argument('--model', default='lstm', type=str,
                        help='Model for classification (default: LSTM)')
    parser.add_argument('--seed', default=12345, type=int,
                        help='Default seed for all random generators')
    parser.add_argument('--task', default=task, type=str,
                        help='Task for network training (default: activity)')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool,
                        help='Option flag for using cuda trining (default: True)')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='Default device for cuda')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='starting epoch of training (default: 0)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of epochs for training (default: 50)')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size for training (default: 1)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for the feature extraction process (default: 1e-3)')
    parser.add_argument('--lr_decay', type=lambda x: restricted_float(x, [0.01, 1]), default=1.,
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--lr_freq', default=25, type=float,
                        help='learing rate decay frequency while updating')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='Intervals for logging (default: 10 batch)')
    parser.add_argument('--save_interval', type=int, default=1, metavar='N',
                        help='Intervals for saving checkpoint (default: 1 epochs)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='ratio of data for training purposes (default: 0.65)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='ratio of data for validation purposes (default: 0.1)')

    parser.add_argument('--eval', default=False, type=str2bool,
                        help='indicates whether need to run evaluation on testing set')
    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='dropout_rate')

    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample frequency for Breakfast dataset')

    args = parser.parse_args()
    if args.task == 'affordance':
        raise ValueError('Segmentation experiment should be on Activity task')
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)