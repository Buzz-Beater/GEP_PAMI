"""
Created on 12/2/18

@author: Baoxiong Jia

Description:

"""
import sys
sys.path.append('/mnt/hdd/home/baoxiong/Projects/TPAMI2019/src')

import os
import time
import argparse
import copy

from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import experiments.exp_config as exp_config
import utils.logutils as logutils
import utils.evalutils as evalutils
import models.BiLSTM as lstm_model
import models.MLP as mlp_model

def loss_func(criterion, model_outputs, labels, total_lengths):
    loss = 0
    for i_batch in range(model_outputs.size(1)):
        seg_length = int(total_lengths[i_batch])
        loss += criterion(model_outputs[:seg_length, i_batch, :], labels[:seg_length, i_batch])
    return loss

def train(data_loader, model, criterion, optimizer, epoch, args):
    logger = logutils.Logger()
    model.train()
    start_time = time.time()
    task_acc_ratio = logutils.AverageMeter()
    task_macro_prec = logutils.AverageMeter()
    task_macro_rec = logutils.AverageMeter()
    task_macro_f1 = logutils.AverageMeter()
    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='Batch Loop Training')):
        logger.data_time.update(time.time() - start_time)
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit
        batch_num = features_batch.size(1)
        if args.cuda:
            features_batch = features_batch.cuda()
            labels_batch = labels_batch.cuda()
        if args.task == 'affordance':
            obj_num, _ = torch.max(obj_nums, dim=-1)
            features_batch = features_batch[:, :, : obj_num, :]
            labels_batch = labels_batch[:, :, : obj_num]
            features_batch = features_batch.view((features_batch.size(0), -1, features_batch.size(-1)))
            labels_batch = labels_batch.view((labels_batch.size(0), -1))

        output = model(features_batch)
        _, pred_labels = torch.max(output, dim=-1)
        loss = loss_func(criterion, output, labels_batch, total_lengths)
        video_length = torch.sum(total_lengths).item()
        logger.losses.update(loss.item(), video_length)

        for in_batch_idx in range(batch_num):
            detections = pred_labels[:, in_batch_idx].cpu().data.numpy().flatten().tolist()
            if args.subsample != 1:
                all_total_labels, all_total_lengths = additional
                gt_detections = all_total_labels[:all_total_lengths[in_batch_idx], in_batch_idx].flatten().tolist()
                video_length = len(gt_detections)
                detections = evalutils.upsample(detections, freq=args.subsample, length=video_length)
            else:
                gt_detections = labels_batch[:total_lengths[in_batch_idx], in_batch_idx].cpu().data.numpy().flatten().tolist()
                detections = detections[:total_lengths[in_batch_idx]]
                video_length = len(gt_detections)
            micro_prec = logutils.compute_accuracy(gt_detections, detections)
            macro_prec, macro_rec, macro_f1 = logutils.compute_accuracy(gt_detections, detections, metric='macro')
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

def validate(data_loader, model, args, test=False, save=False):
    task_acc_ratio = logutils.AverageMeter()
    task_macro_prec = logutils.AverageMeter()
    task_macro_rec = logutils.AverageMeter()
    task_macro_f1 = logutils.AverageMeter()
    all_labels = list()
    all_gt_labels = list()

    count = 0
    # switch to evaluate mode
    model.eval()
    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='Batch Loop Validation' if not test else 'Batch Loop Testing')):
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit
        batch_num = features_batch.size(1)
        count += torch.sum(total_lengths)

        if args.cuda:
            features_batch = features_batch.cuda()
            labels_batch = labels_batch.cuda()
        if args.task == 'affordance':
            obj_num, _ = torch.max(obj_nums, dim=-1)
            features_batch = features_batch[:, :, : obj_num, :]
            labels_batch = labels_batch[:, :, : obj_num]
            features_batch = features_batch.view((features_batch.size(0), -1, features_batch.size(-1)))
            labels_batch = labels_batch.view((labels_batch.size(0), -1))

        output = model(features_batch)
        for batch_i in range(output.size()[1]):
            save_output = output[:int(total_lengths[batch_i]), batch_i, :].squeeze().cpu().data.numpy()
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            np.save(os.path.join(args.save_path, '{}_out_s{}_b{}_c{}.npy'.format(sequence_ids[batch_i],
                                 args.subsample, args.batch_size, args.epochs)), save_output)

        _, pred_labels = torch.max(output, dim=-1)

        for in_batch_idx in range(batch_num):
            detections = pred_labels[:, in_batch_idx].cpu().data.numpy().flatten().tolist()
            if args.subsample != 1:
                all_total_labels, all_total_lengths = additional
                gt_detections = all_total_labels[:all_total_lengths[in_batch_idx], in_batch_idx].flatten().tolist()
                video_length = len(gt_detections)
                detections = evalutils.upsample(detections, freq=args.subsample, length=video_length)
            else:
                gt_detections = labels_batch[:total_lengths[in_batch_idx], in_batch_idx].cpu().data.numpy().flatten().tolist()
                detections = detections[:total_lengths[in_batch_idx]]
                video_length = len(gt_detections)
            all_labels.extend(detections)
            all_gt_labels.extend(gt_detections)

            micro_prec = logutils.compute_accuracy(gt_detections, detections)
            macro_prec, macro_rec, macro_f1 = logutils.compute_accuracy(gt_detections, detections, metric='macro')
            task_acc_ratio.update(micro_prec, video_length)
            task_macro_prec.update(macro_prec, video_length)
            task_macro_rec.update(macro_rec, video_length)
            task_macro_f1.update(macro_f1, video_length)

        if not test:
            if (batch_idx + 1) % args.log_interval == 0:
                tqdm.write('[Validation] Task {} {} Batch [{}/{}]\t'
                           'Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                           'Prec {prec.val:.4f} ({prec.avg:.4f})\t'
                           'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
                           'F1 {f1.val:.4f} ({f1.avg:.4f})'.format(
                    args.task, 'val' if not test else 'test', batch_idx + 1, len(data_loader), top1=task_acc_ratio,
                    prec=task_macro_prec, recall=task_macro_rec, f1=task_macro_f1))

        if args.task == 'affordance':
            output = output.view((output.size(0), batch_num, -1, output.size(-1)))
        if save:
            for batch_i in range(output.size()[1]):
                if args.task == 'affordance':
                    model_probs = torch.nn.Softmax(dim=-1)(output[:int(total_lengths[batch_i]), batch_i, :, :].squeeze()).cpu().data.numpy()
                else:
                    model_probs = torch.nn.Softmax(dim=-1)(output[:int(total_lengths[batch_i]), batch_i, :].squeeze()).cpu().data.numpy()
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                np.save(os.path.join(args.save_path, '{}.npy'.format(sequence_ids[batch_i])), model_probs)
    print(count)
    micro_prec = logutils.compute_accuracy(all_gt_labels, all_labels)
    macro_prec, macro_recall, macro_fscore = logutils.compute_accuracy(all_gt_labels, all_labels, metric='macro')
    tqdm.write('[Evaluation] Micro Prec: {}\t'
                'Macro Precision: {}\t'
                'Macro Recall: {}\t'
                'Macro F-score: {}'.format(micro_prec, macro_prec, macro_recall, macro_fscore))
    return micro_prec

def main(args):
    exp_info = exp_config.Experiment(args.dataset)
    paths = exp_info.paths
    args.paths = paths
    args.save_path = os.path.join(paths.inter_root, 'likelihood', args.task, args.model)
    args.resume = os.path.join(paths.checkpoint_root,
                               'detection_{}_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.model, args.epochs,
                                                                      args.lr, args.batch_size, args.lr_decay,
                                                                          1 if not args.subsample else args.subsample,
                                                                        args.dropout_rate))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    feature_size, train_loader, val_loader, test_loader, all_loader = exp_info.get_dataset(args, save=True)
    label_num = exp_info.get_label_num(args)

    criterion = torch.nn.CrossEntropyLoss()
    hidden_size = 256
    hidden_layers = 2
    if args.model == 'lstm':
        model = lstm_model.BiLSTM(feature_size, hidden_size, hidden_layers, label_num, dropout_rate=args.dropout_rate)
    else:
        model = mlp_model.MLP(feature_size, hidden_size, label_num, dropout_rate=args.dropout_rate)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, args.lr_freq, args.lr_decay)
    model = torch.nn.DataParallel(model)
    if args.cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    if args.resume:
        logutils.load_checkpoint_epoch(args, model, args.trained_epochs, optimizer, scheduler)
    best_prec = 0.0

    if not args.eval:
        best_dict = None
        for epoch in tqdm(range(args.start_epoch, args.epochs), desc='Epochs Loop'):
            train(train_loader, model, criterion, optimizer, epoch, args)
            prec = validate(test_loader, model, args)
            best_prec = max(prec, best_prec)
            is_best = (best_prec == prec)
            tqdm.write('Best precision: {:.03f}'.format(best_prec))
            scheduler.step()
            if is_best or (epoch + 1) % args.save_interval == 0:
                current_dict = {
                        'epoch': epoch + 1,
                        'state_dict': copy.deepcopy(model.state_dict()),
                        'best_prec': prec,
                        'optimizer': copy.deepcopy(optimizer.state_dict()),
                        'scheduler': copy.deepcopy(scheduler.state_dict())
                }
                if is_best or (not best_dict):
                    best_dict = copy.deepcopy(current_dict)
                logutils.save_checkpoint(best_dict, is_best, args)
                if (epoch + 1) % args.save_interval == 0:
                    logutils.save_checkpoint_epoch(best_dict, epoch + 1, args)
    else:
        validate(test_loader, model, args, test=True)
    if args.save:
        validate(all_loader, model, args, test=True, save=True)


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
    task = tasks[1]
    parser = argparse.ArgumentParser(description='Detection LSTM baseline')
    parser.add_argument('--dataset', default='CAD', type=str,
                        help='Indicating which dataset to use')
    parser.add_argument('--model', default='lstm', type=str,
                        help='Model for classification (default: LSTM)')
    parser.add_argument('--seed', default=12345, type=int,
                        help='Default seed for all random generators')
    parser.add_argument('--task', default=task, type=str,
                        help='Task for network training (default: affordance)')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool,
                        help='Option flag for using cuda trining (default: True)')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='Default device for cuda')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='starting epoch of training (default: 0)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size for training (default: 1)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for the feature extraction process (default: 1e-3)')
    parser.add_argument('--lr_decay', type=lambda x: restricted_float(x, [0.01, 1]), default=1,
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--lr_freq', default=20, type=float,
                        help='learing rate decay frequency while updating')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='Intervals for logging (default: 10 batch)')
    parser.add_argument('--save_interval', type=int, default=1, metavar='N',
                        help='Intervals for saving checkpoint (default: 3 epochs)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='ratio of data for training purposes (default: 0.65)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='ratio of data for validation purposes (default: 0.1)')
    parser.add_argument('--eval', default=False, type=str2bool,
                        help='indicates whether need to run evaluation on testing set')
    parser.add_argument('--save', default=False, type=str2bool,
                        help='flag for saving likelihood')
    parser.add_argument('--subsample', default=1, type=int,
                        help='subsample frequency for Breakfast dataset')
    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='Dropout rate for LSTM training')
    parser.add_argument('--trained_epochs', default=100, type=int,
                        help='The number of iterations for trained model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)