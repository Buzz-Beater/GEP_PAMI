"""
Created on 12/9/18

@author: Baoxiong Jia

Description:

"""
"""
Created on 10/29/18

@author: Baoxiong Jia

Description: Extract activity and affordance feature for VCLA gaze dataset

"""
import sys
sys.path.append('/media/hdd/home/baoxiong/Projects/TPAMI2019/src')

import os
import argparse
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.optim as optim

import datasets.CAD.finetune.model as models
from datasets.CAD.metadata import CAD_METADATA
import datasets.CAD.finetune.cad_finetune as cad
import datasets.CAD.cad_config as cad_config
import utils.logutils as utils
metadata = CAD_METADATA()


def train(data_loader, model, criterion, optimizer, epoch, args):
    logger = utils.Logger()
    model.train()
    start_time = time.time()
    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='Batch Loop Training')):
        logger.data_time.update(time.time() - start_time)
        sequence_id, rgb_image, depth_image, aligned_image, activity, object_labels, object_images, affordance, skeleton, affordance_features = data_unit

        if args.task == 'affordance':
            for object_id in range(object_labels.size(1)):
                # Batch_size * object_num * 3 * 224 * 224
                object_name = metadata.objects[np.argmax(object_labels[0, object_id].numpy())]
                object_image = object_images[:, object_id, :, :, :].squeeze(1).cuda()
                # affordance (batch_size, )
                affordance_label = affordance[:, object_id].cuda()
                feature, output = model(object_image)
                loss = criterion(output, affordance_label)
                _, pred_labels = torch.max(torch.nn.Softmax(dim=-1)(output), dim=-1)
                pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()
                gt_labels = affordance_label.cpu().data.numpy().flatten().tolist()
                prec = utils.compute_accuracy(gt_labels, pred_labels, metric='micro')
                logger.multi_losses.update(object_name, loss.item(), len(sequence_id))
                logger.top1.update(object_name, prec, len(sequence_id))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            rgb_image = rgb_image.cuda()
            affordance_features = affordance_features.cuda()
            activity = activity.squeeze(1).cuda()
            feature, output = model(rgb_image, affordance_features)
            loss = criterion(output, activity)
            _, pred_labels = torch.max(torch.nn.Softmax(dim=-1)(output), dim=-1)
            pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()
            gt_labels = activity.cpu().data.numpy().flatten().tolist()
            prec = utils.compute_accuracy(gt_labels, pred_labels, metric='micro')
            logger.multi_losses.update('Activity', loss.item(), len(sequence_id))
            logger.top1.update('Activity', prec, len(sequence_id))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.batch_time.update(time.time() - start_time)
        start_time = time.time()

        if (batch_idx + 1) % args.log_interval == 0:
            tqdm.write('Task {} Epoch: [{}][{}/{}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val} ({top1.avg})'.format(
                args.task, epoch, batch_idx, len(data_loader), batch_time=logger.batch_time,
                data_time=logger.data_time, loss=logger.multi_losses, top1=logger.top1))

def validate(data_loader, model, args, test=False, save=False):
    logger = utils.Logger()
    model.eval()
    start_time = time.time()
    if save and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    all_detections = list()
    all_gt_detections = list()

    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='Batch Loop Validating' if not test else 'Batch Loop Testing')):
        logger.data_time.update(time.time() - start_time)
        sequence_id, rgb_image, depth_image, aligned_image, activity, object_labels, object_images, affordance, skeleton, affordance_features = data_unit
        features = None
        if args.task == 'affordance':
            for object_id in range(object_labels.size(1)):
                # Batch_size * object_num * 3 * 224 * 224
                object_name = metadata.objects[np.argmax(object_labels[0, object_id].numpy())]
                object_image = object_images[:, object_id, :, :, :].squeeze(1).cuda()
                # affordance (batch_size, )
                affordance_label = affordance[:, object_id].cuda()
                feature, output = model(object_image)
                _, pred_labels = torch.max(torch.nn.Softmax(dim=-1)(output), dim=-1)
                pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()
                gt_labels = affordance_label.cpu().data.numpy().flatten().tolist()
                all_detections.extend(pred_labels)
                all_gt_detections.extend(gt_labels)
                prec = utils.compute_accuracy(gt_labels, pred_labels, metric='micro')
                logger.top1.update(object_name, prec, len(sequence_id))
                if save:
                    feature = feature.detach().cpu().numpy()
                    if features is None:
                        features = feature
                    else:
                        features = np.vstack((features, feature))
        else:
            rgb_image = rgb_image.cuda()
            affordance_features = affordance_features.cuda()
            feature, output = model(rgb_image, affordance_features)
            _, pred_labels = torch.max(torch.nn.Softmax(dim=-1)(output), dim=-1)
            pred_labels = pred_labels.cpu().data.numpy().flatten().tolist()
            gt_labels = activity.cpu().data.numpy().flatten().tolist()
            all_detections.extend(pred_labels)
            all_gt_detections.extend(gt_labels)
            prec = utils.compute_accuracy(gt_labels, pred_labels, metric='micro')
            logger.top1.update('Activity', prec, len(sequence_id))
            if save:
                features = feature.detach().cpu().numpy()

        if save:
            assert(len(sequence_id) == 1)
            np.save(os.path.join(args.save_path, sequence_id[0] + '.npy'), features)

        logger.batch_time.update(time.time() - start_time)
        start_time = time.time()
        if not test:
            if (batch_idx + 1) % args.log_interval == 0:
                tqdm.write('Task {} Test: [{}/{}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    args.task, batch_idx, len(data_loader), batch_time=logger.batch_time,
                    data_time=logger.data_time, top1=logger.top1))
    if test:
        micro_prec = utils.compute_accuracy(all_gt_detections, all_detections)
        macro_prec, macro_recall, macro_fscore = utils.compute_accuracy(all_gt_detections, all_detections, metric='weighted')
        tqdm.write('Micro Prec: {}\t'
                   'Macro Precision: {}\t'
                   'Macro Recall: {}\t'
                   'Macro F-score: {}'.format(micro_prec, macro_prec, macro_recall, macro_fscore))
    return logger.top1.avg

def main(args):
    paths = args.paths
    with open(os.path.join(paths.tmp_root, 'image_list.p'), 'rb') as f:
        sequence_list = pickle.load(f)

    # Constants for training process
    feature_dim = 1500
    best_prec = 0.0

    # Split data into train/validation/test sets
    sequence_ids = np.random.permutation(sequence_list)
    train_ids = sequence_ids[: int(args.train_ratio * len(sequence_list))]
    val_ids = sequence_ids[int(args.train_ratio * len(sequence_list)):
                           int((args.train_ratio + args.val_ratio) * len(sequence_list))]
    test_ids = sequence_ids[int((args.train_ratio + args.val_ratio) * len(sequence_list)):]
    train_dataset = cad.CAD_FEATURE(paths, train_ids, 'train', task=args.task)
    val_dataset = cad.CAD_FEATURE(paths, val_ids, 'validation', task=args.task)
    test_dataset = cad.CAD_FEATURE(paths, test_ids, 'test', task=args.task)
    all_dataset = cad.CAD_FEATURE(paths, sequence_ids, 'all', task=args.task)

    model = models.TaskNet(feature_dim, task=args.task)
    if not args.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.parallel.DistributedDataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)

    if args.resume:
        if os.path.isfile(os.path.join(args.resume, 'model_best.pth')):
            tqdm.write("===> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, 'model_best.pth'))
            # Make sure that the saved model and initialized model are of same architecture
            if args.model != checkpoint['model']:
                raise ValueError('Loading model and saved model of different type')
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            tqdm.write("===> Loaded checkpoint '{}' (epoch {} prec: {})".format(args.resume, checkpoint['epoch'], best_prec))
        else:
            tqdm.write("===> No checkpoint found at '{}'".format(args.resume))

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc='Epochs Loop'):
        train(train_loader, model, criterion, optimizer, epoch, args)
        prec = validate(val_loader, model, args)
        if (epoch + 1) % args.save_interval == 0:
            best_prec = max(prec, best_prec)
            is_best = (best_prec == prec)
            tqdm.write('Best precision: {:.03f}'.format(best_prec))
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'model': args.model,
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict()
            }, is_best, args)
    if args.eval:
        validate(test_loader, model, args, test=True)
    if args.save:
        validate(all_loader, model, args, test=True, save=True)


def parse_args():
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("{} not in range [{}, {}]".format(x, inter[0], inter[1]))
        return x

    paths = cad_config.Paths()
    model_name = 'resnet'
    tasks = ['affordance', 'activity']
    task = tasks[0]

    parser = argparse.ArgumentParser(description='VCLA feature extraction')
    parser.add_argument('--task', default=task, type=str,
                        help='Default task for network training')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                        help='Option flag for using cuda trining (default: True)')
    parser.add_argument('--distributed', default=False, type=bool,
                        help='Option flag for using distributed training (default: True)')
    parser.add_argument('--model', default=model_name, type=str,
                        help='model to use when extracting features (default: resnet)')
    parser.add_argument('--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='starting epoch of training (default: 0)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='batch size for training (default: 16)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate for the feature extraction process (default: 1e-3)')
    parser.add_argument('--lr_decay', type=lambda x: restricted_float(x, [0.01, 1]),
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='Intervals for logging (default: 10 batch)')
    parser.add_argument('--save_interval', type=int, default = 1, metavar='N',
                        help='Intervals for saving checkpoint (default: 3 epochs)')

    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='ratio of data for training purposes (default: 0.65)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='ratio of data for validation purposes (default: 0.1)')

    parser.add_argument('--eval', default=False, type=bool,
                        help='indicates whether need to run evaluation on testing set')
    parser.add_argument('--save', default=False, type=bool,
                        help='flag for saving likelihood')
    args = parser.parse_args()
    args.paths = paths
    args.save_path = os.path.join(paths.inter_root, 'finetune', args.task)
    args.resume = os.path.join(paths.checkpoint_root, 'finetune', '{}'.format(model_name), '{}'.format(args.task))
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
