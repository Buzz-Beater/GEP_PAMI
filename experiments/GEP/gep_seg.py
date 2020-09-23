"""
Created on 5/21/19

@author: Baoxiong Jia

Description:

"""


# System imports
import sys
sys.path.append('/media/hdd/home/baoxiong/Projects/TPAMI2019/src')

import os
import argparse
import json
from tqdm import tqdm

# Libraries
import numpy as np
import torch

# Local imports
from models.BiLSTM import BiLSTM
from models.LSTM_pred import LSTM_Pred
import models.parser.GEP_online as GEP
import models.parser.grammarutils as grammarutils
import utils.logutils as logutils
import utils.evalutils as evalutils
import experiments.exp_config as exp_config

def predict(detection_outputs, activities, total_lengths, args):
    detection_outputs_probs = torch.nn.Softmax(dim=-1)(detection_outputs)
    detection_outputs_probs = detection_outputs_probs.data.cpu().numpy()
    class_num = detection_outputs_probs.shape[2]
    pred_probs = np.empty_like(detection_outputs_probs)
    for batch_i in range(detection_outputs_probs.shape[1]):
        grammar_file = os.path.join(args.paths.grammar_root, activities[batch_i] + '.pcfg')
        grammar = grammarutils.read_grammar(grammar_file, index=True)
        gen_earley_parser = GEP.GeneralizedEarley(grammar, class_num, mapping=args.metadata.action_index)
        for frame in range(total_lengths[batch_i]):
            gen_earley_parser.update_prob(detection_outputs_probs[frame, batch_i, :])
            gen_earley_parser.parse()
            pred_probs[frame, batch_i, :] = gen_earley_parser.future_predict(args.epsilon)
    return pred_probs

def get_gt_pred(labels, total_lengths):
    all_gt_pred_labels = list()
    for i_batch in range(labels.size(1)):
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

def validate(data_loader, detection_model, prediction_model, args):
    all_gt_segment_predictions = list()
    all_segment_predictions = list()
    all_nn_segment_predictions = list()

    task_acc_ratio = logutils.AverageMeter()
    task_macro_prec = logutils.AverageMeter()
    task_macro_rec = logutils.AverageMeter()
    task_macro_f1 = logutils.AverageMeter()
    task_acc_ratio_nn = logutils.AverageMeter()

    # switch to evaluate mode
    detection_model.eval()
    prediction_model.eval()

    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='GEP evaluation')):
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit

        prediction_output = prediction_model(features_batch)
        detection_output = detection_model(features_batch)

        pred_probs = predict(detection_output, activities, total_lengths, args)
        pred_labels = np.argmax(pred_probs * prediction_output.data.cpu().numpy(), axis=-1).flatten().tolist()

        _, nn_pred_labels = torch.max(prediction_output, dim=-1)
        gt_pred_labels = get_gt_pred(labels_batch, total_lengths)
        video_length = len(gt_pred_labels)
        nn_pred_labels = nn_pred_labels.cpu().data.numpy().flatten().tolist()

        micro_prec = logutils.compute_accuracy(gt_pred_labels, pred_labels)
        nn_micro_prec = logutils.compute_accuracy(gt_pred_labels, nn_pred_labels)
        macro_prec, macro_rec, macro_f1 = logutils.compute_accuracy(gt_pred_labels, nn_pred_labels, metric='macro')
        task_acc_ratio.update(micro_prec, video_length)
        task_acc_ratio_nn.update(nn_micro_prec, video_length)
        task_macro_prec.update(macro_prec, video_length)
        task_macro_rec.update(macro_rec, video_length)
        task_macro_f1.update(macro_f1, video_length)

        all_gt_segment_predictions.extend(gt_pred_labels)
        all_segment_predictions.extend(pred_labels)
        all_nn_segment_predictions.extend(nn_pred_labels)

        tqdm.write('Task {} {} Batch [{}/{}]\t'
                   'Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                   'NN Acc {nn.val:.4f} ({nn.avg:.4f})\t'
                   'Prec {prec.val:.4f} ({prec.avg:.4f})\t'
                   'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
                   'F1 {f1.val:.4f} ({f1.avg:.4f})'.format(
                args.task, 'test', batch_idx, len(data_loader), top1=task_acc_ratio, nn=task_acc_ratio_nn,
                prec=task_macro_prec, recall=task_macro_rec, f1=task_macro_f1))

    micro_prec = logutils.compute_accuracy(all_gt_segment_predictions, all_segment_predictions)
    nn_micro_prec = logutils.compute_accuracy(all_gt_segment_predictions, all_nn_segment_predictions)
    macro_prec, macro_recall, macro_fscore = logutils.compute_accuracy(all_gt_segment_predictions, all_segment_predictions, metric='weighted')
    tqdm.write('[Evaluation] Micro Prec: {}\t'
               'NN Micro Prec: {}\t'
               'Macro Precision: {}\t'
               'Macro Recall: {}\t'
               'Macro F-score: {}'.format(micro_prec, nn_micro_prec, macro_prec, macro_recall, macro_fscore))

def main(args):
    exp_info = exp_config.Experiment(args.dataset)
    paths = exp_info.paths
    args.paths = paths
    args.metadata = exp_info.metadata

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batch_size = args.batch_size
    args.batch_size = 1
    feature_size, train_loader, val_loader, test_loader, all_loader = exp_info.get_dataset(args, save=True)
    label_num = exp_info.get_label_num(args)

    hidden_size = 256
    hidden_layers = 2
    args.save_path = os.path.join(paths.inter_root, 'likelihood', args.task)
    args.resume = os.path.join(paths.checkpoint_root,
                               'detection_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.epochs,
                                                                                 args.lr, args.batch_size,
                                                                                 args.lr_decay,
                                                                                 1 if not args.subsample else args.subsample,
                                                                                 args.dropout_rate))
    detection_model = BiLSTM(feature_size, hidden_size, hidden_layers, label_num)
    detection_model = torch.nn.DataParallel(detection_model)
    logutils.load_checkpoint(args, detection_model)

    args.resume = os.path.join(paths.checkpoint_root,
                               'segment_prediction_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.epochs,
                                                                                             args.lr, args.batch_size,
                                                                                             args.lr_decay,
                                                                                             1 if not args.subsample else args.subsample,
                                                                                             args.dropout_rate))
    prediction_model = LSTM_Pred(feature_size, hidden_size, hidden_layers, label_num)
    prediction_model = torch.nn.DataParallel(prediction_model)
    logutils.load_checkpoint(args, prediction_model)

    validate(test_loader, detection_model, prediction_model, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='VCLA_GAZE', type=str,
                        help='indicating which dataset to use')
    parser.add_argument('--seed', default=12345, type=int,
                        help='Default seed for all random generators')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                        help='Option flag for using cuda trining (default: True)')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--task', default='activity', type=str,
                        help='Default working task activity/affordance')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size for training (default: 1)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for the feature extraction process (default: 1e-3)')
    parser.add_argument('--lr_decay', default=1., type=float,
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--lr_freq', default=25, type=float,
                        help='learing rate decay frequency while updating')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample frequency for Breakfast dataset')
    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='Dropout rate for LSTM training')
    parser.add_argument('--epsilon', default=1e-10, type=float,
                        help='Balance between top-down bottom-up prediction')
    args = parser.parse_args()
    main(args)
