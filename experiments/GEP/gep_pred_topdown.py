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
import time
import copy


# Libraries
from tqdm import tqdm
import numpy as np
import torch

# Local imports
from models.BiLSTM import BiLSTM
from models.LSTM_pred import LSTM_Pred
import models.parser.GEP_online as GEP
import models.parser.grammarutils as grammarutils
import utils.logutils as logutils
import experiments.exp_config as exp_config

def predict(parser, detection_output, duration_prior, record, frame, args, epsilon=1e-5):
    detection_output_prob = torch.nn.Softmax(dim=-1)(detection_output).data.cpu().numpy()
    parser.update_prob(detection_output_prob)
    best_l, _ = parser.parse()
    current_token = args.metadata.action_index[best_l.split()[-1]]
    if 'last' not in record.keys() or current_token != record['last']:
        record['last'] = current_token
        record['start'] = frame

    pred_duration = args.using_pred_duration
    pred_labels = list()
    predict_parser = copy.deepcopy(parser)
    mu, sigma = duration_prior[args.metadata.actions[current_token]]
    current_duration = max(0, int(mu) - (frame - record['start'] + 1))
    pred_labels.extend([current_token for _ in range(current_duration)])
    pred_duration -= current_duration
    while pred_duration > 0:
        prob = np.ones(len(args.metadata.actions)) * epsilon
        prob[current_token] = 1.0
        prob = prob / sum(prob)
        for _ in range(current_duration):
            predict_parser.update_prob(prob)
        predict_parser.parse()
        predict_mat = predict_parser.future_predict()
        current_token = np.argmax(predict_mat, axis=-1)
        mu, sigma = duration_prior[args.metadata.actions[current_token]]
        current_duration = int(mu)
        pred_duration -= current_duration
        pred_labels.extend([current_token for _ in range(current_duration)])
    pred_labels = pred_labels[: args.using_pred_duration]
    return pred_labels

def validate(data_loader, detection_model, prediction_model, args):
    all_gt_frame_predictions = list()
    all_frame_predictions = list()
    all_nn_frame_predictions = list()

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

        padding = features_batch[0, :, :].repeat(args.using_pred_duration - 1, 1, 1)
        prediction_features = torch.cat((padding, features_batch), dim=0)
        prediction_output = prediction_model(prediction_features)
        detection_output = detection_model(features_batch)

        _, detection_labels = torch.max(detection_output, dim=-1)
        detection_labels = detection_labels.cpu().numpy()

        for batch_i in range(detection_output.size(1)):

            gt_all_pred_labels = labels_batch[1: total_lengths[batch_i], batch_i].cpu().numpy().tolist()
            _, nn_all_pred_labels = torch.max(prediction_output[:total_lengths[batch_i] - 1, batch_i, :], dim=-1)
            nn_all_pred_labels = nn_all_pred_labels.cpu().numpy().tolist()

            # Initialization of Earley Parser
            class_num = detection_output.shape[2]
            grammar_file = os.path.join(args.paths.grammar_root, activities[batch_i] + '.pcfg')
            grammar = grammarutils.read_grammar(grammar_file, index=True)
            gen_earley_parser = GEP.GeneralizedEarley(grammar, class_num, mapping=args.metadata.action_index)
            with open(os.path.join(args.paths.prior_root, 'duration_prior.json')) as f:
                duration_prior = json.load(f)

            record = dict()

            start_time = time.time()
            for frame in range(total_lengths[batch_i] - args.using_pred_duration):
                nn_pred_labels = nn_all_pred_labels[frame : frame + args.using_pred_duration]
                gt_pred_labels = gt_all_pred_labels[frame : frame + args.using_pred_duration]
                update_length = len(nn_pred_labels)

                pred_labels = predict(gen_earley_parser, detection_output[frame, batch_i, :],
                                                                                    duration_prior, record, frame, args)
                # gt = torch.ones(detection_output.size(2)) * 1e-5
                # gt[labels_batch[frame, batch_i]] = 1
                # gt = torch.log(gt / torch.sum(gt))
                # pred_labels = predict(gen_earley_parser, gt,
                #                       duration_prior, record, frame, args)
                # print(frame)
                # print('detection_labels', detection_labels[max(0, frame - 44) : frame + 1, batch_i].tolist())
                # print('gt_detect labels', labels_batch[max(0, frame - 44) :frame+1, batch_i].cpu().numpy().tolist())
                # print('gt_predic_labels', gt_pred_labels)
                # print('nn_predic_labels', nn_pred_labels)
                # print('xx_predic_labels', pred_labels)

                micro_prec = logutils.compute_accuracy(gt_pred_labels, pred_labels)
                nn_micro_prec = logutils.compute_accuracy(gt_pred_labels, nn_pred_labels)
                macro_prec, macro_rec, macro_f1 = logutils.compute_accuracy(gt_pred_labels, nn_pred_labels,
                                                                            metric='macro')
                task_acc_ratio.update(micro_prec, update_length)
                task_acc_ratio_nn.update(nn_micro_prec, update_length)
                task_macro_prec.update(macro_prec, update_length)
                task_macro_rec.update(macro_rec, update_length)
                task_macro_f1.update(macro_f1, update_length)

                all_gt_frame_predictions.extend(gt_pred_labels)
                all_frame_predictions.extend(pred_labels)
                all_nn_frame_predictions.extend(nn_pred_labels)


            print(time.time() - start_time)

        tqdm.write('Task {} {} Batch [{}/{}]\t'
                   'Acc {top1.val:.4f} ({top1.avg:.4f})\t'
                   'NN Acc {nn.val:.4f} ({nn.avg:.4f})\t'
                   'Prec {prec.val:.4f} ({prec.avg:.4f})\t'
                   'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
                   'F1 {f1.val:.4f} ({f1.avg:.4f})'.format(
                args.task, 'test', batch_idx, len(data_loader), top1=task_acc_ratio, nn=task_acc_ratio_nn,
                prec=task_macro_prec, recall=task_macro_rec, f1=task_macro_f1))

    micro_prec = logutils.compute_accuracy(all_gt_frame_predictions, all_frame_predictions)
    nn_micro_prec = logutils.compute_accuracy(all_gt_frame_predictions, all_nn_frame_predictions)
    macro_prec, macro_recall, macro_fscore = logutils.compute_accuracy(all_gt_frame_predictions, all_nn_frame_predictions, metric='weighted')
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

    args.resume = os.path.join(paths.checkpoint_root, 'detection_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.epochs,
                                                                      args.lr, args.batch_size, args.lr_decay,
                                                                          1 if not args.subsample else args.subsample,
                                                                        args.dropout_rate))
    detection_model = BiLSTM(feature_size, hidden_size, hidden_layers, label_num)
    detection_model = torch.nn.DataParallel(detection_model)
    logutils.load_checkpoint(args, detection_model)

    args.resume = os.path.join(paths.checkpoint_root,
                               'frame_prediction_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}_pd{}'.format(args.task, args.epochs,
                                                                                             args.lr, args.batch_size,
                                                                                             args.lr_decay,
                                                                                             1 if not args.subsample else args.subsample,
                                                                                             args.dropout_rate,
                                                                                             args.using_pred_duration))
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
    parser.add_argument('--lr_decay', default=1, type=float,
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--lr_freq', default=25, type=float,
                        help='learing rate decay frequency while updating')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample frequency for Breakfast dataset')
    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='Dropout rate for LSTM training')
    parser.add_argument('--using_pred_duration', default=45, type=int,
                        help='Using model that is trained to predict')
    args = parser.parse_args()
    main(args)
