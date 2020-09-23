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
from models.LSTM_pred import LSTM_Pred
from models.BiLSTM import BiLSTM
from models.MLP import MLP
import models.parser.GEP_old as GEP
import models.parser.grammarutils as grammarutils
import utils.logutils as logutils
import experiments.exp_config as exp_config

def inference(prob_mat, activity, sequence_id, args):
    grammar_file = os.path.join(args.paths.grammar_root, activity+'.pcfg')
    grammar = grammarutils.read_grammar(grammar_file, index=True, mapping=args.metadata.subactivity_index)
    gen_earley_parser = GEP.GeneralizedEarley(grammar)
    best_string, prob = gen_earley_parser.parse(prob_mat)
    # print([int(s) for s in best_string.split()], "{:.2e}".format(decimal.Decimal(prob)))

    # Back trace to get labels of the entire sequence
    earley_pred_labels, tokens, seg_pos = gen_earley_parser.compute_labels()
    nn_pred_labels = np.argmax(prob_mat, axis=1)
    return nn_pred_labels, earley_pred_labels, tokens, seg_pos

def predict():
    return

def validate(data_loader, detection_model, prediction_model, args):
    all_gt_frame_predictions = list()
    all_frame_predictions = list()
    all_nn_frame_predictions = list()

    task_acc_ratio = logutils.AverageMeter()
    task_acc_ratio_nn = logutils.AverageMeter()

    # switch to evaluate mode
    detection_model.eval()
    prediction_model.eval()

    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='GEP evaluation')):
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit
        detection_likelihood = torch.nn.Softmax(dim=-1)(detection_model(features_batch)).data.cpu().numpy()

        padding = features_batch[0, :, :].repeat(args.using_pred_duration - 1, 1, 1)
        prediction_features = torch.cat((padding, features_batch), dim=0)
        prediction_output = prediction_model(prediction_features)
        prediction_likelihood = torch.nn.Softmax(dim=-1)(prediction_output).data.cpu().numpy()

        for batch_i in range(features_batch.size(1)):
            _, pred_labels = torch.max(prediction_output[:total_lengths[batch_i] - 1, batch_i, :], dim=-1)
            prediction_likelihood = prediction_likelihood[:total_lengths[batch_i] - 1, batch_i, :]

            skip_size = args.using_pred_duration - args.pred_duration

            # for frame in range(0, total_lengths[batch_i]-1, skip_size):
            for frame in range(0, total_lengths[batch_i] - args.using_pred_duration, skip_size):
                det = detection_likelihood[:frame + 1, batch_i, :]
                # det = detection_likelihood[:frame+1+args.using_pred_duration, batch_i, :]
                gt_det = torch.zeros(det.shape)
                gt_det.scatter_(1, labels_batch[:frame+1,batch_i].unsqueeze(1), 1)
                gt_det = gt_det * 0.95 + (0.05/10) * torch.ones(det.shape)
                gt_det = gt_det.numpy()

                pred = prediction_likelihood[frame:frame+args.using_pred_duration, :]
                prob_mat = np.concatenate((det, pred), axis=0)
                pred_labels, batch_earley_pred_labels, batch_tokens, batch_seg_pos = inference(prob_mat, activities[batch_i],
                                                                                               sequence_ids[batch_i], args)

                # Testing
                gep_predictions = batch_earley_pred_labels[frame+1:frame+args.using_pred_duration+1]
                all_frame_predictions.extend(gep_predictions)
                nn_frame_predictions = pred_labels[frame+1:frame+args.using_pred_duration+1]
                all_nn_frame_predictions.extend(nn_frame_predictions)
                gt_frame_predictions = labels_batch[frame+1:frame + args.using_pred_duration + 1,
                                       batch_i].cpu().numpy().tolist()
                all_gt_frame_predictions.extend(gt_frame_predictions)

                video_length = len(gt_frame_predictions)
                micro_prec_nn = logutils.compute_accuracy(gt_frame_predictions, nn_frame_predictions)
                task_acc_ratio_nn.update(micro_prec_nn, video_length)

                continue
            micro_prec = logutils.compute_accuracy(all_gt_frame_predictions, all_frame_predictions)
            nn_mirco_prec = logutils.compute_accuracy(all_gt_frame_predictions, all_nn_frame_predictions)
            macro_prec, macro_recall, macro_fscore = logutils.compute_accuracy(all_gt_frame_predictions,
                                                                               all_frame_predictions,
                                                                               metric='macro')
            tqdm.write('[Evaluation] Micro Prec: {}\t'
                       'NN Precision: {}\t'
                       'Macro Precision: {}\t'
                       'Macro Recall: {}\t'
                       'Macro F-score: {}'.format(micro_prec, nn_mirco_prec, macro_prec, macro_recall, macro_fscore))


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

    args.resume = os.path.join(paths.checkpoint_root, 'detection_{}_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.model, args.epochs,
                                                                      args.lr, args.batch_size, args.lr_decay,
                                                                          1 if not args.subsample else args.subsample,
                                                                        args.dropout_rate))
    if args.model == 'lstm':
        detection_model = BiLSTM(feature_size, hidden_size, hidden_layers, label_num)
    else:
        detection_model = MLP(feature_size, hidden_size, label_num)
    detection_model = torch.nn.DataParallel(detection_model)
    logutils.load_checkpoint(args, detection_model)

    args.resume = os.path.join(paths.checkpoint_root,
                               'frame_prediction_{}_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}_pd{}'.format(args.task, args.model, args.epochs,
                                                                                             args.lr, args.batch_size,
                                                                                             args.lr_decay,
                                                                                             1 if not args.subsample else args.subsample,
                                                                                             args.dropout_rate,
                                                                                             args.using_pred_duration))
    if args.model == 'lstm':
        prediction_model = LSTM_Pred(feature_size, hidden_size, hidden_layers, label_num)
    else:
        prediction_model = MLP(feature_size, hidden_size, label_num)
    prediction_model = torch.nn.DataParallel(prediction_model)
    logutils.load_checkpoint(args, prediction_model)

    validate(test_loader, detection_model, prediction_model, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CAD', type=str,
                        help='indicating which dataset to use')
    parser.add_argument('--model', default='lstm', type=str,
                        help='Model for classification (default: LSTM)')
    parser.add_argument('--seed', default=12345, type=int,
                        help='Default seed for all random generators')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                        help='Option flag for using cuda trining (default: True)')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--task', default='activity', type=str,
                        help='Default working task activity/affordance')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size for training (default: 1)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for the feature extraction process (default: 1e-3)')
    parser.add_argument('--lr_decay', default=1,
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--lr_freq', default=25, type=float,
                        help='learing rate decay frequency while updating')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample frequency for Breakfast dataset')
    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='Dropout rate for LSTM training')
    parser.add_argument('--pred_duration', default=45, type=int,
                        help='length of frame prediction')
    parser.add_argument('--using_pred_duration', default=55, type=int,
                        help='Using model that is trained to predict')
    args = parser.parse_args()
    main(args)
