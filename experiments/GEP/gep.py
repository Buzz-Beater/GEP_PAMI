"""
Created on 12/11/18

@author: Baoxiong Jia

Description:

"""

# System imports
import sys
sys.path.append('/mnt/hdd/home/baoxiong/Projects/TPAMI2019/src')

import os
import argparse
import json
from tqdm import tqdm

# Libraries
import numpy as np
import torch

# Local imports
import models.BiLSTM as lstm_model
import models.MLP as mlp_model
import models.parser.GEP_adj as GEP
import models.parser.grammarutils as grammarutils
import utils.logutils as logutils
import utils.evalutils as evalutils
import utils.vizutils as vizutils
import experiments.exp_config as exp_config

def inference(model_outputs, activities, sequence_ids, args):
    model_output_probs = torch.nn.Softmax(dim=-1)(model_outputs)
    model_output_probs = model_output_probs.data.cpu().numpy()
    batch_earley_pred_labels = list()
    batch_tokens = list()
    batch_seg_pos = list()
    for batch_i in range(model_outputs.size()[1]):
        grammar_file = os.path.join(args.paths.grammar_root, activities[batch_i]+'.pcfg')
        grammar = grammarutils.read_grammar(grammar_file, index=True, mapping=args.metadata.action_index)
        gen_earley_parser = GEP.GeneralizedEarley(grammar)
        best_string, prob = gen_earley_parser.parse(model_output_probs[:, batch_i, :])
        # print([int(s) for s in best_string.split()], "{:.2e}".format(decimal.Decimal(prob)))

        # Back trace to get labels of the entire sequence
        earley_pred_labels, tokens, seg_pos = gen_earley_parser.compute_labels()
        batch_earley_pred_labels.append(earley_pred_labels)
        batch_tokens.append(tokens)
        batch_seg_pos.append(seg_pos)

    _, nn_pred_labels = torch.max(model_outputs, dim=2)

    return nn_pred_labels, batch_earley_pred_labels, batch_tokens, batch_seg_pos

def validate(data_loader, model, args):
    all_gt_detections = list()
    all_detections = list()

    task_acc_ratio = logutils.AverageMeter()
    task_macro_prec = logutils.AverageMeter()
    task_macro_rec = logutils.AverageMeter()
    task_macro_f1 = logutils.AverageMeter()
    task_acc_ratio_nn = logutils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, data_unit in enumerate(tqdm(data_loader, desc='GEP evaluation')):
        features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = data_unit
        print(os.path.join(args.save_path,  '{}_out_s{}_b{}_c{}.npy'.format(sequence_ids[0],
                                 args.subsample, args.using_batch_size, args.trained_epochs)))
        # exit()
        model_outputs = torch.tensor(np.load(os.path.join(args.save_path,  '{}_out_s{}_b{}_c{}.npy'.format(sequence_ids[0],
                                 args.subsample, args.using_batch_size, args.trained_epochs)))).unsqueeze(1)

        # Inference
        tqdm.write('[{}] Inference'.format(sequence_ids[0]))

        seg_path = os.path.join(args.paths.inter_root, 'segmentation')
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        # # If no prior model outputs are provided
        # if not os.path.isfile(os.path.join(seg_path, '{}.npy'.format(sequence_ids[0]))):
        #     _, nn_pred_labels = torch.max(model_outputs, dim=-1)
        #     nn_detections = nn_pred_labels.cpu().data.numpy().flatten().tolist()
        #     pred_labels, batch_earley_pred_labels, batch_tokens, batch_seg_pos = inference(model_outputs, activities, sequence_ids, args)
        #
        #     # Evaluation
        #     # Frame-wise detection
        #     detections = [l for pred_labels in batch_earley_pred_labels for l in pred_labels.tolist()]
        #     if args.subsample != 1:
        #         all_total_labels, all_total_lengths = additional
        #         gt_detections = all_total_labels[:all_total_lengths[0]].flatten().tolist()
        #         video_length = len(gt_detections)
        #
        #         detections = evalutils.upsample(detections, freq=args.subsample, length=video_length)
        #         nn_detections = evalutils.upsample(nn_detections, freq=args.subsample, length=video_length)
        #     else:
        #         gt_detections = labels_batch[:total_lengths[0]].cpu().data.numpy().flatten().tolist()
        #         detections = detections[:total_lengths[0]]
        #     np.save(os.path.join(args.paths.inter_root, 'segmentation', '{}.npy'.format(sequence_ids[0])),
        #             [gt_detections, nn_detections, detections])
        # else:
        #     results = np.load(os.path.join(seg_path, '{}.npy'.format(sequence_ids[0])))
        #     gt_detections, nn_detections, detections = results[0], results[1], results[2]

        _, nn_pred_labels = torch.max(model_outputs, dim=-1)
        nn_detections = nn_pred_labels.cpu().data.numpy().flatten().tolist()
        pred_labels, batch_earley_pred_labels, batch_tokens, batch_seg_pos = inference(model_outputs, activities, sequence_ids, args)

        # Evaluation
        # Frame-wise detection
        detections = [l for pred_labels in batch_earley_pred_labels for l in pred_labels.tolist()]
        if args.subsample != 1:
            all_total_labels, all_total_lengths = additional
            gt_detections = all_total_labels[:all_total_lengths[0]].flatten().tolist()
            video_length = len(gt_detections)

            detections = evalutils.upsample(detections, freq=args.subsample, length=video_length)
            nn_detections = evalutils.upsample(nn_detections, freq=args.subsample, length=video_length)
        else:
            gt_detections = labels_batch[:total_lengths[0]].cpu().data.numpy().flatten().tolist()
            detections = detections[:total_lengths[0]]
        video_length = len(gt_detections)

       	# # Visualization code for figures
        # vizutils.plot_segmentation([gt_detections, nn_detections, detections], video_length,
        #                            filename=os.path.join(args.paths.visualize_root, '{}.jpg'.format(sequence_ids[0])), border=False)

        micro_prec = logutils.compute_accuracy(gt_detections, detections)
        micro_prec_nn = logutils.compute_accuracy(gt_detections, nn_detections)
        macro_prec, macro_rec, macro_f1 = logutils.compute_accuracy(gt_detections, detections, metric='macro')
        task_acc_ratio.update(micro_prec, video_length)
        task_acc_ratio_nn.update(micro_prec_nn, video_length)
        task_macro_prec.update(macro_prec, video_length)
        task_macro_rec.update(macro_rec, video_length)
        task_macro_f1.update(macro_f1, video_length)

        all_gt_detections.extend(gt_detections)
        all_detections.extend(detections)

        micro_prec = logutils.compute_accuracy(all_gt_detections, all_detections)
        macro_prec, macro_recall, macro_fscore = logutils.compute_accuracy(all_gt_detections, all_detections,
                                                                           metric='macro')
        tqdm.write('[Evaluation] Micro Prec: {}\t'
                   'Macro Precision: {}\t'
                   'Macro Recall: {}\t'
                   'Macro F-score: {}'.format(micro_prec, macro_prec, macro_recall, macro_fscore))

    micro_prec = logutils.compute_accuracy(all_gt_detections, all_detections)
    macro_prec, macro_recall, macro_fscore = logutils.compute_accuracy(all_gt_detections, all_detections, metric='macro')
    tqdm.write('Detection:\n'
               'Micro Prec: {}\t'
               'NN Prec:{}\t'
               'Macro Precision: {}\t'
               'Macro Recall: {}\t'
               'Macro F-score: {}\n\n'.format(micro_prec, task_acc_ratio_nn.avg, macro_prec, macro_recall, macro_fscore))

def main(args):
    exp_info = exp_config.Experiment(args.dataset)
    paths = exp_info.paths
    args.paths = paths
    args.metadata = exp_info.metadata

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = 1
    feature_size, train_loader, val_loader, test_loader, all_loader = exp_info.get_dataset(args, save=True)
    label_num = exp_info.get_label_num(args)

    hidden_size = 256
    hidden_layers = 2
    if args.model == 'lstm':
        parsing_model = lstm_model.BiLSTM(feature_size, hidden_size, hidden_layers, label_num)
    else:
        parsing_model = mlp_model.MLP(feature_size, hidden_size, label_num)
    parsing_model = torch.nn.DataParallel(parsing_model)
    prev = args.subsample
    args.subsample = 1
    args.save_path = os.path.join(paths.inter_root, 'likelihood', args.task, args.model)
    args.resume = os.path.join(paths.checkpoint_root,
                               'detection_{}_{}_e{}_lr{}_b{}_lrd{}_s{}_do{}'.format(args.task, args.model, args.epochs,
                                                                      args.lr, args.using_batch_size, args.lr_decay,
                                                                          1 if not args.subsample else args.subsample,
                                                                        args.dropout_rate))
    args.subsample = prev
    logutils.load_checkpoint(args, parsing_model)
    validate(test_loader, parsing_model, args=args)

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
    parser.add_argument('--using_batch_size', default=1, type=int, metavar='N',
                        help='using model trained on args.using_batch_size')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate for the feature extraction process (default: 1e-3)')
    parser.add_argument('--lr_decay', default=1,
                        help='decay rate of learning rate (default: between 0.01 and 1)')
    parser.add_argument('--lr_freq', default=25, type=float,
                        help='learing rate decay frequency while updating')
    parser.add_argument('--subsample', default=1, type=int,
                        help='subsample frequency for Breakfast dataset')
    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='Dropout rate for LSTM training')
    parser.add_argument('--trained_epochs', default=100, type=int,
                        help='The number of iterations for trained model')
    args = parser.parse_args()
    main(args)
