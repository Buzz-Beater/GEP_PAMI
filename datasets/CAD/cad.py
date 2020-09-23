"""
Created on 12/9/18

@author: Baoxiong Jia

Description:

"""
import os
import pickle
import numpy as np
import torch
import torch.utils.data
from datasets.CAD.metadata import CAD_METADATA
class CAD(torch.utils.data.Dataset):
    def __init__(self, paths, mode, task, subsample=None):
        super(CAD, self).__init__()
        self.paths = paths.inter_root
        with open(os.path.join(self.paths, 'features', 'cad_{}.p'.format(mode)), 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        self.sequence_ids = list()
        for key in self.data.keys():
            self.sequence_ids.append(key)
        self.task = task
        self.mode = mode

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        if self.task == 'affordance':
            return self.data[sequence_id]['u_features'], self.data[sequence_id]['u_labels'],\
                   self.data[sequence_id]['seg_lengths'], self.data[sequence_id]['total_length'], \
                   self.data[sequence_id]['activity'], sequence_id, None
        else:
            return self.data[sequence_id]['features'], self.data[sequence_id]['labels'], \
                   self.data[sequence_id]['seg_lengths'], self.data[sequence_id]['total_length'],\
                   self.data[sequence_id]['activity'], sequence_id, None

    def __len__(self):
        return len(self.sequence_ids)

    @staticmethod
    def collate_fn(batch):
        metadata = CAD_METADATA()
        features, labels, seg_lengths, total_length, activity, sequence_id, additional = batch[0]
        feature_size = features[0].shape[1]
        label_num = len(metadata.subactivities)

        max_seq_length = np.max(np.array([total_length for (features, labels, seg_lengths, total_length, activity, sequence_id, additional) in batch]))
        features_batch = np.zeros((max_seq_length, len(batch), feature_size))
        labels_batch = np.ones((max_seq_length, len(batch))) * -1
        probs_batch = np.zeros((max_seq_length, len(batch), label_num))
        total_lengths = np.zeros(len(batch))
        ctc_labels = list()
        ctc_lengths = list()
        activities = list()
        sequence_ids = list()

        for batch_i, (features, labels, seg_lengths, total_length, activity, sequence_id, additional) in enumerate(batch):
            current_len = 0
            ctc_labels.append(labels)
            ctc_lengths.append(len(labels))
            for seg_i, feature in enumerate(features):
                features_batch[current_len:current_len + seg_lengths[seg_i], batch_i, :] = np.repeat(features[seg_i],
                                                                                                     seg_lengths[seg_i],
                                                                                                     axis=0)
                labels_batch[current_len:current_len + seg_lengths[seg_i], batch_i] = labels[seg_i]
                probs_batch[current_len:current_len + seg_lengths[seg_i], batch_i, labels[seg_i]] = 1.0
                current_len += seg_lengths[seg_i]
            total_lengths[batch_i] = total_length
            activities.append(activity)
            sequence_ids.append(sequence_id)

        features_batch = torch.FloatTensor(features_batch)
        labels_batch = torch.LongTensor(labels_batch)
        probs_batch = torch.FloatTensor(probs_batch)
        total_lengths = torch.IntTensor(total_lengths)
        ctc_lengths = torch.IntTensor(ctc_lengths)

        return features_batch, labels_batch, activities, sequence_ids, total_lengths, 0, ctc_labels, ctc_lengths, probs_batch, None

    # @staticmethod
    # def collate_fn(batch):
    #     metadata = CAD_METADATA()
    #     features, labels, seg_lengths, total_length, activity, sequence_id, additional = batch[0]
    #     feature_size = features[0].shape[1]
    #     label_num = len(metadata.subactivities)
    #
    #     max_seq_length = len(labels)
    #     features_batch = np.zeros((max_seq_length, len(batch), feature_size))
    #     labels_batch = np.ones((max_seq_length, len(batch))) * -1
    #     probs_batch = np.zeros((max_seq_length, len(batch), label_num))
    #     total_lengths = np.zeros(len(batch))
    #     ctc_labels = list()
    #     ctc_lengths = list()
    #     activities = list()
    #     sequence_ids = list()
    #
    #     for batch_i, (features, labels, seg_lengths, total_length, activity, sequence_id, additional) in enumerate(
    #             batch):
    #         current_len = 0
    #         ctc_labels.append(labels)
    #         ctc_lengths.append(len(labels))
    #         for seg_i, feature in enumerate(features):
    #             features_batch[current_len:current_len + seg_lengths[seg_i], batch_i, :] = np.repeat(features[seg_i],
    #                                                                                                  1, axis=0)
    #             labels_batch[current_len:current_len + 1, batch_i] = labels[seg_i]
    #             probs_batch[current_len:current_len + 1, batch_i, labels[seg_i]] = 1.0
    #             current_len += 1
    #         total_lengths[batch_i] = total_length
    #         activities.append(activity)
    #         sequence_ids.append(sequence_id)
    #
    #     features_batch = torch.FloatTensor(features_batch)
    #     labels_batch = torch.LongTensor(labels_batch)
    #     probs_batch = torch.FloatTensor(probs_batch)
    #     total_lengths = torch.IntTensor(total_lengths)
    #     ctc_lengths = torch.IntTensor(ctc_lengths)
    #
    #     return features_batch, labels_batch, activities, sequence_ids, total_lengths, 0, ctc_labels, ctc_lengths, probs_batch, None