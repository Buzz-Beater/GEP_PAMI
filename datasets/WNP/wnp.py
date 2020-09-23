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
from datasets.WNP.metadata import  WNP_METADATA

class WNP(torch.utils.data.Dataset):
    def __init__(self, paths, mode, task='activity', subsample=None):
        self.path = paths.inter_root
        self.sequence_ids = list()
        with open(os.path.join(self.path, 'features', 'wnp_{}.p'.format(mode)), 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        for key in self.data.keys():
            self.sequence_ids.append(key)
        self.task = task
        self.mode = mode

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        return self.data[sequence_id]['features'], self.data[sequence_id]['labels'], \
               self.data[sequence_id]['seg_lengths'], self.data[sequence_id]['total_length'], \
               self.data[sequence_id]['activity'], sequence_id, None

    def __len__(self):
        return len(self.sequence_ids)

    @staticmethod
    def collate_fn(batch):
        metadata = WNP_METADATA()
        features, labels, seg_lengths, total_length, activity, sequence_id, additional = batch[0]
        feature_size = features.shape[1]
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
            features_batch[:total_length, batch_i, :] = np.nan_to_num(features)
            labels_batch[:total_length, batch_i] = labels
            for frame in range(features.shape[0]):
                probs_batch[frame, batch_i, int(labels[frame])] = 1.0

            merged_labels = list()
            current_label = -1
            for label in labels:
                if label != current_label:
                    current_label = label
                    merged_labels.append(current_label)
            ctc_labels.append(merged_labels)
            ctc_lengths.append(len(merged_labels))
            total_lengths[batch_i] = total_length
            activities.append(activity)
            sequence_ids.append(sequence_id)

        features_batch = torch.FloatTensor(features_batch)
        labels_batch = torch.LongTensor(labels_batch)
        probs_batch = torch.FloatTensor(probs_batch)
        total_lengths = torch.IntTensor(total_lengths)
        ctc_lengths = torch.IntTensor(ctc_lengths)

        # Feature_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional_label
        return features_batch, labels_batch, activities, sequence_ids, total_lengths, 0, ctc_labels, ctc_lengths, probs_batch, None