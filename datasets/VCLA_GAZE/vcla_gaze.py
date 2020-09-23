"""
Created on 12/3/18

@author: Baoxiong Jia

Description:

"""
import numpy as np
import os
import torch
import torch.utils.data
import pickle
import datasets.VCLA_GAZE.vcla_gaze_config as config
from datasets.VCLA_GAZE.metadata import VCLA_METADATA

class VCLA_GAZE(torch.utils.data.Dataset):
    def __init__(self, paths, mode, task, subsample=None):
        self.path = paths.inter_root
        with open(os.path.join(self.path, 'features', 'vcla_gaze_{}.p'.format(mode)), 'rb') as f:
            self.data = pickle.load(f)
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
        metadata = VCLA_METADATA()
        affordance = False
        features, labels, seg_lengths, total_length, activity, sequence_id, additional = batch[0]
        feature_dim = list(features.shape)
        if len(feature_dim) > 2:
            affordance = True
        max_seq_length = np.max(
            np.array([total_length for (features, labels, seg_lengths, total_length, activity, sequence_id, additional) in batch]))
        feature_dim[0] = max_seq_length
        feature_dim.insert(1, len(batch))  # max_length * batch * (obj_num) * feature_size
        obj_nums = np.zeros(len(batch))
        if affordance:
            max_obj_num = metadata.MAXIMUM_OBJ_VIDEO
            feature_dim[-2] = max_obj_num
            total_lengths = np.zeros(len(batch) * max_obj_num)
        else:
            total_lengths = np.zeros(len(batch))
        features_batch = np.zeros(feature_dim)
        labels_batch = np.zeros(feature_dim[: -1])
        probs_batch = np.zeros(feature_dim[: 2] + [len(metadata.subactivities)])

        activities = list()
        sequence_ids = list()
        ctc_labels = list()
        ctc_lengths = list()
        for batch_i, (features, labels, seg_lengths, total_length, activity, sequence_id, additional) in enumerate(batch):
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

            if affordance:
                obj_num = labels.shape[1]
                features_batch[:total_length, batch_i, :obj_num, :] = np.nan_to_num(features)
                labels_batch[:total_length, batch_i, :obj_num] = labels
                for rel_idx in range(3):
                    total_lengths[batch_i * 3 + rel_idx] = total_length
                obj_nums[batch_i] = obj_num
            else:
                features_batch[:total_length, batch_i, :] = np.nan_to_num(features)
                labels_batch[:total_length, batch_i] = labels
                total_lengths[batch_i] = total_length
            activities.append(activity)
            sequence_ids.append(sequence_id)

        features_batch = torch.FloatTensor(features_batch)
        labels_batch = torch.LongTensor(labels_batch)
        total_lengths = torch.IntTensor(total_lengths)
        obj_nums = torch.IntTensor(obj_nums)
        ctc_lengths = torch.IntTensor(ctc_lengths)

        # Feature_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, all_labels
        return features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, None, None


def main():
    paths = config.Paths()
    dataset = VCLA_GAZE(paths, 'train', 'affordance')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                               num_workers=1, pin_memory=True)
    features_batch, labels_batch, activities, sequence_ids, total_lengths, obj_nums, ctc_labels, ctc_lengths, probs_batch, additional = dataset[0]

    print('Finished')

if __name__ == '__main__':
    main()
