"""
Created on 04/20/19

@author: Baoxiong Jia

Description:

"""

import os
import time
import json
import glob
import pickle
from random import shuffle
import numpy as np
import sys
sys.path.append('/media/hdd/home/baoxiong/Projects/TPAMI2019/src')

import datasets.Breakfast.breakfast_config as config
from datasets.Breakfast.metadata import BREAKFAST_METADATA
metadata = BREAKFAST_METADATA()


def parse_data(paths, subsample=False):
    metadata_path = os.path.join(paths.data_root, 'metadata')
    save_path = os.path.join(paths.inter_root, 'features')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(metadata_path, 'sequence_ids.json'), 'r') as f:
        sequence_ids = json.load(f)
    data_dict = dict()
    count = 0
    for sequence_id in sequence_ids:
        data_dict[sequence_id] = dict()
        activity_id, video_id = sequence_id.split('$')
        feature_path = os.path.join(paths.data_root, 'fisher_vector', activity_id)
        annotation_file = os.path.join(metadata_path, 'annotations', sequence_id + '.p')
        feature_files = glob.glob(os.path.join(feature_path, video_id + '*'))
        features = None
        for feature_file in feature_files:
            features = np.loadtxt(feature_file)[:, 1:]
            break

        frames = features.shape[0]
        subsample_freq = 1000
        total_length = features.shape[0]
        subsample_indices = None
        if subsample:
            subsample_indices = np.arange(0, frames, subsample_freq)
            features = features[subsample_indices]
        labels = np.ones(features.shape[0]) * metadata.action_index['SIL']

        data_dict[sequence_id]['features'] = features
        data_dict[sequence_id]['total_length'] = features.shape[0]
        data_dict[sequence_id]['activity'] = activity_id
        data_dict[sequence_id]['seg_lengths'] = list()

        with open(annotation_file, 'rb') as f:
            start, end, activity = pickle.load(f)

        all_labels = np.ones(total_length) * metadata.action_index['SIL']
        all_segs = list()
        if(end[-1] != total_length):
            count += 1
            if(abs(end[-1] - total_length) > 10):
                print('Feature error for {}'.format(sequence_id))
        for s, e, a in zip(start, end, activity):
            if (s > e):
                print(s, e)
                print('Error for {}'.format(sequence_id))
                exit()
            e = min(e, total_length)
            all_segs.append(e - s + 1)
            all_labels[s - 1 : e] = metadata.action_index[a]

        if subsample:
            start = 0
            end = 0
            all_segs = list()
            for idx, sub_idx in enumerate(subsample_indices):
                if idx == len(subsample_indices) - 1:
                    all_segs.append(idx - start + 1)
                    break
                if all_labels[sub_idx] == all_labels[subsample_indices[idx + 1]]:
                    end = end + 1
                else:
                    all_segs.append(end - start + 1)
                    start = end + 1
                    end = start

                labels[idx] = all_labels[sub_idx]
        else:
            labels = all_labels

        data_dict[sequence_id]['labels'] = labels
        data_dict[sequence_id]['all_labels'] = all_labels
        data_dict[sequence_id]['seg_lengths'] = all_segs
        print('Finished processing for {}, from {} to {}'.format(sequence_id, frames, data_dict[sequence_id]['total_length']))


    with open(os.path.join(metadata_path, 'train_test_split.json'), 'r') as f:
        split = json.load(f)

    for split_idx, ids in enumerate(split):
        train_dict = dict()
        test_dict = dict()
        for other_idx, other_ids in enumerate(split):
            for id in other_ids:
                if other_idx != split_idx:
                    train_dict[id] = data_dict[id]
                else:
                    test_dict[id] = data_dict[id]
        if not subsample:
            train_file = 'breakfast_train_{}_ori.p'.format(split_idx)
            test_file = 'breakfast_test_{}_ori.p'.format(split_idx)
            val_file = 'breakfast_val_{}_ori.p'.format(split_idx)
            all_file = 'breakfast_all_{}_ori.p'.format(split_idx)
        else:
            train_file = 'breakfast_train_{}_{}.p'.format(split_idx, subsample_freq)
            test_file = 'breakfast_test_{}_{}.p'.format(split_idx, subsample_freq)
            val_file = 'breakfast_val_{}_{}.p'.format(split_idx, subsample_freq)
            all_file = 'breakfast_all_{}_{}.p'.format(split_idx, subsample_freq)

        with open(os.path.join(save_path, train_file), 'wb') as f:
            pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, test_file), 'wb') as f:
            pickle.dump(test_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, val_file), 'wb') as f:
            pickle.dump(test_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_path, all_file), 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    paths = config.Paths()
    start_time = time.time()
    parse_data(paths, subsample=False)
    print('Time elapsed: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()