"""
Created on 11/27/18

@author: Baoxiong Jia

Description:

"""

import os
import time
import pickle
from random import shuffle
import numpy as np
import sys
sys.path.append('/media/hdd/home/baoxiong/Projects/TPAMI2019/src')

import datasets.VCLA_GAZE.vcla_gaze_config as config
from models import parsegraph as parsegraph
from datasets.VCLA_GAZE.metadata import VCLA_METADATA
metadata = VCLA_METADATA()


def parse_data(paths):
    activity_feature_path = os.path.join(paths.inter_root, 'finetune', 'activity')
    affordance_feature_path = os.path.join(paths.inter_root, 'finetune', 'affordance')
    save_path = os.path.join(paths.inter_root, 'features')

    # for STAOG formulation
    activity_corpus = dict()
    with open(os.path.join(paths.tmp_root, 'video_data_list.p'), 'rb') as f:
        data_list = pickle.load(f)
    data_dict = dict()
    for sequence_id, data in data_list.items():
        names = sequence_id.split("$")
        activity_id, video_id = names[0], names[1]
        data_dict[sequence_id] = dict()
        if activity_id not in activity_corpus.keys():
            activity_corpus[activity_id] = list()
        tpg = parsegraph.TParseGraph(activity_id, sequence_id=video_id)
        segmentation = data['segment']
        activity = data['activity_mat']
        objects = data['object_mat']
        affordance = data['affordance_mat']
        skeleton = data['skeleton_mat']
        obj_nums = data['object_num']
        data_dict[sequence_id]['total_length'] = activity.shape[0]
        data_dict[sequence_id]['labels'] = activity
        data_dict[sequence_id]['u_labels'] = affordance[:, : obj_nums]
        data_dict[sequence_id]['seg_lengths'] = list()
        data_dict[sequence_id]['activity'] = activity_id

        # feature reformat for GEP
        activity_features = None
        affordance_features = None

        start_ori = segmentation[0][0]
        for (start, end) in segmentation:
            end = end - start_ori
            start = start - start_ori
            data_dict[sequence_id]['seg_lengths'].append(end - start + 1)
            subactivity = metadata.subactivities[int(activity[start])]
            object_data = objects[start : end + 1, : obj_nums, :]
            obj_positions = [object_data[obj_idx, metadata.OBJECT_NUM : ] for obj_idx in range(obj_nums)]
            obj_names = [metadata.objects[np.argmax(object_data[0, obj_idx, : metadata.OBJECT_NUM])] for obj_idx in range(obj_nums)]
            affordance_labels = affordance[start, : obj_nums]
            affordance_labels = [metadata.affordances[int(affordance_labels[obj_idx])] for obj_idx in range(obj_nums)]
            spg = parsegraph.SParseGraph(start, end, subactivity, subactivity, obj_names, affordance_labels)
            spg.set_obj_positions(obj_positions)
            spg.set_skeletons(skeleton[start : end + 1, :])
            tpg.append_terminal(spg)

            for feature_idx in range(start, end + 1):
                image_id = sequence_id + '$' + str(feature_idx + start_ori)
                activity_feature = np.load(os.path.join(activity_feature_path, '{}.npy'.format(image_id)))
                affordance_feature = np.expand_dims(np.load(os.path.join(affordance_feature_path,
                                                                         '{}.npy'.format(image_id)))[: obj_nums, :], axis =0)
                if activity_features is None:
                    activity_features = activity_feature
                else:
                    activity_features = np.vstack((activity_features, activity_feature))
                if affordance_features is None:
                    affordance_features = affordance_feature
                else:
                    affordance_features = np.vstack((affordance_features, affordance_feature))

        data_dict[sequence_id]['features'] = activity_features
        data_dict[sequence_id]['u_features'] = affordance_features

        activity_corpus[activity_id].append(tpg)
        print('Finished processing for {}'.format(sequence_id))
    with open(os.path.join(paths.tmp_root, 'activity_corpus.p'), 'wb') as f:
        pickle.dump(activity_corpus, f)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    training_rate = 0.65
    validation_rate = 0.15
    training_num = training_rate * len(data_dict)
    validation_num = (training_rate + validation_rate) * len(data_dict)
    keys = list(data_dict.keys())
    shuffle(keys)

    training_dict = dict()
    validation_dict = dict()
    testing_dict = dict()

    for idx, key in enumerate(keys):
        if idx < training_num:
            training_dict[key] = data_dict[key]
        if idx >= training_num and idx < validation_num:
            validation_dict[key] = data_dict[key]
        if idx >= validation_num:
            testing_dict[key] = data_dict[key]

    with open(os.path.join(save_path, 'vcla_gaze_all.p'), 'wb') as f:
        pickle.dump(data_dict, f)
    with open(os.path.join(save_path, 'vcla_gaze_train.p'), 'wb') as f:
        pickle.dump(training_dict, f)
    with open(os.path.join(save_path, 'vcla_gaze_val.p'), 'wb') as f:
        pickle.dump(validation_dict, f)
    with open(os.path.join(save_path, 'vcla_gaze_test.p'), 'wb') as f:
        pickle.dump(testing_dict, f)

def main():
    paths = config.Paths()
    start_time = time.time()
    parse_data(paths)
    print('Time elapsed: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()