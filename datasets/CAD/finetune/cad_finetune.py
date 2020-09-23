"""
Created on 12/9/18

@author: Baoxiong Jia

Description:

"""
import os
import pickle
import torch.utils.data
import torch
from random import shuffle
import numpy as np
import datasets.CAD.cad_config as config
from datasets.CAD.metadata import CAD_METADATA
metadata = CAD_METADATA()

class CAD_FEATURE(torch.utils.data.Dataset):
    def __init__(self, paths, sequence_ids, task, verbose=False):
        self.root = paths.img_root
        self.tmp_root = paths.tmp_root
        self.inter_root = paths.inter_root
        self.task = task
        self.verbose = verbose
        self.sequence_ids = sequence_ids
        with open(os.path.join(paths.tmp_root, 'features.p'), 'rb') as f:
            self.data_list = pickle.load(f)
        with open(os.path.join(paths.tmp_root, 'label_list.p'), 'rb') as f:
            self.label_list = pickle.load(f)

    # Using framewise information for prediction purposes
    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        video_id, frame = sequence_id.split('$')
        label = self.label_list[sequence_id]
        sequence_info = self.data_list[video_id][int(frame)]
        feature = list()
        if self.task == 'affordance':
            object_affordance_feature = np.array(sequence_info['o_fea'])
            skeleton_object_feature = np.array(sequence_info['s_o_fea'])
            feature = np.hstack((object_affordance_feature, skeleton_object_feature))
        else:
            h_feature = np.array(sequence_info['h_fea'])
            # with open(os.path.join(self.inter_root, 'finetune', 'affordance'), )

        feature = torch.FloatTensor(feature)
        label = torch.LongTensor(label)
        return feature, label

    def __len__(self):
        return len(self.sequence_ids)

def main():
    paths = config.Paths()
    with open(os.path.join(paths.tmp_root, 'label_list.p'), 'rb') as f:
        sequence_ids = pickle.load(f)
    train_num = 10
    keys = list(sequence_ids.keys())
    shuffle(keys)
    train_ids = ['1130144242$4']
    train_set = CAD_FEATURE(paths, train_ids, 'affordance')
    feature, label = train_set[0]
    print('Finished')

if __name__ == '__main__':
    main()