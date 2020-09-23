"""
Created on 12/9/18

@author: Baoxiong Jia

Description:

"""
import torch

import datasets.VCLA_GAZE.vcla_gaze_config as vcla_gaze_config
from datasets.VCLA_GAZE.metadata import VCLA_METADATA
import datasets.VCLA_GAZE.vcla_gaze as vcla_gaze

import datasets.CAD.cad_config as cad_config
from datasets.CAD.metadata import CAD_METADATA
import datasets.CAD.cad as cad

import datasets.WNP.wnp_config as wnp_config
from datasets.WNP.metadata import WNP_METADATA
import datasets.WNP.wnp as wnp

import datasets.Breakfast.breakfast_config as breakfast_config
from datasets.Breakfast.metadata import BREAKFAST_METADATA
import datasets.Breakfast.breakfast as breakfast

class Experiment(object):
    def __init__(self, dataset='VCLA_GAZE'):
        self.paths_dict = {
                                'WNP': wnp_config.Paths(),
                                'VCLA_GAZE': vcla_gaze_config.Paths(),
                                'CAD': cad_config.Paths(),
                                'Breakfast': breakfast_config.Paths()
                            }
        self.metadata_dict = {
                                'WNP': WNP_METADATA(),
                                'VCLA_GAZE': VCLA_METADATA(),
                                'CAD': CAD_METADATA(),
                                'Breakfast': BREAKFAST_METADATA()
                            }
        self.dataset_dict = {
                                'WNP': lambda path, mode, task, subsample: wnp.WNP(path, mode, task, subsample),
                                'VCLA_GAZE': lambda path, mode, task, subsample: vcla_gaze.VCLA_GAZE(path, mode, task, subsample),
                                'CAD': lambda path, mode, task, subsample: cad.CAD(path, mode, task, subsample),
                                'Breakfast': lambda path, mode, task, subsample: breakfast.Breakfast(path, mode, task, subsample)
                            }
        self.dataset = self.dataset_dict[dataset]
        self.paths = self.paths_dict[dataset]
        self.metadata = self.metadata_dict[dataset]

    def get_dataset(self, args, save=False):
        all_set = None
        train_set = self.dataset(args.paths, 'train', args.task, args.subsample)
        val_set = self.dataset(args.paths, 'val', args.task, args.subsample)
        test_set = self.dataset(args.paths, 'test', args.task, args.subsample)
        if save:
            all_set = self.dataset(args.paths, 'all', args.task, args.subsample)
        all_loader = None
        train_loader = torch.utils.data.DataLoader(train_set, collate_fn=train_set.collate_fn,
                                                   batch_size=args.batch_size, num_workers=args.workers,
                                                   pin_memory=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, collate_fn=train_set.collate_fn,
                                                 batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, collate_fn=train_set.collate_fn,
                                                  batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        if save:
            all_loader = torch.utils.data.DataLoader(all_set, collate_fn=train_set.collate_fn,
                                                 batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        features, labels, seg_lengths, total_length, activity, sequence_id, additional = train_set[0]
        feature_size = features[0].shape[-1]
        return feature_size, train_loader, val_loader, test_loader, all_loader

    def get_label_num(self, args):
        if args.task == 'affordance':
            return self.metadata.AFFORDANCE_NUM
        else:
            return self.metadata.ACTION_NUM