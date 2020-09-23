"""
Created on 10/18/18

@author: Baoxiong Jia

Description:

"""

import os
import config

class Paths(config.Paths):
    """
    Configuration of data paths
        data_root:  root folder of all videos and annotations
        tmp_root:   intermediate result for vcla_gaze dataset
    """
    def __init__(self):
        super(Paths, self).__init__()
        self.data_root = self.vcla_data_root
        self.tmp_root = os.path.join(self.tmp_root, 'vcla_gaze')

        self.inter_root = os.path.join(self.tmp_root, 'intermediate')
        if not os.path.exists(self.inter_root):
            os.makedirs(self.inter_root)

        self.log_root = os.path.join(self.tmp_root, 'log')
        self.checkpoint_root = os.path.join(self.tmp_root, 'checkpoints')
        self.vis_root = os.path.join(self.vis_root, 'vcla_gaze')
        if not os.path.exists(self.vis_root):
            os.makedirs(self.vis_root)

        self.prior_root = os.path.join(self.tmp_root, 'prior')
        if not os.path.exists(self.prior_root):
            os.makedirs(self.prior_root)

        self.grammar_root = os.path.join(self.tmp_root, 'grammar')
        self.label_root = os.path.join(self.data_root, 'labels')
        self.metadata_root = os.path.join(self.label_root, 'metadata')
        self.anno_root =os.path.join(self.label_root, 'clean_annotations')
        self.img_root = os.path.join(self.data_root, 'images')
        self.bbox_root = os.path.join(self.label_root, 'ObjBbox')
