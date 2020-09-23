"""
Created on 10/18/18

@author: Baoxiong Jia

Description:

"""

import os

class Paths(object):

    def __init__(self):
        self.project_root = '/mnt/hdd/home/baoxiong/Projects/TPAMI2019'
        self.vcla_data_root = '/mnt/hdd/home/baoxiong/Datasets/VCLA/'
        self.wnp_root = '/mnt/hdd/home/baoxiong/Datasets/Watch-n-Patch/'
        self.cad_root = '/mnt/hdd/home/baoxiong/Datasets/CAD120/'
        self.breakfast_root = '/mnt/hdd/home/baoxiong/Datasets/Breakfast/'

        self.tmp_root = os.path.join(self.project_root, 'tmp')
        if not os.path.exists(self.tmp_root):
            os.makedirs(self.tmp_root)
        self.vis_root = os.path.join(self.project_root, 'vis')
        if not os.path.exists(self.vis_root):
            os.makedirs(self.vis_root)
        self.log_root = os.path.join(self.project_root, 'log')
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
