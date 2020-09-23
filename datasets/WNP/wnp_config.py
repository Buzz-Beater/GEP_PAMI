"""
Created on 11/27/18

@author: Baoxiong Jia

Description: Watch-n-Patch dataset config
             No feature extraction, using kernel descriptor results

"""
import os
import config

class Paths(config.Paths):
    def __init__(self):
        super(Paths, self).__init__()
        self.data_root = self.wnp_root
        self.tmp_root = os.path.join(self.tmp_root, 'wnp')

        self.inter_root = os.path.join(self.tmp_root, 'intermediate')
        if not os.path.exists(self.inter_root):
            os.makedirs(self.inter_root)

        self.log_root = os.path.join(self.tmp_root, 'log')
        self.checkpoint_root = os.path.join(self.tmp_root, 'checkpoints')

        self.grammar_root = os.path.join(self.tmp_root, 'grammar')
        self.prior_root = os.path.join(self.tmp_root, 'prior')

        self.visualize_root = os.path.join(self.tmp_root, 'visualization')
        if not os.path.exists(self.visualize_root):
            os.makedirs(self.visualize_root)
        self.metadata_root = os.path.join(self.tmp_root, 'metadata')


if __name__ == '__main__':
    a = Paths()