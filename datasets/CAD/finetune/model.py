"""
Created on 12/9/18

@author: Baoxiong Jia

Description:

"""

import torch
from datasets.CAD.metadata import CAD_METADATA
metadata = CAD_METADATA()

class TaskNet(torch.nn.Module):
    def __init__(self, feature_dim, task='affordance', hidden_dim=1500):
        super(TaskNet, self).__init__()
        if task == 'affordance':
            num_classes = metadata.AFFORDANCE_NUM
        else:
            num_classes = metadata.ACTION_NUM
        self.module = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.module(x)
        output = self.fc(x)
        return features, output