"""
  
    Created on 11/5/19

    @author: Baoxiong Jia

    Description:

"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.linear3(self.dropout(self.linear2(self.linear1(x))))