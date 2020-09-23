"""
Created on 12/2/18

@author: Baoxiong Jia

Description:

"""
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.):
        super(BiLSTM, self).__init__()
        self.hidden_layer = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features):
        # # # Initialize hidden states, 2 for bidirectional RNN
        # h0 = torch.zeros(self.num_layers * 2, features.size(1), self.hidden_layer).to(device=features.device)
        # c0 = torch.zeros(self.num_layers * 2, features.size(1), self.hidden_layer).to(device=features.device)

        # out, _ = self.lstm(features, (h0, c0))
        packed = rnn_utils.pack_sequence(features)
        out, _ = self.lstm(features)
        out = self.dropout(out)
        out = self.fc(out)
        return out