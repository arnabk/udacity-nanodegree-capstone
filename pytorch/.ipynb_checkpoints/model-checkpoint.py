# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

num_layers = 2

class NeuralNet(nn.Module):
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.05)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, features = x.shape
        out = self.lstm(x.reshape(1, batch_size, features))
        out = self.fc(out.reshape(batch_size, features)[:, -1, :])
        return out
