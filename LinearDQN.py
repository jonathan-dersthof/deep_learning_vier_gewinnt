import torch
import torch.nn as nn
import numpy

class LinearDQN(nn.Module):
    def __init__(self, hidden_size : int):
        super(LinearDQN, self).__init__()

        rows, cols = 6, 7
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = rows * cols, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = cols)
        )

    def forward(self, state : numpy.ndarray):
        return self.network(state)
