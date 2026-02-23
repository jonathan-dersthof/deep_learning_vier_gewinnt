import torch
import torch.nn as nn

class LinearDQN(nn.Module):
    def __init__(self, hidden_size):
        super(LinearDQN, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 42, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = 7)
        )

    def forward(self, board):
        return self.network(board)
