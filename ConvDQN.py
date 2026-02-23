import torch
import torch.nn as nn


class ConvDQN(nn.Module):
    def __init__(self, hidden_size = 128):
        super(ConvDQN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 5 * 6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7)  # 7 Columns output
        )

    def forward(self, board):
        board = self.conv_block(board)

        board = self.flatten(board)
        return self.fc_layer(board)
