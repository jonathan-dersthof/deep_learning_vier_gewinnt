import torch


class ConvDQN(torch.nn.Module):
    def __init__(self, hidden_size = 128):
        super(ConvDQN, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        self.flatten = torch.nn.Flatten()

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(64 * 5 * 6, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 7)
        )

    def forward(self, board):
        board = self.conv_block(board)
        board = self.flatten(board)

        return self.fc_layer(board)
