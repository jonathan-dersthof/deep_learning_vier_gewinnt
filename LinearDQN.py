import torch.nn as nn
import numpy

class LinearDQN(nn.Module):
    """ Das genutzte Lineare DQN """
    def __init__(self, hidden_size : int):
        super(LinearDQN, self).__init__()

        # Größe des Spielfelds ist fest, da es das originale Brettspiel repräsentieren soll
        rows : int = 6
        cols : int = 7
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

    """ Methode wird nicht explizit im Code, sondern nur implizit von PyTorch aufgerufen """
    def forward(self, state : numpy.ndarray):
        """ Übergibt dem Netzwerk den Spielstand. """
        return self.network(state)
