import torch
import torch.nn as nn

class ChessOutcomeModel(nn.Module):
    def __init__(self):
        super(ChessOutcomeModel, self).__init__()
        self.fc1 = nn.Linear(64, 25)
        self.fc2 = nn.Linear(25, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x