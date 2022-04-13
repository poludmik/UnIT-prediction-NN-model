import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.layer_1 = nn.Linear(5, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU(0.1)
        self.layer_out = nn.Linear(256, 1)

    def forward(self, inputs):
        x = self.layer_1(inputs)
        # x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_out(x)
        x = F.sigmoid(x)
        return x