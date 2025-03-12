import torch
from torch import nn
from torch.nn import functional


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(functional.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.swish = Swish()

    def forward(self, x):
        residual = x  # Store input for residual connection
        out = self.fc(x)
        out = self.bn(out)
        out = self.swish(out)

        # Add skip connection if dimensions match
        if residual.shape == out.shape:
            out += residual

        return out
