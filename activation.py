import kornia
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

class CLAHE:
    def __init__(self, types='bnw', clip_limit=2.0, grid_size=(8, 8), padding=0):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.padding = padding
        self.type = types.lower()
        if self.type not in ['bnw', 'rgb']:
            raise ValueError("Type must be either 'bnw' or 'rgb'")

    def __call__(self, img):
        if torch.rand(1).item() > self.padding:
            return img  # Skip CLAHE with probability

        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        # Add batch dimension: [C, H, W] -> [1, C, H, W]
        img_tensor = img.unsqueeze(0)

        if self.type == 'bnw':
            if img_tensor.shape[1] == 3:
                img_tensor = kornia.color.rgb_to_grayscale(img_tensor)
            img_clahe = kornia.enhance.clahe(img_tensor, clip_limit=self.clip_limit, grid_size=self.grid_size)
            return img_clahe.squeeze(0)

        elif self.type == 'rgb':
            # Apply CLAHE to each channel independently
            channels = []
            for c in range(3):
                channel = img_tensor[:, c:c+1, :, :]  # Keep shape [1,1,H,W]
                clahe_channel = kornia.enhance.clahe(channel, clip_limit=self.clip_limit, grid_size=self.grid_size)
                channels.append(clahe_channel)
            img_clahe = torch.cat(channels, dim=1)  # [1, 3, H, W]
            return img_clahe.squeeze(0)

