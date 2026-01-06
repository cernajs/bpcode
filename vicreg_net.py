import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhiNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def ema_update(target, source, decay):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
    # invariance
    sim_loss = F.mse_loss(z1, z2)

    # variance
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean())

    # covariance
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov_z1 = (z1c.T @ z1c) / (z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / (z2c.shape[0] - 1)
    
    def off_diag(m):
        return m.flatten()[:-1].view(m.shape[0] - 1, m.shape[1] + 1)[:, 1:].flatten()

    cov = off_diag(cov_z1).pow(2).mean() + off_diag(cov_z2).pow(2).mean()
    return sim_weight * sim_loss + var_weight * var + cov_weight * cov


def random_shift(x, pad = 4):
    n, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    eps_h = torch.randint(-pad, pad + 1, (n, 1, 1, 1), device=x.device)
    eps_w = torch.randint(-pad, pad + 1, (n, 1, 1, 1), device=x.device)
    return x[:, :, pad + eps_h:pad + eps_h + h, pad + eps_w:pad + eps_w + w]

def mild_jitter(x, brightness=0.4):
    return x + (torch.rand((x.shape[0], 1, 1, 1), device=x.device) * 2 - 1) * brightness

def phi_augment(x):
    x = random_shift(x)
    x = mild_jitter(x)
    return x