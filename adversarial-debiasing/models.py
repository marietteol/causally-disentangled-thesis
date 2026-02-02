import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBED_DIM, PROJ_DIM, TEMPERATURE

# --- CNN Encoder ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    def forward(self, x): return self.pool(self.conv(x))

class SmallCNNEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1,32), ConvBlock(32,64), ConvBlock(64,128), ConvBlock(128,256, pool=False)
        )
        self.fc = nn.Linear(256*2, embed_dim)
    def forward(self, x):
        z = self.net(x).flatten(2)
        mean, std = z.mean(-1), z.std(-1)
        return F.normalize(self.fc(torch.cat([mean,std], dim=1)), dim=-1)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim,in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x): return self.net(x)

def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=torch.bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    positives = torch.cat([torch.diag(sim,batch_size), torch.diag(sim,-batch_size)], dim=0)
    return (-torch.log(torch.exp(positives)/exp_sim.sum(dim=1))).mean()

# --- Gradient Reversal ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
def grad_reverse(x, lambda_): return GradReverse.apply(x, lambda_)

# --- Adversary ---
class Adversary(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, n_classes)
        )
    def forward(self, x): return self.net(x)

# --- MLP Probe ---
class MLPProbe(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.net(x)
