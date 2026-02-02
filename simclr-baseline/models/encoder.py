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
    """
    Convolutional encoder for log-mel spectrograms.

    Input:
        (B, 1, n_mels, time)

    Output:
        (B, embed_dim) L2-normalized embeddings
    """
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
