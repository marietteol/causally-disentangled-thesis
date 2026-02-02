class ProjectionHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim,in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x): return self.net(x)
