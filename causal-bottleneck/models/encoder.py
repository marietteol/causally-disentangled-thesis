class SmallCNNEncoder(nn.Module):
    """
    CNN-based encoder for log-mel spectrogram inputs.

    Architecture:
    - Stacked convolutional blocks
    - Temporal statistics pooling (mean + std)
    - Linear projection to embedding space

    Input:
        x: Tensor of shape (B, 1, n_mels, time)

    Output:
        embeddings: Tensor of shape (B, embed_dim)

    Notes:
        - Uses statistics pooling instead of temporal flattening
        - No normalization is applied (intentional for Phase-1 training)
        - Output magnitude is scaled to reduce representational collapse
    """
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),
        )

        # Stats pooling doubles channel dimension (mean + std)
        self.fc = nn.Linear(256 * 2, embed_dim)
        self.output_dim = embed_dim

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: log-mel spectrograms (B, 1, n_mels, time)

        Returns:
            Unnormalized embedding vectors (B, embed_dim)
        """
        z = self.net(x).flatten(2)          # (B, C, T)
        mean = z.mean(dim=-1)
        std  = z.std(dim=-1)

        fc_in = torch.cat([mean, std], dim=1)
        return self.fc(fc_in) * 10.0

  def freeze(module):
    """
    Freezes all parameters of a module (no gradient updates).

    Args:
        module: torch.nn.Module
    """
    for p in module.parameters():
        p.requires_grad = False
