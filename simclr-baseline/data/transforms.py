class SimCLRAudioAugment(nn.Module):
    """
    Time-domain audio augmentations for SimCLR.

    Augmentations are applied independently to each view
    to encourage invariant representations.
    """
    def __init__(self,
                 noise_std=0.005,
                 gain_range=(0.8, 1.2),
                 p_noise=0.5,
                 p_gain=0.5):
        super().__init__()
        self.noise_std = noise_std
        self.gain_range = gain_range
        self.p_noise = p_noise
        self.p_gain = p_gain

    def forward(self, wav):
        # wav: (1, T)
        if torch.rand(1).item() < self.p_gain:
            gain = torch.empty(1).uniform_(*self.gain_range).item()
            wav = wav * gain

        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(wav) * self.noise_std
            wav = wav + noise

        return wav
