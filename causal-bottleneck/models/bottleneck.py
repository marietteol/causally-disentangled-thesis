class CausalBottleneck(nn.Module):
    """
    Causal bottleneck that decomposes representations into:

    - z_demo: demographic-related subspace
    - z_task: residual task-related subspace

    The residual is computed by reconstructing the demographic
    contribution in the original feature space and subtracting it.
    """
    def __init__(self, input_dim, k):
        """
        Args:
            input_dim: dimensionality of encoder output
            k: dimensionality of demographic bottleneck
        """
        super().__init__()
        self.linear_demo = nn.Linear(input_dim, k)
        self.linear_to_h = nn.Linear(k, input_dim)

    def forward(self, h, detach_residual=True):
        """
        Args:
            h: encoder features (B, input_dim)
            detach_residual: stops gradients through residual subtraction

        Returns:
            z_demo: demographic embedding (B, k)
            z_task: residual embedding (B, input_dim)
        """
        z_demo = self.linear_demo(h)
        h_demo = self.linear_to_h(z_demo)

        if detach_residual:
            z_task = h - h_demo.detach()
        else:
            z_task = h - h_demo

        return z_demo, z_task
