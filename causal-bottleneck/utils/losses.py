def orthogonality_loss(z_demo, z_task):
    """
    Encourages demographic and residual embeddings to be linearly uncorrelated.

    Computes Frobenius norm of the cross-covariance matrix.

    Args:
        z_demo: Tensor [B, k_demo]
        z_task: Tensor [B, k_task]

    Returns:
        Scalar loss value
    """
    z_demo_c = z_demo - z_demo.mean(dim=0, keepdim=True)
    z_task_c = z_task - z_task.mean(dim=0, keepdim=True)

    cov = (z_demo_c.T @ z_task_c) / z_demo.shape[0]
    return torch.norm(cov, p="fro")
