class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).

    Forward: identity
    Backward: multiplies gradients by -λ

    Used to adversarially remove demographic information
    from the residual representation.
    """
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    """
    Convenience wrapper for applying gradient reversal.

    Args:
        x: input tensor
        lambd: adversarial strength
    """
    return GradReverse.apply(x, lambd)
