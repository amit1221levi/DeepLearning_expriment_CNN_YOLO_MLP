import torch

from . import layers


def compare_layer_to_torch(layer: layers.Layer, x, y=None, seed=42):
    """
    Compares the manually calculated gradients of a Layer (it's backward
    function) to the gradients produced by PyTorch's autograd.
    """
    # Forward pass
    torch.manual_seed(seed)
    z = layer(x, y=y)
    # Invent some output gradient
    dz = torch.randn(*z.shape) if z.dim() > 0 else torch.tensor(1.0)
    # Backward pass (ours)
    dx = layer.backward(dz)

    # Attach autograd gradients to params and input and re-run forward pass on
    # the same input
    for t, _ in layer.params() + [(x, None)]:
        t.requires_grad = True

    torch.manual_seed(seed)
    z = layer(x, y=y)

    # Backward pass (this time with PyTorch autograd)
    z.backward(dz)

    print("Comparing gradients... ")
    diffs = []

    # Compare input gradient
    dx_autograd = x.grad
    diffs.append(torch.norm(dx_autograd - dx))
    print(f'{"input":8s} diff={diffs[-1]:.3f}')

    # Compare parameter gradients
    for i, (p, dp) in enumerate(layer.params()):
        dp_autograd = p.grad
        diffs.append(torch.norm(dp_autograd - dp))
        print(f"param#{i+1:02d} diff={diffs[-1]:.3f}")

    return diffs
