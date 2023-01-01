import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        # TODO:
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        # ====== YOUR CODE: ======
        super().__init__()
        layers = []
        layers += [nn.Linear(in_dim, dims[0])]
        layers += [ACTIVATIONS[nonlins[0]](**ACTIVATION_DEFAULT_KWARGS[nonlins[0]]) if type(nonlins[0]) is str else nonlins[0]]
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if type(nonlins[i+1]) is str:
                act_f = ACTIVATIONS[nonlins[i+1]](**ACTIVATION_DEFAULT_KWARGS[nonlins[i+1]]) 
            else: 
                act_f = nonlins[i+1]
            layers += [act_f]
                
        self.fc_layers = nn.Sequential(*layers[:-1])
        self.out_activation = layers[-1]
        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # TODO: Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.
        # ====== YOUR CODE: ======
        assert(self.in_dim==x.shape[1])
        x = x.reshape(x.shape[0], -1)
        z = self.fc_layers(x)
        z = self.out_activation(z)
        assert(z.shape[1]==self.out_dim)
        return z
        # ========================
