import torch
from torch import nn
from torch.distributions import Laplace
import pytorch_lightning as pl
from typing import Literal
import torch.nn.functional as F

# Laplace NLL Loss Function
class LaplaceNLL:
    def __init__(
        self,
        non_negativity_fcn: Literal["exp", "softplus"] = "sofplus",
        l2_on_log_scale: bool = False,
        max_scale: int =10.0
        ):

        assert non_negativity_fcn in ("exp", "softplus"), "non_negativity_fcn must be either 'exp' or 'softplus'"

        if non_negativity_fcn == "softplus":
            self.scale_fn = lambda log_scale: F.softplus(log_scale, threshold=max_scale) + 1e-9
        else:
            self.scale_fn = lambda log_scale: torch.clamp(torch.exp(log_scale), max=max_scale) + 1e-9

        self.l2_reg = l2_on_log_scale
        
    def __call__(
        self, 
        mean: torch.Tensor,
        log_scale: torch.Tensor, 
        target: torch.Tensor
        ):

        scale = self.scale_fn(log_scale=log_scale)
        dist = Laplace(mean, scale)
        nll = -dist.log_prob(target)
        nll_loss = nll.mean()

        if self.l2_reg: 
            l2_regularization = 1e-3 * torch.mean(log_scale**2)
            total_loss = nll_loss + l2_regularization
            return total_loss
        
        return nll_loss


def rearrange_kspace(kspace: torch.Tensor, imag: Literal[0,1] = 0): 
    return kspace.permute(0,3,1,2)[:,0+imag:1+imag,:,:]


class LaplaceConv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True, eps=1e-6):
        """
        Outputs parameters (mean, log_scale) of a Laplace distribution for each pixel.
        Normalization added to make training more robust.
        """
        super(LaplaceConv2dBN, self).__init__()
        self.eps = eps
        
        # Define convolution layers for mean and log_scale
        self.conv_mean = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_log_scale = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Add Batch Normalization for both conv layers (to stabilize outputs)
        self.bn_mean = nn.BatchNorm2d(out_channels)
        self.bn_log_scale = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Compute mean and log_scale
        mean = self.conv_mean(x)
        mean = self.bn_mean(mean)  # Normalize mean

        log_scale = self.conv_log_scale(x)
        log_scale = self.bn_log_scale(log_scale)  # Normalize log_scale
        
        # # Clamp log_scale to avoid extreme values and numerical issues
        print("Clamp: ", torch.log(torch.tensor(self.eps, device=log_scale.device)))
        log_scale = torch.clamp(log_scale, min=torch.log(torch.tensor(self.eps, device=log_scale.device)))
        # print("log_scale")

        return mean, log_scale

# LaplaceConv2d Definition
class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True):
        """
        Outputs parameters (mean, log_scale) of a Laplace distribution for each pixel.
        """
        super(LaplaceConv2d, self).__init__()
        self.conv_mean = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_log_scale = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        mean = self.conv_mean(x)
        log_scale = self.conv_log_scale(x)  # Log of the scale parameter
        return mean, log_scale

class MaskedConv2d(nn.Conv2d):
    """A Conv2d layer masked to respect the autoregressive property.

    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_type='A', the
    computation of the current pixel does not depend on itself.

    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_type='A'      [1 0 0],    mask_type='B'     [1 1 0],
                           [0 0 0]]                      [0 0 0]
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    
class GatedActivation(nn.Module):
    """Gated activation function as introduced in [2].

    The function computes activation_fn(f) * sigmoid(g). The f and g correspond to the
    top 1/2 and bottom 1/2 of the input channels.
    """

    def __init__(self, activation_fn=torch.tanh):
        """Initializes a new GatedActivation instance.

        Args:
            activation_fn: Activation to use for the top 1/2 input channels.
        """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, "x must have an even number of channels."
        x, gate = x[:, : c // 2, :, :], x[:, c // 2 :, :, :]
        return self._activation_fn(x) * torch.sigmoid(gate)
    

class CausalResidualBlock(nn.Module):
    """A residual block masked to respect the autoregressive property."""

    def __init__(self, n_channels):
        """Initializes a new CausalResidualBlock instance.

        Args:
            n_channels: The number of input (and output) channels.
        """
        super().__init__()
        self._net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            ),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU(),
            MaskedConv2d(
                mask_type='B',
                in_channels=n_channels // 2,
                out_channels=n_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=n_channels // 2, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        return x + self._net(x)
    
class GatedPixelCNNLayer(nn.Module):
    """A Gated PixelCNN layer.

    The layer takes as input 'vstack' and 'hstack' from previous
    'GatedPixelCNNLayers' and returns 'vstack', 'hstack', 'skip' where 'skip' is
    the skip connection to the pre-logits layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False):
        """Initializes a new GatedPixelCNNLayer instance.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of output channels.
            kernel_size: The size of the (causal) convolutional kernel to use.
            mask_center: Whether the 'GatedPixelCNNLayer' is causal. If 'True', the
                center pixel is masked out so the computation only depends on pixels to
                the left and above. The residual connection in the horizontal stack is
                also removed.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size cannot be even"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activation = GatedActivation()
        self._kernel_size = kernel_size
        self._padding = (kernel_size - 1) // 2  # (kernel_size - stride) / 2
        self._mask_center = mask_center

        # Vertical stack convolutions.
        self._vstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=(1, self._kernel_size),
            padding=(0, self._padding),
        )
        # TODO(eugenhotaj): Is it better to shift down the the vstack_Nx1 output
        # instead of adding extra padding to the convolution? When we add extra
        # padding, the cropped output rows will no longer line up with the rows of
        # the vstack_1x1 output.
        self._vstack_Nx1 = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(self._kernel_size // 2 + 1, 1),
            padding=(self._padding + 1, 0),
        )
        self._vstack_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2 * out_channels, kernel_size=1
        )

        self._link = nn.Conv2d(
            in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=1
        )

        # Horizontal stack convolutions.
        self._hstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(1, self._kernel_size // 2 + 1),
            padding=(0, self._padding + int(self._mask_center)),
        )
        self._hstack_residual = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
        self._hstack_skip = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, vstack_input, hstack_input):
        """Computes the forward pass.

        Args:
            vstack_input: The input to the vertical stack.
            hstack_input: The input to the horizontal stack.
        Returns:
            (vstack,  hstack, skip) where vstack and hstack are the vertical stack and
            horizontal stack outputs respectively and skip is the skip connection
            output.
        """
        _, _, h, w = vstack_input.shape  # Assuming NCHW.

        # Compute vertical stack.
        vstack = self._vstack_Nx1(self._vstack_1xN(vstack_input))[:, :, :h, :]
        link = self._link(vstack)
        vstack = vstack + self._vstack_1x1(vstack_input)
        vstack = self._activation(vstack)

        # Compute horizontal stack.
        hstack = link + self._hstack_1xN(hstack_input)[:, :, :, :w]
        hstack = self._activation(hstack)
        skip = self._hstack_skip(hstack)
        hstack = self._hstack_residual(hstack)
        # NOTE(eugenhotaj): We cannot use a residual connection for causal layers
        # otherwise we'll have access to future pixels.
        if not self._mask_center:
            hstack = hstack + hstack_input

        return vstack, hstack, skip
       

class PixelCNN(nn.Module):
    def __init__(self, in_channels=1, n_layers=8, hidden_channels=64):
        super().__init__()
        self._in_channels = in_channels
        self._n_layers = n_layers
        self._kernel = 7
        self._channels = hidden_channels

        self._input = nn.Sequential(MaskedConv2d('A', self._in_channels, self._channels, self._kernel, 1, self._kernel//2, bias=False), nn.BatchNorm2d(self._channels), nn.ReLU(True))

        self._layers = nn.ModuleList([
                nn.Sequential(MaskedConv2d('B',self._channels, self._channels, self._kernel, 1, self._kernel//2, bias=False), nn.BatchNorm2d(self._channels), nn.ReLU(True)) for _ in range(self._n_layers - 1)
            ])

        self._head = LaplaceConv2d(self._channels, self._in_channels, kernel_size=1, stride=1, padding=0)  # Outputs mean and log_scale

    def forward(self, x):
        x = self._input(x)
        for layer in self._layers:
            x = layer(x)
        return self._head(x)
    

class ResidualPixelCNN(nn.Module):
    """The PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=256,
        n_residual=15,
        residual_channels=128,
        head_channels=32,
        sample_fn=None,
        *args, 
        **kwargs
    ):
        """Initializes a new PixelCNN instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_residual: The number of residual blocks.
            residual_channels: The number of channels to use in the residual layers.
            head_channels: The number of channels to use in the two 1x1 convolutional
                layers at the head of the network.
            sample_fn: See the base class.
        """
        super().__init__(*args, **kwargs)
        self._input = MaskedConv2d(
            mask_type='A',
            in_channels=in_channels,
            out_channels=2 * residual_channels,
            kernel_size=7,
            padding=3,
        )
        self._causal_layers = nn.ModuleList(
            [
                CausalResidualBlock(n_channels=2 * residual_channels)
                for _ in range(n_residual)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * residual_channels,
                out_channels=head_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            LaplaceConv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            )
        )

    def forward(self, x):
        x = self._input(x)
        for layer in self._causal_layers:
            x = x + layer(x)
        return self._head(x)

class GatedPixelCNN(nn.Module):
    """The Gated PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_gated=10,
        gated_channels=128,
        head_channels=32,
        sample_fn=None,
        *args, 
        **kwargs
    ):
        """Initializes a new GatedPixelCNN instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_gated: The number of gated layers (not including the input layers).
            gated_channels: The number of channels to use in the gated layers.
            head_channels: The number of channels to use in the 1x1 convolution blocks
                in the head after all the gated channels.
            sample_fn: See the base class.
        """
        super().__init__(*args, **kwargs)
        self._input = GatedPixelCNNLayer(
            in_channels=in_channels,
            out_channels=gated_channels,
            kernel_size=7,
            mask_center=True,
        )
        self._gated_layers = nn.ModuleList(
            [
                GatedPixelCNNLayer(
                    in_channels=gated_channels,
                    out_channels=gated_channels,
                    kernel_size=3,
                    mask_center=False,
                )
                for _ in range(n_gated)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=gated_channels, out_channels=head_channels, kernel_size=1
            ),
            nn.ReLU(),
            LaplaceConv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        vstack, hstack, skip_connections = self._input(x, x)
        for gated_layer in self._gated_layers:
            vstack, hstack, skip = gated_layer(vstack, hstack)
            skip_connections += skip
        return self._head(skip_connections)

