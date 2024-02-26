import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.registry import register_model

from compressai.models.google import JointAutoregressiveHierarchicalPriors



class SIMO(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(256, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=1), #TODO
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=1), #TODO
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 1), #TODO
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1), #TODO
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), #rm TODO P4
            ResidualBlock(N, N),
            subpel_conv3x3(N, 256, 1), #TODO P4
        )

        self.g_s_p2 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1), #TODO
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), #rm TODO P4
            ResidualBlock(N, N),
            subpel_conv3x3(N, 256, 2), #rm TODO P4
        )

        self.g_s_p4 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),  # TODO
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),  # rm TODO P4
            ResidualBlock(N, N),
            subpel_conv3x3(N, 256, 1),  # rm TODO P4
        )

        self.g_s_p5 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),  # TODO
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),  # rm TODO P4
            ResidualBlock(N, N),
            subpel_conv3x3(N, 256, 1),  # rm TODO P4
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        x_hat_p2 = self.g_s_p2(y_hat)
        x_hat_p4 = self.g_s_p4(y_hat)
        x_hat_p5 = self.g_s_p5(y_hat)

        return {
            "x_hat": x_hat,
            "x_hat_p2": x_hat_p2,
            "x_hat_p4": x_hat_p4,
            "x_hat_p5": x_hat_p5,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net
    
    def compress(self, x):
        # if next(self.parameters()).device != torch.device("cpu"):
        #     warnings.warn(
        #         "Inference on GPU is not recommended for the autoregressive "
        #         "models (the entropy coder is run sequentially on CPU)."
        #     )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        # s = 4  # scaling factor between z and y
        s = 2  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # if next(self.parameters()).device != torch.device("cpu"):
        #     warnings.warn(
        #         "Inference on GPU is not recommended for the autoregressive "
        #         "models (the entropy coder is run sequentially on CPU)."
        #     )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 2  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
            dtype=torch.float64,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat_p2 = self.g_s_p2(y_hat).clamp_(0, 1)
        x_hat_p3 = self.g_s(y_hat).clamp_(0, 1)
        x_hat_p4 = self.g_s_p4(y_hat).clamp_(0, 1)
        x_hat_p5 = self.g_s_p5(y_hat).clamp_(0, 1)
        return {
                "x_hat_p2": x_hat_p2,
                "x_hat_p3": x_hat_p3,
                "x_hat_p4": x_hat_p4,
                "x_hat_p5": x_hat_p5,
        }