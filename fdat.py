# fdat_arch.py
#
# A standalone, well-documented implementation of the FDAT architecture
# for image super-resolution.

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_


# --- Helper Modules ---

def drop_path(
    x: Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> Tensor:
    """
    Stochastically drops paths per sample.

    Args:
        x (Tensor): The input tensor.
        drop_prob (float): Probability of dropping a path.
        training (bool): Whether the model is in training mode.
        scale_by_keep (bool): If True, scales the output by 1 / (1 - drop_prob).

    Returns:
        Tensor: The output tensor after applying drop path.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Applies DropPath to the main path of residual blocks.

    This module stochastically zeroes out entire samples in a batch.

    Args:
        drop_prob (float): Probability of an element to be zeroed.
        scale_by_keep (bool): Whether to scale the output by `1/keep_prob`.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Upsampler(nn.Module):
    """
    A versatile upsampler module for the final stage of the SR model.

    This module supports two primary upsampling strategies:
    - 'pixelshuffle': Uses a sequence of convolutions and `PixelShuffle` layers.
    - 'transpose+conv': Uses `ConvTranspose2d` layers for upsampling.

    Args:
        upsampler_type (str): The upsampling method. One of ['pixelshuffle', 'transpose+conv'].
        scale (int): The upscaling factor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of intermediate channels for some methods.
    """

    def __init__(
        self,
        upsampler_type: str,
        scale: int,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels

        if upsampler_type == "pixelshuffle":
            if scale == 1:
                self.body = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            elif (scale & (scale - 1)) == 0 and scale != 0:  # Power of 2
                num_doublings = int(math.log2(scale))
                layers = [nn.Conv2d(in_channels, mid_channels, 3, 1, 1)]
                current_channels = mid_channels
                for _ in range(num_doublings):
                    layers.extend([
                        nn.Conv2d(current_channels, current_channels * 4, 3, 1, 1),
                        nn.PixelShuffle(2),
                    ])
                layers.append(nn.Conv2d(mid_channels, out_channels, 3, 1, 1))
                self.body = nn.Sequential(*layers)
            else:  # Direct pixelshuffle for non-power-of-2 scales
                self.body = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * (scale**2), 3, 1, 1),
                    nn.PixelShuffle(scale),
                )
        elif upsampler_type == "transpose+conv":
            # Specific implementation for scale=4 as used in FDAT presets
            if scale != 4:
                raise NotImplementedError(f"'transpose+conv' only supports scale 4, but got {scale}")
            self.body = nn.Sequential(
                nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2),
                nn.GELU(),
                nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown upsampler_type: {upsampler_type}")

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


# --- FDAT Components ---

class FastSpatialWindowAttention(nn.Module):
    """
    Efficient window-based multi-head self-attention with a relative position bias.

    Attention is computed only within non-overlapping local windows to reduce
    computational complexity from O(N^2) to O(N * window_size^2).

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the attention window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
    """
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, qkv_bias: bool = False) -> None:
        super().__init__()
        self.dim, self.ws, self.nh = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        trunc_normal_(self.bias, std=0.02)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, C = x.shape
        # Pad features to be divisible by window size
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x.view(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b)).view(B, -1, C)

        H_pad, W_pad = H + pad_b, W + pad_r
        # Partition into windows
        x = (
            x.view(B, H_pad // self.ws, self.ws, W_pad // self.ws, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.ws * self.ws, C)
        )
        # Multi-head self-attention
        qkv = self.qkv(x).view(-1, self.ws * self.ws, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale @ k.transpose(-2, -1)) + self.bias
        x = (F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(-1, self.ws * self.ws, C)
        # Merge windows
        x = (
            self.proj(x)
            .view(B, H_pad // self.ws, W_pad // self.ws, self.ws, self.ws, C)
            .permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, C)
        )
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.view(B, L, C)


class FastChannelAttention(nn.Module):
    """
    Computes self-attention across the channel dimension.

    This attention mechanism models inter-channel feature relationships,
    as opposed to spatial relationships.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
    """
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = False) -> None:
        super().__init__()
        self.nh = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Normalize for stability
        q = F.normalize(q.transpose(-2, -1), dim=-1)
        k = F.normalize(k.transpose(-2, -1), dim=-1)
        # Attention
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.temp, dim=-1)
        x = (attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        return self.proj(x)


class SimplifiedAIM(nn.Module):
    """
    A simplified Attention Interaction Module (AIM) for fusing attention-based
    and convolution-based features.

    It supports two interaction types:
    - `spatial_modulates_channel`: Spatial attention features modulate convolutional features.
    - `channel_modulates_spatial`: Channel attention features modulate convolutional features.

    Args:
        dim (int): Number of input channels.
        reduction_ratio (int): Channel reduction ratio for the channel gate.
    """
    def __init__(self, dim: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        # Spatial gate
        self.sg = nn.Sequential(nn.Conv2d(dim, 1, 1, bias=False), nn.Sigmoid())
        # Channel gate
        self.cg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, attn_feat: Tensor, conv_feat: Tensor, interaction_type: str, H: int, W: int) -> Tensor:
        B, L, C = attn_feat.shape
        if interaction_type == "spatial_modulates_channel":
            sm = self.sg(attn_feat.transpose(1, 2).view(B, C, H, W)).view(B, 1, L).transpose(1, 2)
            return attn_feat + (conv_feat * sm)
        else: # channel_modulates_spatial
            cm = self.cg(conv_feat.transpose(1, 2).view(B, C, H, W)).view(B, C, 1).transpose(1, 2)
            return (attn_feat * cm) + conv_feat


class SimplifiedFFN(nn.Module):
    """
    A simplified Feed-Forward Network with a depth-wise convolution for spatial mixing.

    Args:
        dim (int): Number of input channels.
        expansion_ratio (float): Ratio to expand the hidden dimension.
        drop (float): Dropout rate.
    """
    def __init__(self, dim: int, expansion_ratio: float = 2.0, drop: float = 0.0) -> None:
        super().__init__()
        hd = int(dim * expansion_ratio)
        self.fc1 = nn.Linear(dim, hd, False)
        self.act = nn.GELU()
        self.smix = nn.Conv2d(hd, hd, 3, 1, 1, groups=hd, bias=False)
        self.fc2 = nn.Linear(hd, dim, False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, C = x.shape
        x = self.drop(self.act(self.fc1(x)))
        # Spatial mixing with depth-wise convolution
        x_s = self.smix(x.transpose(1, 2).view(B, -1, H, W)).view(B, -1, L).transpose(1, 2)
        return self.drop(self.fc2(x_s))


class SimplifiedDATBlock(nn.Module):
    """
    The core building block of FDAT, a Simplified Dual-Attention Transformer Block.

    This block consists of two main branches:
    1. An attention branch (either spatial or channel).
    2. A parallel convolutional branch.
    The features from these branches are fused using a SimplifiedAIM module. This is
    followed by a SimplifiedFFN. Both steps use residual connections.

    Args:
        dim (int): Number of input channels.
        nh (int): Number of attention heads.
        ws (int): Window size for spatial attention.
        ffn_exp (float): Expansion ratio for the FFN.
        aim_re (int): Reduction ratio for the AIM module.
        btype (str): Block type, either 'spatial' or 'channel'.
        dp (float): Drop path rate.
        qkv_b (bool): If True, add bias to QKV projections.
    """
    def __init__(self, dim: int, nh: int, ws: int, ffn_exp: float, aim_re: int, btype: str, dp: float, qkv_b: bool = False) -> None:
        super().__init__()
        self.btype = btype
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.attn = (
            FastSpatialWindowAttention(dim, ws, nh, qkv_b)
            if btype == "spatial" else FastChannelAttention(dim, nh, qkv_b)
        )
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.GELU())
        self.inter = SimplifiedAIM(dim, aim_re)
        self.dp = DropPath(dp) if dp > 0.0 else nn.Identity()
        self.ffn = SimplifiedFFN(dim, ffn_exp)

    def _conv_fwd(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, C = x.shape
        return self.conv(x.transpose(1, 2).view(B, C, H, W)).view(B, C, L).transpose(1, 2)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        n1 = self.n1(x)
        itype = "channel_modulates_spatial" if self.btype == "spatial" else "spatial_modulates_channel"
        # Fuse attention and convolution branches
        fused = self.inter(self.attn(n1, H, W), self._conv_fwd(n1, H, W), itype, H, W)
        # First residual connection
        x = x + self.dp(fused)
        # Second residual connection with FFN
        x = x + self.dp(self.ffn(self.n2(x), H, W))
        return x


class SimplifiedResidualGroup(nn.Module):
    """
    A group of several SimplifiedDATBlocks with a final residual connection.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks in the group.
        nh (int): Number of attention heads.
        ws (int): Window size for spatial attention.
        ffn_exp (float): Expansion ratio for the FFN.
        aim_re (int): Reduction ratio for the AIM module.
        pattern (List[str]): Alternating pattern of 'spatial' and 'channel' blocks.
        dp_rates (List[float]): List of drop path rates for each block.
    """
    def __init__(self, dim: int, depth: int, nh: int, ws: int, ffn_exp: float, aim_re: int, pattern: List[str], dp_rates: List[float]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SimplifiedDATBlock(dim, nh, ws, ffn_exp, aim_re, pattern[i % len(pattern)], dp_rates[i])
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # Reshape from (B, C, H, W) to (B, L, C)
        x_seq = x.view(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            x_seq = block(x_seq, H, W)
        # Reshape back and add residual connection
        return self.conv(x_seq.transpose(1, 2).view(B, C, H, W)) + x


# --- Main FDAT Model ---

class FDAT(nn.Module):
    """
    FDAT: A Fast Dual-Attention Transformer for Image Super-Resolution.

    This model processes an image through three stages:
    1. Shallow Feature Extraction: A single convolution layer.
    2. Deep Feature Extraction: A series of SimplifiedResidualGroups.
    3. Image Reconstruction: An upsampler module combines deep and shallow features.

    Args:
        num_in_ch (int): Number of input image channels.
        num_out_ch (int): Number of output image channels.
        scale (int): Super-resolution scale factor.
        embed_dim (int): The main embedding dimension of the model.
        num_groups (int): Number of residual groups.
        depth_per_group (int): Number of DAT blocks per attention type within a group.
        num_heads (int): Number of attention heads.
        window_size (int): Window size for spatial attention.
        ffn_expansion_ratio (float): Expansion ratio for FFN hidden dimension.
        aim_reduction_ratio (int): Reduction ratio for the AIM module.
        group_block_pattern (List[str], optional): Alternating attention types.
        drop_path_rate (float): Stochastic depth drop path rate.
        mid_dim (int): Intermediate dimension for the upsampler.
        upsampler_type (str): Upsampling method ('pixelshuffle' or 'transpose+conv').
        img_range (float): The range of the input image data (e.g., 1.0 for [0,1]).
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        embed_dim: int = 120,
        num_groups: int = 4,
        depth_per_group: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
        ffn_expansion_ratio: float = 2.0,
        aim_reduction_ratio: int = 8,
        group_block_pattern: Optional[List[str]] = None,
        drop_path_rate: float = 0.1,
        mid_dim: int = 64,
        upsampler_type: str = "pixelshuffle",
        img_range: float = 1.0,
    ) -> None:
        super().__init__()
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel"]
        self.img_range = img_range
        self.upscale = scale

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1, bias=True)

        # 2. Deep Feature Extraction
        ad = depth_per_group * len(group_block_pattern) # Total depth of one group
        td = num_groups * ad # Total depth of the model
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, td)] # Drop path rates

        self.groups = nn.Sequential(*[
            SimplifiedResidualGroup(
                embed_dim, ad, num_heads, window_size, ffn_expansion_ratio,
                aim_reduction_ratio, group_block_pattern, dpr[i * ad : (i + 1) * ad]
            )
            for i in range(num_groups)
        ])

        # 3. Image Reconstruction
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = Upsampler(upsampler_type, scale, embed_dim, num_out_ch, mid_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x_shallow = self.conv_first(x)
        x_deep = self.groups(x_shallow)
        x_deep = self.conv_after(x_deep)
        x_out = self.upsampler(x_deep + x_shallow)
        return x_out


# --- FDAT Presets ---

def fdat_tiny(**kwargs) -> FDAT:
    """Constructs a 'tiny' version of the FDAT model."""
    params = {
        'embed_dim': 96, 'num_groups': 2, 'depth_per_group': 2,
        'num_heads': 3, 'ffn_expansion_ratio': 1.5, 'drop_path_rate': 0.05,
        'upsampler_type': 'pixelshuffle',
    }
    params.update(kwargs)
    return FDAT(**params)

def fdat_light(**kwargs) -> FDAT:
    """Constructs a 'light' version of the FDAT model."""
    params = {
        'embed_dim': 108, 'num_groups': 3, 'depth_per_group': 2,
        'num_heads': 4, 'ffn_expansion_ratio': 2.0, 'drop_path_rate': 0.08,
        'upsampler_type': 'pixelshuffle',
    }
    params.update(kwargs)
    return FDAT(**params)

def fdat_medium(**kwargs) -> FDAT:
    """Constructs a 'medium' version of the FDAT model."""
    params = {
        'embed_dim': 120, 'num_groups': 4, 'depth_per_group': 3,
        'num_heads': 4, 'ffn_expansion_ratio': 2.0, 'drop_path_rate': 0.1,
        'upsampler_type': 'transpose+conv',
    }
    params.update(kwargs)
    return FDAT(**params)

def fdat_large(**kwargs) -> FDAT:
    """Constructs a 'large' version of the FDAT model."""
    params = {
        'embed_dim': 180, 'num_groups': 4, 'depth_per_group': 4,
        'num_heads': 6, 'ffn_expansion_ratio': 2.0, 'drop_path_rate': 0.1,
        'upsampler_type': 'transpose+conv',
    }
    params.update(kwargs)
    return FDAT(**params)

def fdat_xl(**kwargs) -> FDAT:
    """Constructs an 'extra-large' version of the FDAT model."""
    params = {
        'embed_dim': 180, 'num_groups': 6, 'depth_per_group': 6,
        'num_heads': 6, 'ffn_expansion_ratio': 2.0, 'drop_path_rate': 0.1,
        'upsampler_type': 'transpose+conv',
    }
    params.update(kwargs)
    return FDAT(**params)
