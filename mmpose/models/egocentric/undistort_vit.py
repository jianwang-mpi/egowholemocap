#  Copyright Jian Wang @ MPI-INF (c) 2023.

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from mmpose.models import build_posenet
from mmpose.models.backbones.base_backbone import BaseBackbone
from ..builder import BACKBONES
from ..builder import NECKS


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedMLP(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_num_horizontal, patch_num_vertical, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = patch_num_horizontal * patch_num_vertical
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        in_dims = in_chans * patch_size[0] * patch_size[1]

        self.proj = nn.Linear(in_dims, embed_dim)
        self.input_layernorm = nn.LayerNorm(in_dims)
        self.output_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        B, patch_num, C, H, W = x.shape
        assert patch_num == self.num_patches
        x = x.view(B * patch_num, C * H * W)
        x = self.input_layernorm(x)
        x = self.proj(x)
        x = self.output_layernorm(x)
        x = x.view(B, patch_num, self.embed_dim)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_num_horizontal, patch_num_vertical, patch_size=32, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = patch_num_horizontal * patch_num_vertical
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        in_dims = in_chans * patch_size[0] * patch_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio),
                              padding=0)

    def forward(self, x, **kwargs):
        B, patch_num, C, H, W = x.shape
        assert patch_num == self.num_patches
        x = x.view(B * patch_num, C, H, W)
        x = self.proj(x)
        x = x.view(B, patch_num, self.embed_dim)
        return x



class PatchEmbedWithCNN(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_num_horizontal, patch_num_vertical, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = patch_num_horizontal * patch_num_vertical
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        in_dims = in_chans * patch_size[0] * patch_size[1]

        # use resnet to extract feature
        self.proj = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.proj.fc = nn.Linear(2048, self.embed_dim)

    def forward(self, x, **kwargs):
        B, patch_num, C, H, W = x.shape
        assert patch_num == self.num_patches
        x = x.view(B * patch_num, C, H, W)
        x = self.proj(x)
        x = x.view(B, patch_num, self.embed_dim)
        return x


@BACKBONES.register_module()
class UndistortViT(BaseBackbone):
    def __init__(self,
                 img_size=224, patch_size=32, in_chans=3, num_classes=80, embed_dim=768, embed_type='linear', depth=12,
                 num_heads=12, ratio=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False, fisheye2sphere_configs=None,
                 linear_embedding=False,
                 ):
        # Protect mutable default arguments
        super(UndistortViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        self.fisheye2sphere_config = deepcopy(fisheye2sphere_configs)
        self.fisheye2sphere = build_posenet(self.fisheye2sphere_config)

        assert embed_type in ['linear', 'cnn', 'mlp']
        if embed_type == 'linear':
            self.patch_embed = PatchEmbed(self.fisheye2sphere_config['patch_num_horizontal'],
                                          self.fisheye2sphere_config['patch_num_vertical'],
                                          patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                          ratio=ratio)
        elif embed_type == 'mlp':
            self.patch_embed = PatchEmbedMLP(self.fisheye2sphere_config['patch_num_horizontal'],
                                             self.fisheye2sphere_config['patch_num_vertical'],
                                             patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbedWithCNN(self.fisheye2sphere_config['patch_num_horizontal'],
                                                 self.fisheye2sphere_config['patch_num_vertical'],
                                                 patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # super().init_weights(pretrained, patch_padding=self.patch_padding)
        super().init_weights(pretrained)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        x = self.fisheye2sphere(x)
        B, patch_num, C, H, W = x.shape
        Hp = self.fisheye2sphere_config['patch_num_horizontal']
        Wp = self.fisheye2sphere_config['patch_num_vertical']
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)
        assert x.shape[1] == patch_num and x.shape[2] == self.patch_embed.embed_dim
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

