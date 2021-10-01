import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from utils import trunc_normal_


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class EmbeddingStem(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        channels=3,
        embedding_dim=768,
        hidden_dims=None,
        conv_patch=False,
        linear_patch=False,
        conv_stem=True,
        conv_stem_original=True,
        conv_stem_scaled_relu=False,
        position_embedding_dropout=None,
        cls_head=True,
    ):
        super(EmbeddingStem, self).__init__()

        assert (
            sum([conv_patch, conv_stem, linear_patch]) == 1
        ), "Only one of three modes should be active"

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        assert not (
            conv_stem and cls_head
        ), "Cannot use [CLS] token approach with full conv stems for ViT"

        if linear_patch or conv_patch:
            self.grid_size = (
                image_height // patch_height,
                image_width // patch_width,
            )
            num_patches = self.grid_size[0] * self.grid_size[1]

            if cls_head:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
                num_patches += 1

            # positional embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embedding_dim)
            )
            self.pos_drop = nn.Dropout(p=position_embedding_dropout)

        if conv_patch:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    channels,
                    embedding_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
            )
        elif linear_patch:
            patch_dim = channels * patch_height * patch_width
            self.projection = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_height,
                    p2=patch_width,
                ),
                nn.Linear(patch_dim, embedding_dim),
            )
        elif conv_stem:
            assert (
                conv_stem_scaled_relu ^ conv_stem_original
            ), "Can use either the original or the scaled relu stem"

            if not isinstance(hidden_dims, list):
                raise ValueError("Cannot create stem without list of sizes")

            if conv_stem_original:
                """
                Conv stem from https://arxiv.org/pdf/2106.14881.pdf
                """

                hidden_dims.insert(0, channels)
                modules = []
                for i, (in_ch, out_ch) in enumerate(
                    zip(hidden_dims[:-1], hidden_dims[1:])
                ):
                    modules.append(
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=3,
                            stride=2 if in_ch != out_ch else 1,
                            padding=1,
                            bias=False,
                        ),
                    )
                    modules.append(nn.BatchNorm2d(out_ch),)
                    modules.append(nn.ReLU(inplace=True))

                modules.append(
                    nn.Conv2d(
                        hidden_dims[-1], embedding_dim, kernel_size=1, stride=1,
                    ),
                )
                self.projection = nn.Sequential(*modules)

            elif conv_stem_scaled_relu:
                """
                Conv stem from https://arxiv.org/pdf/2109.03810.pdf
                """
                assert (
                    len(hidden_dims) == 1
                ), "Only one value for hidden_dim is allowed"
                mid_ch = hidden_dims[0]

                # fmt: off
                self.projection = nn.Sequential(
                    nn.Conv2d(
                        channels, mid_ch,
                        kernel_size=7, stride=2, padding=3, bias=False,
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mid_ch, mid_ch,
                        kernel_size=3, stride=1, padding=1, bias=False,
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mid_ch, mid_ch,
                        kernel_size=3, stride=1, padding=1, bias=False,
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mid_ch, embedding_dim,
                        kernel_size=patch_size // 2, stride=patch_size // 2,
                    ),
                )
                # fmt: on

            else:
                raise ValueError("Undefined convolutional stem type defined")

        self.conv_stem = conv_stem
        self.conv_patch = conv_patch
        self.linear_patch = linear_patch
        self.cls_head = cls_head

        self._init_weights()

    def _init_weights(self):
        if not self.conv_stem:
            trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if self.conv_stem:
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)
            return x

        # paths for cls_token / position embedding
        elif self.linear_patch:
            x = self.projection(x)
        elif self.conv_patch:
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)

        if self.cls_head:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        return self.pos_drop(x + self.pos_embed)
