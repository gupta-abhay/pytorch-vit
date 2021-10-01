import torch.nn as nn

from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer


__all__ = ['ViT_B16', 'ViT_B32', 'ViT_L16', 'ViT_L32', 'ViT_H14']


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        # transformer parameters
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_revised_ffn=False,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        # embedding parameters
        use_conv_stem=True,
        use_conv_patch=False,
        use_linear_patch=False,
        use_conv_stem_original=True,
        use_stem_scaled_relu=False,
        hidden_dims=None,
        # output parameters
        cls_head=False,
        num_classes=1000,
        representation_size=None,
    ):
        super(VisionTransformer, self).__init__()

        # embedding parameters
        self.embedding_layer = EmbeddingStem(
            image_size=image_size,
            patch_size=patch_size,
            channels=in_channels,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            conv_patch=use_conv_patch,
            linear_patch=use_linear_patch,
            conv_stem=use_conv_stem,
            conv_stem_original=use_conv_stem_original,
            conv_stem_scaled_relu=use_stem_scaled_relu,
            position_embedding_dropout=dropout_rate,
            cls_head=cls_head,
        )

        # transformer parameters
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )
        self.post_transformer_ln = nn.LayerNorm(embedding_dim)

        # output layer params
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
            representation_size=representation_size,
            cls_head=cls_head,
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.post_transformer_ln(x)
        x = self.cls_layer(x)
        return x


def ViT_B16(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_B32(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L16(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L32(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_H14(dataset='imagenet'):
    if dataset == 'imagenet':
        img_dim = 224
        out_dim = 1000
        patch_dim = 14
    elif 'cifar' in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )
