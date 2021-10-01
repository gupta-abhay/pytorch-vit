import torch.nn as nn

from patch_embed import EmbeddingStem
from transformer import Transformer
from modules import OutputLayer


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_revised_ffn=False,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_conv_stem=True,
        use_conv_patch=False,
        use_linear_patch=False,
        use_conv_stem_original=True,
        use_stem_scaled_relu=False,
        hidden_dims=None,
        cls_head=False,
        num_classes=1000,
        representation_size=None,
    ):
        super(VisionTransformer, self).__init__()

        # embedding layer
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

        # transformer
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

        # output layer
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
