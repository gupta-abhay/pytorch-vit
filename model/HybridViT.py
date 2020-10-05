import torch
import torch.nn as nn
import torch.nn.functional as F
from BiT import ResNetV2Model
from Transformer import TransformerModel
from PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding


class HybridVisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        out_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        backbone='r50x1',
        include_conv5=False,
        dropout_rate=0.1,
        positional_encoding_type="learned",
    ):
        super(HybridVisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.include_conv5 = include_conv5
        self.backbone_model, self.flatten_dim = self.configure_backbone(
            backbone, out_dim
        )

        self.projection_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.decoder_dim = int(img_dim / 16.0) ** 2
        if self.include_conv5:
            self.decoder_dim = int(img_dim / 32.0) ** 2

        self.decoder_dim += 1  # for the cls token

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.decoder_dim, self.embedding_dim, self.decoder_dim
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.transformer = TransformerModel(
            embedding_dim, num_layers, num_heads, hidden_dim
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.to_cls_token = nn.Identity()

    def forward(self, x):
        # apply bit backbone
        x = self.backbone_model(x, include_conv5=self.include_conv5)
        x = x.view(x.size(0), -1, self.flatten_dim)

        x = self.projection_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x)

        # apply transformer
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        x = F.log_softmax(x, dim=-1)

        return x

    def configure_backbone(self, backbone, out_dim):
        """
        Current support offered for all BiT models
        KNOWN_MODELS in https://github.com/google-research/big_transfer/blob/master/bit_pytorch/models.py

        expects model name of style 'r{depth}x{width}'
        where depth in [50, 101, 152]
        where width in [1,2,3,4]
        """
        splits = backbone.split('x')
        model_name = splits[0]
        width_factor = int(splits[1])

        if model_name in ['r50', 'r101'] and width_factor in [2, 4]:
            return ValueError(
                "Invalid Configuration of models -- expect 50x1, 50x3, 101x1, 101x3"
            )
        elif model_name == 'r152' and width_factor in [1, 3]:
            return ValueError(
                "Invalid Configuration of models -- expect 152x2, 152x4"
            )

        block_units_dict = {
            'r50': [3, 4, 6, 3],
            'r101': [3, 4, 23, 3],
            'r152': [3, 8, 36, 3],
        }
        block_units = block_units_dict.get(model_name, [3, 4, 6, 3])
        model = ResNetV2Model(block_units, width_factor, head_size=out_dim)

        if self.num_channels == 3:
            flatten_dim = 1024 * width_factor
        if self.include_conv5:
            flatten_dim *= 2

        return model, flatten_dim


if __name__ == '__main__':
    model = HybridVisionTransformer(224, 1000, 3, 768, 12, 12, 3072)
    x = torch.randn(8, 3, 224, 224)
    print(model(x))
