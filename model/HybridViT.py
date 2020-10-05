import torch
import torch.nn as nn
import torch.nn.functional as F
from bit import ResNetV2
from positionEncoding import PositionalEncoding


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
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.include_conv5 = include_conv5
        self.backbone_model, self.flatten_dim = self.configure_backbone(
            backbone, out_dim
        )

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.position_encoding = PositionalEncoding(
            embedding_dim, dropout_rate=0.1
        )

        self.decoder_dim = int(img_dim / 16.0) ** 2
        if self.include_conv5:
            self.decoder_dim = int(img_dim / 32.0) ** 2

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(embedding_dim * self.decoder_dim, out_dim)

    def forward(self, x):
        # apply bit backbone
        x = self.backbone_model(x, include_conv5=self.include_conv5)
        x = x.view(x.size(0), -1, self.flatten_dim)

        x = F.relu(self.linear_encoding(x))
        x = self.position_encoding(x)

        # apply transformer
        x = self.encoder(x)
        x = x.view(-1, self.embedding_dim * self.decoder_dim)
        x = self.decoder(x)
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
                "Invalid Configuration of models -- exepect 50x1, 50x3, 101x1, 101x3"
            )
        elif model_name == 'r152' and width_factor in [1, 3]:
            return ValueError(
                "Invalid Configuration of models -- exepect 152x2, 152x4"
            )

        block_units_dict = {
            'r50': [3, 4, 6, 3],
            'r101': [3, 4, 23, 3],
            'r152': [3, 8, 36, 3],
        }
        block_units = block_units_dict.get(model_name, [3, 4, 6, 3])
        model = ResNetV2(block_units, width_factor, head_size=out_dim)

        if self.num_channels == 3:
            flatten_dim = 1024 * width_factor
        if self.include_conv5:
            flatten_dim *= 2

        return model, flatten_dim


if __name__ == '__main__':
    model = HybridVisionTransformer(224, 1000, 3, 768, 12, 12, 3072)
    x = torch.randn(8, 3, 224, 224)
    model(x)
