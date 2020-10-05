import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import TransformerModel
from positionEncoding import FixedPositionalEncoding, LearnedPositionalEncoding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        out_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        positional_encoding_type="learned",
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
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
        x = (
            x.unfold(2, self.patch_dim, self.patch_dim)
            .unfold(3, self.patch_dim, self.patch_dim)
            .contiguous()
        )
        x = x.view(x.size(0), -1, self.flatten_dim)

        x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x)

        # apply transformer
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        x = F.log_softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    # imagenet example
    model = VisionTransformer(224, 16, 1000, 3, 768, 12, 12, 3072)

    x = torch.randn(8, 3, 224, 224)
    print(model(x))
