import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


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
        dropout_rate=0.1,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.position_encoding = PositionalEncoding(
            embedding_dim, dropout_rate=0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(embedding_dim * self.num_patches, out_dim)

    def forward(self, x):
        x = (
            x.unfold(2, self.patch_dim, self.patch_dim)
            .unfold(3, self.patch_dim, self.patch_dim)
            .contiguous()
        )
        x = x.view(x.size(0), -1, self.flatten_dim)

        x = F.relu(self.linear_encoding(x))
        x = self.position_encoding(x)

        # apply transformer
        x = self.encoder(x)
        x = x.view(-1, self.embedding_dim * self.num_patches)
        x = self.decoder(x)
        x = F.log_softmax(x, dim=-1)

        return x
