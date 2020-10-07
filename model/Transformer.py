from torch import nn
from Attention import SelfAttention, AxialAttention


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(PreNorm(dim, SelfAttention(dim, heads=heads))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim))),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
