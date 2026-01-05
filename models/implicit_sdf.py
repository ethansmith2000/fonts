import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 2, num_frequencies: int = 64, scale: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projections = 2 * math.pi * x @ self.B  # (N, num_frequencies)
        return torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)


class ImplicitGlyphSDF(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 5,
                 fourier_frequencies: int = 64,
                 coord_dim: int = 2):
        super().__init__()
        self.encoding = FourierFeatures(coord_dim, fourier_frequencies)
        in_dim = fourier_frequencies * 2
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        encoded = self.encoding(coords)
        return self.net(encoded)

