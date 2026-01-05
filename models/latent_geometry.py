import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GlyphGeometryVAE(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 encoder_layers: int = 2,
                 decoder_layers: int = 2):
        super().__init__()
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=encoder_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_out_dim = hidden_dim * 2
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_layers = decoder_layers
        self.decoder = nn.GRU(
            input_dim + latent_dim,
            hidden_dim,
            num_layers=decoder_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def encode(self, points, lengths):
        packed = pack_padded_sequence(
            points, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        hidden = hidden[-2:]
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, points, z, lengths):
        prev_points = torch.zeros_like(points)
        prev_points[:, 1:, :] = points[:, :-1, :]
        latent_seq = z.unsqueeze(1).expand(-1, points.size(1), -1)
        decoder_input = torch.cat([prev_points, latent_seq], dim=-1)
        packed = pack_padded_sequence(
            decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        init_hidden = self.latent_to_hidden(z).unsqueeze(0).repeat(
            self.decoder_layers, 1, 1)
        packed_outputs, _ = self.decoder(packed, init_hidden)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=points.size(1))
        recon = self.output(outputs)
        return recon

    def forward(self, points, lengths):
        mu, logvar = self.encode(points, lengths)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(points, z, lengths)
        return recon, mu, logvar

