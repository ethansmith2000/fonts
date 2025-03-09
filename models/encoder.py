import torch
import torch.nn as nn
from .basics import ResBlock

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, norm_type="group", norm_groups=32):
        super().__init__()
        
        if norm_type == "batch":
            norm_fn = lambda channels: nn.BatchNorm2d(channels)
        elif norm_type == "group":
            def get_group_norm(channels):
                if channels % norm_groups != 0 and channels > norm_groups:
                    return nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
                else:
                    return nn.GroupNorm(num_groups=channels, num_channels=channels)
            norm_fn = get_group_norm
        
        self.initial_conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True)
        self.norm1 = norm_fn(8)
        self.relu = nn.ReLU(inplace=True)
        
        # Progressive downsampling with increasing channels
        self.enc_block1 = ResBlock(8, 16, downsample=True) # 512x512 -> 256x256
        self.enc_block2 = ResBlock(16, 32, downsample=True) # 256x256 -> 128x128
        self.enc_block3 = ResBlock(32, 64, downsample=True) # 128x128 -> 64x64
        self.enc_block4 = ResBlock(64, 128, downsample=True) # 64x64 -> 32x32
        self.enc_block5 = ResBlock(128, 256, downsample=True)   # 32x32 -> 16x16
        self.enc_block6 = ResBlock(256, 512, downsample=True)   # 16x16 -> 8x8
        self.enc_block7 = ResBlock(512, 768, downsample=True) # 8x8 -> 4x4 
        self.enc_block8 = ResBlock(768, 1024, downsample=True) # 4x4 -> 2x2
        self.enc_block9 = ResBlock(1024, 1024, downsample=True) # 2x2 -> 1x1

        
        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.norm2 = norm_fn(1024)

        self.epilogue = nn.Sequential(
            nn.Linear(1024, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 128, bias=False),
        )

        self.mu_norm = nn.LayerNorm(128)
        self.var_norm = nn.LayerNorm(128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)


        # print params
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = self.enc_block4(x)
        x = self.enc_block5(x)
        x = self.enc_block6(x)
        x = self.enc_block7(x)
        x = self.enc_block8(x)
        x = self.enc_block9(x)
        x = self.norm2(x)
        x = self.flatten(x)
        x = self.epilogue(x)

        mu = self.mu_norm(x)
        mu = self.fc_mu(mu)

        log_var = self.var_norm(x)
        log_var = self.fc_var(log_var)

        sample = self.reparameterize(mu, log_var)

        return mu, log_var, sample