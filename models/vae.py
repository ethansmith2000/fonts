
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basics import ResBlock

class VAE(nn.Module):
    def __init__(self, latent_dim=128, norm_type="group", norm_groups=32):
        super(VAE, self).__init__()
        
        if norm_type == "batch":
            norm_fn = lambda channels: nn.BatchNorm2d(channels)
        elif norm_type == "group":
            def get_group_norm(channels):
                if channels % norm_groups != 0 and channels > norm_groups:
                    return nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
                else:
                    return nn.GroupNorm(num_groups=channels, num_channels=channels)
            norm_fn = get_group_norm
        
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_fn(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Progressive downsampling with increasing channels
        self.enc_block1 = ResBlock(32, 64, downsample=True)     # 128x128 -> 64x64
        self.enc_block2 = ResBlock(64, 128, downsample=True)    # 64x64 -> 32x32
        self.enc_block3 = ResBlock(128, 256, downsample=True)   # 32x32 -> 16x16
        self.enc_block4 = ResBlock(256, 512, downsample=True)   # 16x16 -> 8x8
        self.enc_block5 = ResBlock(512, 768, downsample=True) # 8x8 -> 4x4 
        self.enc_block6 = ResBlock(768, 1024, downsample=True) # 4x4 -> 2x2

        
        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(1024 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(1024 * 2 * 2, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 1024 * 2 * 2)
        
        # Progressive upsampling with decreasing channels
        self.dec_block1 = ResBlock(1024, 768, upsample=False)  # 2x2 -> 4x4 (maintain size but decrease channels)
        self.dec_block2 = ResBlock(768, 512, upsample=True)    # 4x4 -> 8x8
        self.dec_block3 = ResBlock(512, 384, upsample=True)    # 8x8 -> 16x16
        self.dec_block4 = ResBlock(384, 256, upsample=True)    # 16x16 -> 32x32
        self.dec_block5 = ResBlock(256, 128, upsample=True)     # 32x32 -> 64x64
        self.dec_block6 = ResBlock(128, 64, upsample=True)      # 64x64 -> 128x128
        self.dec_block7 = ResBlock(64, 32, upsample=True)      # 128x128 -> 256x256
        
        # Final layers to get to 512x512
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.final_bn = norm_fn(16)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # print params
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")
        
    def encode(self, x):
        # Initial layers
        x = self.initial_conv(x)  # 512x512 -> 256x256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # 256x256 -> 128x128
        
        # ResBlocks
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = self.enc_block4(x)
        x = self.enc_block5(x)
        x = self.enc_block6(x)
        
        x = self.flatten(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 2, 2)
        
        # ResBlocks
        x = self.dec_block1(x)
        x = self.dec_block2(x)
        x = self.dec_block3(x)
        x = self.dec_block4(x)
        x = self.dec_block5(x)  # 128x128
        x = self.dec_block6(x)
        x = self.dec_block7(x)
        
        # Final upsampling and convolution
        x = self.upsample(x)    # 128x128 -> 256x256
        x = self.final_conv1(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        x = self.upsample(x)    # 256x256 -> 512x512
        x = self.final_conv2(x)
        x = self.tanh(x)
        
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
