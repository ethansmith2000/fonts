import torch
from utils import detect_boxes, font_supports_all_chars, render_char
import os
import string
import matplotlib.font_manager as fm
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataloader import FontDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataloader import FontDataset


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super(ResBlock, self).__init__()
        
        stride = 2 if downsample else 1
        self.upsample = upsample
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            identity = F.interpolate(identity, scale_factor=2, mode='bilinear', align_corners=False)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = self.relu2(out)
        
        return out

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Progressive downsampling with increasing channels
        self.enc_block1 = ResBlock(32, 64, downsample=True)     # 128x128 -> 64x64
        self.enc_block2 = ResBlock(64, 128, downsample=True)    # 64x64 -> 32x32
        self.enc_block3 = ResBlock(128, 256, downsample=True)   # 32x32 -> 16x16
        self.enc_block4 = ResBlock(256, 512, downsample=True)   # 16x16 -> 8x8
        self.enc_block5 = ResBlock(512, 1024, downsample=False) # 8x8 -> 8x8 (maintain size but increase channels)
        
        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(1024 * 8 * 8, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 1024 * 8 * 8)
        
        # Progressive upsampling with decreasing channels
        self.dec_block1 = ResBlock(1024, 512, upsample=False)  # 8x8 -> 8x8 (maintain size but decrease channels)
        self.dec_block2 = ResBlock(512, 256, upsample=True)    # 8x8 -> 16x16
        self.dec_block3 = ResBlock(256, 128, upsample=True)    # 16x16 -> 32x32
        self.dec_block4 = ResBlock(128, 64, upsample=True)     # 32x32 -> 64x64
        self.dec_block5 = ResBlock(64, 32, upsample=True)      # 64x64 -> 128x128
        
        # Final layers to get to 512x512
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.final_bn = nn.BatchNorm2d(16)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
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
        x = x.view(-1, 1024, 8, 8)
        
        # ResBlocks
        x = self.dec_block1(x)
        x = self.dec_block2(x)
        x = self.dec_block3(x)
        x = self.dec_block4(x)
        x = self.dec_block5(x)  # 128x128
        
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


def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #tf32 enable, matmul precision high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    
    # Initialize dataset and dataloader
    dataset = FontDataset()
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
    
    # Initialize model and optimizer
    model = VAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], betas=args["betas"], weight_decay=args["weight_decay"])
    
    # Training loop
    for epoch in range(args["epochs"]):
        model.train()
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args['epochs']}"):
            images = batch["image"].to(device)
            
            # Forward pass
            recon_images, mu, log_var = model(images)
            
            # Reconstruction loss
            recon = F.mse_loss(recon_images, images, reduction='sum')
            
            # KL divergence loss
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon + kl * args["kl_weight"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataset)
        avg_recon = recon_loss / len(dataset)
        avg_kl = kl_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args['epochs']}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Reconstruction Loss: {avg_recon:.4f}")
        print(f"KL Loss: {avg_kl:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'vae_checkpoint_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    args = dict(
        epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=0.001,
        betas=(0.9, 0.999),
        kl_weight=1.0,
    )
    train_vae(args)
