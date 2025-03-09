import torch
from utils import detect_boxes, font_supports_all_chars, render_char
import os
import string
import matplotlib.font_manager as fm
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image

class FontImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/home/ubuntu/fonts/rendered_fonts"):
        self.root_dir = root_dir
        self.dataset = []
        
        # Find all subdirectories in root_dir
        for font_name in os.listdir(root_dir):
            font_dir = os.path.join(root_dir, font_name)
            if os.path.isdir(font_dir):
                self.dataset.append({
                    "font_name": font_name,
                    "font_dir": font_dir
                })
                
        print(f"Found {len(self.dataset)} font examples")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        font_dir = example["font_dir"]
        
        # Load all images in the directory
        tensors = []
        img_path = os.path.join(font_dir, "grid.png")
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Transform image
        img = T.ToTensor()(img)
        # 1024->512
        img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(img)
        img = T.Normalize(mean=[0.5], std=[0.5])(img)
        tensors.append(img)
        
        return {
            "image": img,
            "font_name": example["font_name"]
        }