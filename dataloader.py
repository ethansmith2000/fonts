import torch
from utils import detect_boxes, font_supports_all_chars, render_char
import os
import string
import matplotlib.font_manager as fm
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np


class FontDataset(torch.utils.data.Dataset):
    def __init__(self):
        print("Discovering system fonts")
        system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        print(f"Found {len(system_fonts)} fonts.")


        characters = string.ascii_uppercase + string.ascii_lowercase + string.digits

        # Rendering parameters
        self.image_size = (128, 128)      # Image dimensions (width, height)
        self.bg_color = (255, 255, 255) # Background color (white)
        self.text_color = (0, 0, 0)     # Text color (black)
        self.fixed_font_size = 120       # Fixed font size chosen to roughly fill the canvas

        self.dataset = []

        for font_path in tqdm(system_fonts, desc="Checking and aggregating fonts"):
            if not font_supports_all_chars(font_path, characters, fixed_font_size):
                print(f"Skipping font: {font_path} (missing at least one glyph)")
                continue
            
            # Derive a font name from the file name (without extension)
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            
            self.dataset.append({
                "font_name": font_name,
                "font_path": font_path,
            })

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        task = (example["font_path"], example["char"], example["font_folder"], self.image_size, self.bg_color, self.text_color, self.fixed_font_size)
        img, output_path = render_char(task)
        # tensor
        img = T.ToTensor()(img)
        # normalize
        img = T.Normalize(mean=[0.5], std=[0.5])(img)

        example = {
            "image": img,
            "font_name": example["font_name"],
        }

        return example