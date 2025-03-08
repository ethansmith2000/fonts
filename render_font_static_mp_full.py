import os
import string
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import cv2
import numpy as np
from utils import detect_boxes, font_supports_all_chars, render_char
import shutil


# Optional: Force the 'fork' start method (use with caution on macOS)
if __name__ == '__main__':
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

# --- Configuration ---
output_base_dir = "rendered_fonts"
os.makedirs(output_base_dir, exist_ok=True)

# Characters to render: Uppercase, Lowercase, and Digits
characters = string.ascii_uppercase + string.ascii_lowercase + string.digits

# Rendering parameters
image_size = (64, 64)              # Image dimensions (width, height)
bg_color = (255, 255, 255)         # Background color (white)
text_color = (0, 0, 0)             # Text color (black)
fixed_font_size = 50               # Fixed font size chosen to roughly fill the canvas


def render_and_save_char_fixed(task):
    img, output_path = render_char(task)
    img.save(output_path)
    return output_path

def main():
    # Discover system fonts
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    print(f"Found {len(system_fonts)} fonts.")

    tasks = []
    for font_path in tqdm(system_fonts, desc="Checking and aggregatingfonts"):
        # Check if font supports all characters before creating tasks
        if not font_supports_all_chars(font_path, characters, fixed_font_size):
            print(f"Skipping font: {font_path} (missing at least one glyph)")
            continue
        
        # Derive a font name from the font file (without extension)
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        font_folder = os.path.join(output_base_dir, font_name)
        os.makedirs(font_folder, exist_ok=True)
        
        # If the font is good, queue up tasks for each character
        for char in characters:
            tasks.append((font_path, char, font_folder, image_size, bg_color, text_color, fixed_font_size))

    # Render in parallel
    num_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(render_and_save_char_fixed, task) for task in tasks]
        
        # Track progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering fonts"):
            _ = future.result()
    
    print("Rendering complete!")

    # go through all folders and delete those that do not have all characters
    for font_folder in os.listdir(output_base_dir):
        # get number of files in the folder
        num_files = len(os.listdir(os.path.join(output_base_dir, font_folder)))
        if num_files < len(characters):
            shutil.rmtree(os.path.join(output_base_dir, font_folder))

if __name__ == '__main__':
    main()
