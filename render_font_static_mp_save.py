import os
import string
import shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import numpy as np
from utils import detect_boxes, font_supports_all_chars, render_char

# --- Configuration ---
output_base_dir = "rendered_fonts"
os.makedirs(output_base_dir, exist_ok=True)

# Characters to render: Uppercase, Lowercase, and Digits
characters = string.ascii_uppercase + string.ascii_lowercase + string.digits

# Rendering parameters
image_size = (128, 128)      # Image dimensions (width, height)
bg_color = (255, 255, 255) # Background color (white)
text_color = (0, 0, 0)     # Text color (black)
fixed_font_size = 120       # Fixed font size chosen to roughly fill the canvas

def save_image(img, output_path):
    """
    Save the image to disk.
    """
    img.save(output_path)
    return output_path

def main():
    # Discover system fonts
    print("Discovering system fonts")
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    print(f"Found {len(system_fonts)} fonts.")

    tasks = []  # Each task: (font_path, char, font_folder)
    for font_path in tqdm(system_fonts, desc="Checking and aggregating fonts"):
        if not font_supports_all_chars(font_path, characters, fixed_font_size):
            print(f"Skipping font: {font_path} (missing at least one glyph)")
            continue
        
        # Derive a font name from the file name (without extension)
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        font_folder = os.path.join(output_base_dir, font_name)
        os.makedirs(font_folder, exist_ok=True)
        
        for char in characters:
            tasks.append((font_path, char, font_folder))

    # Use ThreadPoolExecutor for asynchronous image saving
    save_futures = []
    with ThreadPoolExecutor() as executor:
        # Process each task sequentially for rendering,
        # and schedule saving asynchronously.
        for font_path, char, font_folder in tqdm(tasks, desc="Rendering fonts"):
            task = (font_path, char, font_folder, image_size, bg_color, text_color, fixed_font_size)
            img, output_path = render_char(task)
            if img is not None:
                future = executor.submit(save_image, img, output_path)
                save_futures.append(future)
        
        # Wait for all save operations to complete
        for future in tqdm(as_completed(save_futures), total=len(save_futures), desc="Saving images"):
            _ = future.result()
    
    print("Rendering complete!")

    # Remove folders that do not have all characters
    for font_folder in os.listdir(output_base_dir):
        folder_path = os.path.join(output_base_dir, font_folder)
        if os.path.isdir(folder_path):
            num_files = len(os.listdir(folder_path))
            if num_files < len(characters):
                shutil.rmtree(folder_path)

if __name__ == '__main__':
    main()
