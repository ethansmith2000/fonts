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
# import click
import argparse
import multiprocessing as mp

# Characters to render: Uppercase, Lowercase, and Digits
characters = string.ascii_uppercase + string.ascii_lowercase + string.digits

# Rendering parameters
bg_color = (255, 255, 255) # Background color (white)
text_color = (0, 0, 0)     # Text color (black)

def check_font_support(args):
    """
    Check if a font supports all characters.
    Returns a tuple of (font_path, is_supported)
    """
    font_path, chars, font_size = args
    supported = font_supports_all_chars(font_path, chars, font_size)
    if not supported:
        print(f"Skipping font: {font_path} (missing at least one glyph)")
    return (font_path, supported)

def save_image(img, output_path):
    """
    Save the image to disk.
    """
    img.save(output_path)
    return output_path

def create_image_grid(args):
    """
    Create a grid of images for a specific font folder.
    Args:
        args: tuple containing (folder_path, image_size)
    """
    folder_path, image_size = args
    if not os.path.isdir(folder_path):
        return None
        
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    image_paths = sorted(image_paths)
    
    if not image_paths:
        return None
        
    images = [Image.open(image_path) for image_path in image_paths]
    grid = Image.new('RGB', (image_size[0] * 8, image_size[1] * 8))
    for i, img in enumerate(images):
        row = i // 8
        col = i % 8
        grid.paste(img, (image_size[0] * col, image_size[1] * row))
    
    grid_path = os.path.join(folder_path, 'grid.png')
    grid.save(grid_path)
    return grid_path

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover system fonts
    print("Discovering system fonts")
    # system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    system_fonts = [os.path.join(args.font_dir, f) for f in os.listdir(args.font_dir) if f.endswith('.ttf')]
    print(f"Found {len(system_fonts)} fonts.")

    # Check font support in parallel
    print("Checking font support in parallel...")
    check_args = [(font, characters, args.fixed_font_size) for font in system_fonts]

    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(check_font_support, check_args),
            total=len(check_args),
            desc="Checking fonts"
        ))

    # Extract supported fonts
    supported_fonts = [font_path for font_path, supported in results if supported]
    print(f"Found {len(supported_fonts)} supported fonts.")

    tasks = []  # Each task: (font_path, char, font_folder)
    rendering_tasks = []
    for font_path in tqdm(supported_fonts, desc="Aggregating fonts"):
        # Derive a font name from the file name (without extension)
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        font_folder = os.path.join(args.output_dir, font_name)
        os.makedirs(font_folder, exist_ok=True)
        
        for char in characters:
            # tasks.append((font_path, char, font_folder))
            # Create full task tuple for rendering
            task = (font_path, char, font_folder, args.image_size, bg_color, text_color, args.fixed_font_size)
            rendering_tasks.append(task)

    # Use multiprocessing for CPU-intensive rendering
    print(f"Rendering {len(rendering_tasks)} characters using {args.num_workers} processes...")
    rendered_results = []

    with mp.Pool(processes=args.num_workers) as pool:
        rendered_results = list(tqdm(
            pool.imap(render_char, rendering_tasks),
            total=len(rendering_tasks),
            desc="Rendering fonts"
        ))

    # Use ThreadPoolExecutor for asynchronous image saving
    save_futures = []
    valid_results = [(img, path) for img, path in rendered_results if img is not None]
    with ThreadPoolExecutor() as executor:
        for img, output_path in tqdm(valid_results, desc="Scheduling saves"):
            future = executor.submit(save_image, img, output_path)
            save_futures.append(future)
        
        # Wait for all save operations to complete
        for future in tqdm(as_completed(save_futures), total=len(save_futures), desc="Saving images"):
            _ = future.result()
    
    print("Rendering complete!")

    # Remove folders that do not have all characters
    for font_folder in os.listdir(args.output_dir):
        folder_path = os.path.join(args.output_dir, font_folder)
        if os.path.isdir(folder_path):
            num_files = len(os.listdir(folder_path))
            if num_files < len(characters):
                shutil.rmtree(folder_path)

    # # for each font folder, create a grid of images
    # for font_folder in tqdm(os.listdir(args.output_dir), desc="Creating grids"):
    #     folder_path = os.path.join(args.output_dir, font_folder)
    #     if os.path.isdir(folder_path):
    #         image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    #         image_paths = sorted(image_paths)
    #         images = [Image.open(image_path) for image_path in image_paths]
    #         # sort 1,2,3 ... a,b,c
    #         grid = Image.new('RGB', (args.image_size[0] * 8, args.image_size[1] * 8))
    #         for i, img in enumerate(images):
    #             row = i // 8
    #             col = i % 8
    #             grid.paste(img, (args.image_size[0] * col, args.image_size[1] * row))
    #         grid.save(os.path.join(folder_path, 'grid.png'))

    # Create grids in parallel
    print("Creating grids in parallel...")
    grid_tasks = []
    for font_folder in os.listdir(args.output_dir):
        folder_path = os.path.join(args.output_dir, font_folder)
        if os.path.isdir(folder_path):
            grid_tasks.append((folder_path, args.image_size))
    
    with mp.Pool(processes=args.num_workers) as pool:
        grid_results = list(tqdm(
            pool.imap(create_image_grid, grid_tasks),
            total=len(grid_tasks),
            desc="Creating grids"
        ))
    
    print(f"Successfully created {sum(1 for path in grid_results if path)} grid images")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--font_dir", type=str, default="/home/ubuntu/fonts/newfonts/newfonts")
    args.add_argument("--output_dir", type=str, default="rendered_fonts")
    args.add_argument("--image_size", type=int, nargs=2, default=(128, 128))
    args.add_argument("--fixed_font_size", type=int, default=120)
    args.add_argument("--num_workers", type=int, default=10)
    args = args.parse_args()
    main(args)

