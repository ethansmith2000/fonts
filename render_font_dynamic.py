import os
import string
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---

# Output base directory for rendered fonts
output_base_dir = "rendered_fonts"
os.makedirs(output_base_dir, exist_ok=True)

# Characters to render: Uppercase, Lowercase, and Digits
characters = string.ascii_uppercase + string.ascii_lowercase + string.digits

# Rendering parameters
image_size = (64, 64)              # Image dimensions (width, height)
bg_color = (255, 255, 255)         # Background color (white)
text_color = (0, 0, 0)             # Text color (black)
margin = 0.9                       # We want the rendered text to fill ~90% of the canvas

# Maximum font size to search (you can adjust as needed)
MAX_FONT_SIZE = 500

# --- Helper Functions ---

def find_best_font_size(font_path, char, target_size, margin=0.9, low=1, high=MAX_FONT_SIZE):
    """
    Binary search to find the largest font size (using the given font_path) for which
    the rendered character 'char' fits within the target_size * margin.
    """
    best_size = low
    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            # If loading fails at this size, break out
            break

        # Create a dummy image for measuring
        dummy = Image.new("RGB", target_size)
        draw = ImageDraw.Draw(dummy)
        # Use textbbox for a more precise bounding box
        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width < target_size[0] * margin and height < target_size[1] * margin:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1
    return best_size

def render_and_save_char(task):
    """
    Worker function to render a single character image.
    
    Parameters:
      task: a tuple containing:
            (font_path, char, font_folder, image_size, bg_color, text_color, margin)
    Returns:
      The output path of the saved image.
    """
    font_path, char, font_folder, image_size, bg_color, text_color, margin = task

    # Determine the optimal font size for this character
    optimal_font_size = find_best_font_size(font_path, char, image_size, margin)
    try:
        font = ImageFont.truetype(font_path, optimal_font_size)
    except Exception as e:
        return f"Error loading font {font_path} at size {optimal_font_size}: {e}"

    # Create a new blank image
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Get the bounding box of the character; textbbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text by offsetting by the bbox values
    x = (image_size[0] - text_width) // 2 - bbox[0]
    y = (image_size[1] - text_height) // 2 - bbox[1]
    
    # Draw the character
    draw.text((x, y), char, fill=text_color, font=font)
    
    # Save the image
    output_path = os.path.join(font_folder, f"{char}.png")
    img.save(output_path)
    return output_path

# --- Discovering Fonts ---
system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print(f"Found {len(system_fonts)} fonts.")

# Prepare a list of tasks: for each font and for each character, we create one job.
tasks = []
for font_path in system_fonts:
    # Derive a font name from the font file (without extension)
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    font_folder = os.path.join(output_base_dir, font_name)
    os.makedirs(font_folder, exist_ok=True)
    
    for char in characters:
        tasks.append((font_path, char, font_folder, image_size, bg_color, text_color, margin))

# --- Multiprocessing: Rendering in Parallel ---
# We'll use a ProcessPoolExecutor to run tasks concurrently.
num_workers = os.cpu_count() or 4  # Use available cores
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Submit all tasks
    futures = [executor.submit(render_and_save_char, task) for task in tasks]
    
    # Use tqdm to track progress
    for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering fonts"):
        # You can collect the result if needed:
        _ = future.result()

print("Rendering complete!")
