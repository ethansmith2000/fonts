import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, Image, ImageDraw
import os
import freetype
import time

class TimerWithMessage:
    def __init__(self, start_message, end_message):
        self.start_message = start_message
        self.end_message = end_message

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.start_message}...")

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        print(f"{self.end_message} took {end_time - self.start_time:.2f} seconds")

def render_glyph_with_variation(font_path, char, axis_tag="wght", steps=5):
    # Load the font face
    face = freetype.Face(font_path)
    
    # Check if the font is variable by trying to get multiple master information
    try:
        variation_info = face.get_variation_info()
        # axes = variation_info.axes
        # tag = axes[0].tag
        # instances = variation_info.instances
    except freetype.ft_errors.FT_Exception:
        print("This font does not support variable axes.")
        return


    # # Find the desired axis (for simplicity, we assume the axis exists)
    # target_axis = None
    # for axis in variation_info.axes:
    #     # axis.tag is a 4-character code, e.g., 'wght'
    #     if axis == axis_tag:
    #         target_axis = axis
    #         break
    
    # if target_axis is None:
    #     print(f"Axis {axis_tag} not found in this font.")
    #     return

    # # Create a list of values from the minimum to maximum value for the axis
    # values = np.linspace(target_axis.minimum, target_axis.maximum, num=steps)

    images = []
    for instance in variation_info.instances:
        # Set the design coordinate for the axis.
        # If the font has only one variable axis, we pass a single value in a list.
        # print(instance.coords[0])
        face.set_var_design_coords([instance.coords[0]])

        face.set_pixel_sizes(0, 128)  # or whatever pixel height you desire
        
        # Load the character
        face.load_char(char)
        bitmap = face.glyph.bitmap
        
        # Convert the bitmap buffer to a NumPy array.
        # The bitmap buffer is a flat list of pixel intensities.
        width, rows = bitmap.width, bitmap.rows
        # # Create a numpy array from the bitmap buffer
        arr = np.array(bitmap.buffer, dtype=np.uint8).reshape(rows, width)
        
        # # Create a Pillow image from the array
        img = Image.fromarray(arr, mode='L')
        images.append((instance.coords[0], img))
    
    return images
    

def render_char(task):
    """
    Render a character with a fixed font size and save the result.
    task: (font_path, char, font_folder, image_size, bg_color, text_color, fixed_font_size)
    """
    font_path, char, font_folder, image_size, bg_color, text_color, fixed_font_size = task
    
    # Load the font
    font = ImageFont.truetype(font_path, fixed_font_size)
    
    # Create a new blank image
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Get the bounding box for the character and center it
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size[0] - text_width) // 2 - bbox[0]
    y = (image_size[1] - text_height) // 2 - bbox[1]
    
    draw.text((x, y), char, fill=text_color, font=font)

    # if the image is a placeholder box, remove the font folder
    if detect_boxes(img)[0] > 0:
        return (None, None)
    
    # Create a unique filename to avoid overwriting on case-insensitive file systems.
    if char.isupper():
        filename = f"upper_{char}.png"
    elif char.islower():
        filename = f"lower_{char}.png"
    elif char.isdigit():
        filename = f"digit_{char}.png"
    else:
        filename = f"{ord(char)}.png"
    
    output_path = os.path.join(font_folder, filename)
    return img, output_path

def font_supports_all_chars(font_path, chars, font_size):
    """
    Return True if the given font supports all characters in 'chars'.
    Otherwise, return False.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return False
    
    for c in chars:
        try:
            # If the font doesn't have a glyph for c, getmask() should be empty
            mask = font.getmask(c)
            # If there's no bounding box or the bounding box is None, the glyph is missing
            if not mask.getbbox():
                return False
        except Exception as e:
            print(f"Error checking font {font_path} for character {c}: {e}")
            return False
    return True

def detect_boxes(pil_image, debug=False):
    """
    Detect rectangular boxes in an image.
    
    Parameters:
    - image_path: Path to the image file
    - debug: If True, displays intermediate processing steps
    
    Returns:
    - num_boxes: Number of rectangular boxes detected
    - boxes: List of box contours found
    """
    image = np.array(pil_image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Find edges using Canny edge detector
    edges = cv2.Canny(threshold, 50, 150)
    
    # Dilate the edges to connect any gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize list to store boxes
    boxes = []
    
    # Create a copy of the original image for visualization
    if debug:
        vis_image = image.copy()
    
    # Iterate through contours
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's a rectangle (4 points)
        if len(approx) == 4:
            # Check if it's a proper rectangle with right angles
            is_rectangle = validate_rectangle(approx)
            
            if is_rectangle:
                boxes.append(approx)
                if debug:
                    cv2.drawContours(vis_image, [approx], 0, (0, 255, 0), 2)
    
    # Display results if debug is True
    if debug:
        plt.figure(figsize=(10, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        
        plt.subplot(2, 2, 2)
        plt.imshow(threshold, cmap='gray')
        plt.title('Thresholded Image')
        
        plt.subplot(2, 2, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Boxes: {len(boxes)}')
        
        plt.tight_layout()
        plt.show()
        
    return len(boxes), boxes

def validate_rectangle(points):
    """
    Validates if the given points form a proper rectangle with right angles
    and parallel sides of similar lengths.
    
    Parameters:
    - points: Array of 4 points (vertices of a potential rectangle)
    
    Returns:
    - bool: True if it's a valid rectangle, False otherwise
    """
    # Sort points based on their position
    pts = order_points(points.reshape(4, 2))
    
    # Calculate lengths of the sides
    width_top = np.linalg.norm(pts[0] - pts[1])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    height_left = np.linalg.norm(pts[0] - pts[3])
    height_right = np.linalg.norm(pts[1] - pts[2])
    
    # Calculate aspect ratios to check if opposite sides have similar lengths
    width_ratio = min(width_top, width_bottom) / max(width_top, width_bottom)
    height_ratio = min(height_left, height_right) / max(height_left, height_right)
    
    # Check angles (using vectors and dot products)
    is_rectangular = check_angles(pts)
    
    # Define thresholds for what's considered a rectangle
    min_ratio = 0.9  # Opposite sides should be at least 90% similar in length
    
    # Return True if it passes all checks
    return is_rectangular and width_ratio > min_ratio and height_ratio > min_ratio

def order_points(pts):
    """
    Order points in a consistent way: top-left, top-right, bottom-right, bottom-left
    """
    # Initialize ordered points array
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum of coordinates
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference of coordinates
    # The bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def check_angles(pts, eps = 0.0001):
    """
    Check if the angles between connected sides are approximately 90 degrees.
    
    Parameters:
    - pts: Ordered points (top-left, top-right, bottom-right, bottom-left)
    
    Returns:
    - bool: True if all angles are approximately 90 degrees
    """
    # Define vectors for the sides
    v1 = pts[1] - pts[0]  # top edge
    v2 = pts[3] - pts[0]  # left edge
    v3 = pts[2] - pts[1]  # right edge
    v4 = pts[2] - pts[3]  # bottom edge
    
    # Normalize vectors
    v1 = v1 / (np.linalg.norm(v1) + eps)
    v2 = v2 / (np.linalg.norm(v2) + eps)
    v3 = v3 / (np.linalg.norm(v3) + eps)
    v4 = v4 / (np.linalg.norm(v4) + eps)
    
    # Calculate dot products (should be close to 0 for perpendicular vectors)
    dot1 = np.abs(np.dot(v1, v2))  # top-left angle
    dot2 = np.abs(np.dot(v1, v3))  # top-right angle
    dot3 = np.abs(np.dot(v4, v3))  # bottom-right angle
    dot4 = np.abs(np.dot(v4, v2))  # bottom-left angle
    
    # Define threshold for what's considered perpendicular
    perpendicular_threshold = 0.1  # cos(84°) ≈ 0.1
    
    # Check if all angles are approximately 90 degrees
    return (dot1 < perpendicular_threshold and
            dot2 < perpendicular_threshold and
            dot3 < perpendicular_threshold and
            dot4 < perpendicular_threshold)

