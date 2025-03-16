from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen
import matplotlib.pyplot as plt
import numpy as np
import transformers

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from PIL import Image

# def check_if_any_coords_none(command_list):
#     for cmd in command_list:
#         if cmd[0] != "closePath":
#             for coord in cmd[1]:
#                 if coord is None:
#                     return True
#     return False


def extract_glyph_commands(font, char, cmap=None):  
    glyph_set = font.getGlyphSet()
    if cmap is None:
        cmap = font.getBestCmap()
    glyph_name = cmap.get(ord(char))

    # Create a decomposing recording pen
    pen = DecomposingRecordingPen(glyph_set)
    
    # Draw the glyph to the pen
    glyph_set[glyph_name].draw(pen)
    
    # Get the recorded commands
    commands = pen.value
    return commands

def check_font_have_char(font, char, cmap=None):
    char_code = ord(char)
    glyph_set = font.getGlyphSet()
    if cmap is None:
        cmap = font.getBestCmap()
    if cmap is None:
        return False
    glyph_name = cmap.get(char_code)
    return glyph_name and glyph_name in glyph_set


def is_valid_font(font):
    """
    Check if a font file is valid and has all required attributes.
    Returns (bool, str) tuple of (is_valid, error_message)
    """
    try:
        # Check if font has a character map
        if not hasattr(font, 'getBestCmap') or font.getBestCmap() is None:
            return False, "Font missing character map"

        # Check for valid advance widths (avoid huge/negative values)
        # Typical advance widths are usually between 0-3000 units
        MAX_ADVANCE_WIDTH = 5000
        for glyph_name in font.getGlyphNames():
            if hasattr(font, 'getGlyphSet'):
                glyph_set = font.getGlyphSet()
                if glyph_name in glyph_set:
                    glyph = glyph_set[glyph_name]
                    if hasattr(glyph, 'width') and glyph.width > MAX_ADVANCE_WIDTH:
                        return False, f"Invalid advance width in glyph '{glyph_name}'"

        return True, "Font is valid"

    except Exception as e:
        return False, f"Font validation error: {str(e)}"


def open_font(font_path):
    font = TTFont(font_path)   
    return font

# def process_all_glyphs(font, chars_to_process = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?"): 
#     cmap = font.getBestCmap()
#     results = {}
    
#     for char in chars_to_process:
#         if check_font_have_char(font, char, cmap):
#             results[char] = extract_glyph_commands(font, char, cmap)
#         else:
#             # print(f"Glyph '{char}' not found in font")
#             return None

#     font.close()
#     return results


def process_all_glyphs(font, chars_to_process = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?"): 
    try:
        cmap = font.getBestCmap()
        results = {}
        
        for char in chars_to_process:
            results[char] = extract_glyph_commands(font, char, cmap)
        font.close()
        return results
    except Exception as e:
        # print(f"Error processing glyphs: {e}")
        font.close()
        return None


def visualize_commands(commands, title=None, show=True):
    """
    Visualize a glyph from a list of drawing commands.
    
    Parameters:
    - commands: List of font path commands [(cmd_type, points), ...]
    - title: Optional title for the plot
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initialize points to track for bounding box
    all_points = []
    
    # Track active contours for closePath handling
    current_contour_start = None
    prev_x, prev_y = None, None
    
    # Process each command
    for cmd in commands:
        cmd_name = cmd[0]
        points = cmd[1]  # Get points array
        
        if cmd_name == 'moveTo':
            x, y = points[0]
            plt.plot(x, y, 'go', markersize=8)  # Green dot for move commands
            current_contour_start = (x, y)  # Mark the start of a new contour
            prev_x, prev_y = x, y
            all_points.append((x, y))
        
        elif cmd_name == 'lineTo':
            x, y = points[0]
            if prev_x is not None and prev_y is not None:
                plt.plot([prev_x, x], [prev_y, y], 'b-', linewidth=2)  # Blue line
            prev_x, prev_y = x, y
            all_points.append((x, y))
        
        elif cmd_name == 'curveTo':
            # Cubic Bézier curve
            if len(points) == 3 and prev_x is not None and prev_y is not None:  # Ensure correct number of points
                x1, y1 = points[0]  # First control point
                x2, y2 = points[1]  # Second control point
                x3, y3 = points[2]  # End point
                
                # Plot control points and their connections
                plt.plot([prev_x, x1], [prev_y, y1], 'r--', alpha=0.5)  # Connection to first control point
                plt.plot([x3, x2], [y3, y2], 'r--', alpha=0.5)  # Connection to second control point
                plt.plot(x1, y1, 'rx', markersize=5)  # First control point
                plt.plot(x2, y2, 'rx', markersize=5)  # Second control point
                
                # Plot the actual curve using Bézier formula
                t = np.linspace(0, 1, 100)
                curve_x = (1-t)**3 * prev_x + 3*(1-t)**2 * t * x1 + 3*(1-t) * t**2 * x2 + t**3 * x3
                curve_y = (1-t)**3 * prev_y + 3*(1-t)**2 * t * y1 + 3*(1-t) * t**2 * y2 + t**3 * y3
                plt.plot(curve_x, curve_y, 'b-', linewidth=2)
                
                prev_x, prev_y = x3, y3
                all_points.extend([(x1, y1), (x2, y2), (x3, y3)])
        
        elif cmd_name == 'qCurveTo':
            # Quadratic Bézier curve
            if len(points) == 2 and prev_x is not None and prev_y is not None:  # One control point and end point
                x1, y1 = points[0]  # Control point
                x2, y2 = points[1]  # End point
                
                # Plot control point and connections
                plt.plot([prev_x, x1], [prev_y, y1], 'r--', alpha=0.5)
                plt.plot([x2, x1], [y2, y1], 'r--', alpha=0.5)
                plt.plot(x1, y1, 'rx', markersize=5)
                
                # Plot the actual curve
                t = np.linspace(0, 1, 100)
                curve_x = (1-t)**2 * prev_x + 2*(1-t) * t * x1 + t**2 * x2
                curve_y = (1-t)**2 * prev_y + 2*(1-t) * t * y1 + t**2 * y2
                plt.plot(curve_x, curve_y, 'b-', linewidth=2)
                
                prev_x, prev_y = x2, y2
                all_points.extend([(x1, y1), (x2, y2)])
            elif len(points) > 2:
                # Handle multiple control points - TrueType format
                # For multiple control points, we create a sequence of
                # quadratic Bézier curves with implied points
                
                # Plot control points
                for x, y in points[:-1]:
                    plt.plot(x, y, 'rx', markersize=5)
                
                # The last point is the final end point
                end_point = points[-1]
                control_points = points[:-1]
                
                start_x, start_y = prev_x, prev_y
                
                # Draw a sequence of quadratic Bézier curves
                for i in range(len(control_points)):
                    # Get control point
                    cx, cy = control_points[i]
                    
                    # For all but the last control point, the end point is the midpoint
                    # between this control point and the next control point
                    if i < len(control_points) - 1:
                        next_cx, next_cy = control_points[i + 1]
                        end_x, end_y = (cx + next_cx) / 2, (cy + next_cy) / 2
                    else:
                        # For the last control point, use the final end point
                        end_x, end_y = end_point
                    
                    # Plot control lines
                    plt.plot([start_x, cx], [start_y, cy], 'r--', alpha=0.5)
                    plt.plot([end_x, cx], [end_y, cy], 'r--', alpha=0.5)
                    
                    # Plot the quadratic Bézier curve
                    t = np.linspace(0, 1, 100)
                    curve_x = (1-t)**2 * start_x + 2*(1-t) * t * cx + t**2 * end_x
                    curve_y = (1-t)**2 * start_y + 2*(1-t) * t * cy + t**2 * end_y
                    plt.plot(curve_x, curve_y, 'b-', linewidth=2)
                    
                    # The end point of this segment becomes the start point of the next
                    start_x, start_y = end_x, end_y
                
                prev_x, prev_y = end_point
                all_points.extend([p for p in control_points] + [end_point])

        elif cmd_name == 'closePath':
            # Connect the last point to the starting point of the contour
            if prev_x is not None and prev_y is not None and current_contour_start is not None:
                plt.plot([prev_x, current_contour_start[0]], [prev_y, current_contour_start[1]], 'b-', linewidth=2)
                current_contour_start = None  # Reset for the next contour
    
    # Calculate and set appropriate bounds for the plot
    if all_points:
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add some padding
        padding = max((x_max - x_min), (y_max - y_min)) * 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add title and grid
    if title:
        plt.title(title)
    else:
        plt.title("Glyph Visualization")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if show:
        plt.show()
        return None
    else:
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), 
                             fig.canvas.tostring_rgb())
        plt.close(fig)  # Close the figure to free memory
        return img

# Example usage
# Assuming 'results' is the dictionary with character commands from process_all_glyphs
# visualize_commands(results['A'], "Letter A")

def visualize_glyph(font, char):
    # Get glyph name from character code
    cmap = font.getBestCmap()
    glyph_name = cmap.get(ord(char))
    
    if not check_font_have_char(font, char, cmap):
        print(f"Character '{char}' not found in font")
        return
    
    glyph_set = font.getGlyphSet()
    
    # Create a decomposing recording pen
    pen = DecomposingRecordingPen(glyph_set)
    
    # Draw the glyph to the pen
    glyph_set[glyph_name].draw(pen)
    
    # Get the recorded commands
    commands = pen.value
    
    visualize_commands(commands, char)


def analyze_glyph_geometry(commands):
    """Analyze the geometry of a glyph from its drawing commands."""
    points = []
    contours = []
    current_contour = []
    
    for cmd in commands:
        cmd_name = cmd[0]
        # cmd_points = cmd[1:]
        cmd_points = cmd[1]
        
        if cmd_name == 'moveTo':
            # Start a new contour
            if current_contour:
                contours.append(current_contour)
                current_contour = []
            current_contour.append(cmd_points[0])
            points.append(cmd_points[0])
        
        elif cmd_name == 'lineTo':
            current_contour.append(cmd_points[0])
            points.append(cmd_points[0])
        
        elif cmd_name == 'qCurveTo':
            # Quadratic Bézier curve
            current_contour.append(cmd_points[-1])  # End point
            points.extend(cmd_points)  # All points including control points
        
        elif cmd_name == 'curveTo':
            # Cubic Bézier curve
            current_contour.append(cmd_points[-1])  # End point
            points.extend(cmd_points)  # All points including control points
        
        elif cmd_name == 'closePath':
            if current_contour:
                contours.append(current_contour)
                current_contour = []
    
    # Add any remaining contour
    if current_contour:
        contours.append(current_contour)
    
    # Calculate bounding box
    if points:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    else:
        bbox = (0, 0, 0, 0)
    
    return {
        'num_contours': len(contours),
        'num_points': len(points),
        'bbox': bbox,
        'width': bbox[2] - bbox[0],
        'height': bbox[3] - bbox[1],
        'contours': contours
    }



def get_tokenizer(pretrained_path=None):
    # Create a vocabulary dictionary
    vocab = {}
    for i in range(257):  # 0-256
        vocab[str(i)] = i

    # Add special tokens
    vocab["[UNK]"] = len(vocab)
    vocab["[BOS]"] = len(vocab)
    vocab["[EOS]"] = len(vocab)
    vocab["[PAD]"] = len(vocab) 
    vocab["[EC]"] = len(vocab) # end condition
    vocab["moveTo"] = len(vocab)
    vocab["lineTo"] = len(vocab)
    vocab["qCurveTo"] = len(vocab)
    vocab["curveTo"] = len(vocab)
    vocab["closePath"] = len(vocab)
    vocab["[SEP]"] = len(vocab) # seperate paths
    for i in range(1, 10):
        vocab["NUM"+str(i)] = len(vocab)
    for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?":
        vocab[char] = len(vocab)
    for tok in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]:
        vocab[tok] = len(vocab)
    # Create a WordLevel tokenizer with this vocabulary
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # Split on whitespace before tokenization

    # Create a transformers-compatible tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="[EOS]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )

    # Test the tokenizer
    test_text = "NUM2 123 193 256 0 , a A [PAD] ?"
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    return tokenizer


def load_transformer(pretrained_path="gpt2", pos_emb_len=2048):
    condition_mlp = nn.Sequential(
        nn.Linear(64, 128, bias=True),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 64, bias=True),
    )

    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(pretrained_path)
    model.transformer.wte = nn.Embedding(len(tokenizer), model.transformer.wte.embedding_dim)
    model.lm_head = nn.Linear(model.transformer.wte.embedding_dim, len(tokenizer))
    model.register_module("condition_mlp", condition_mlp)

    # interpolate pos embs
    pos_emb = model.transformer.wpe.weight.data.clone() # len, dim
    pos_emb = pos_emb[:pos_emb_len, :].permute(1, 0)[None, :, :] # 1, dim, len
    pos_emb = nn.functional.interpolate(pos_emb, size=pos_emb_len, mode='linear')
    pos_emb = pos_emb.permute(0, 2, 1).squeeze(0) # len, dim
    # model.transformer.wpe.weight = nn.Parameter(pos_emb)
    model.transformer.wpe = nn.Embedding(pos_emb_len, model.transformer.wpe.embedding_dim)
    # model.transformer.wpe.weight = nn.Parameter(pos_emb)

    model.config.vocab_size = len(tokenizer)
    # model.config.attn_implementation = "sdpa"

    return tokenizer, model


def path_to_string(cmd_list):
    string = ""
    for cmd in cmd_list:
        string += cmd[0] + " "
        # if "curveto" in cmd[0].lower():
        #     # add the number of points to prefix
        #     string += "NUM"+str(len(cmd[1])) + " "
        if not cmd[0] == "closePath":
            for coords in cmd[1]:
                if coords is not None:
                    string += str(coords[0]) + " " + str(coords[1]) + " "
    return string

    

def string_to_path(path_string):
    """
    Convert a string representation of path commands back to the command list format.
    
    Parameters:
    - path_string: String representation of path commands
    
    Returns:
    - List of commands in the format [(cmd_type, points), ...]
    """
    tokens = path_string.strip().split()
    cmd_list = []
    i = 0
    
    while i < len(tokens):
        cmd_type = tokens[i]
        i += 1
        
        if cmd_type == "closePath":
            cmd_list.append((cmd_type, []))
        else:
            # if "curveto" in cmd_type.lower():
            #     i += 1 # skip the number of points

            # count num tokens until next command OR end of string
            num_tokens = 0
            while i + num_tokens < len(tokens) and tokens[i + num_tokens] != "moveTo" and tokens[i + num_tokens] != "lineTo" and tokens[i + num_tokens] != "qCurveTo" and tokens[i + num_tokens] != "curveTo" and tokens[i + num_tokens] != "closePath":
                num_tokens += 1

            all_coords = []
            for j in range(num_tokens//2):
                coord_pair = []
                for k in range(2):
                    coord_pair.append(int(tokens[i+j*2+k]))
                all_coords.append(tuple(coord_pair))
            cmd_list.append((cmd_type, all_coords))
            i += num_tokens
    
    return cmd_list


def normalize_commands(commands):
    """
    Goes through a command list and rescales min and max values to be 0 and 256, rounding values to the nearest integer.
    """
    if not commands:
        return []
    
    # Find min and max values
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for cmd_type, points in commands:
        if cmd_type != "closePath":  # closePath has no points
            for coords in points:
                if coords is not None:
                    min_x = min(min_x, coords[0])
                    min_y = min(min_y, coords[1])
                    max_x = max(max_x, coords[0])
                    max_y = max(max_y, coords[1])
    
    # Calculate scaling factors
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Handle edge cases where there's no range
    x_scale = 256 / x_range if x_range > 0 else 1
    y_scale = 256 / y_range if y_range > 0 else 1
    
    # Create new normalized commands
    normalized_commands = []
    for cmd_type, points in commands:
        if cmd_type == "closePath":
            normalized_commands.append((cmd_type, []))
        else:
            normalized_points = []
            for coords in points:
                if coords is not None:
                    # Scale and shift to 0-256 range, then round to nearest integer
                    new_x = round((coords[0] - min_x) * x_scale)
                    new_y = round((coords[1] - min_y) * y_scale)
                    normalized_points.append((new_x, new_y))
            normalized_commands.append((cmd_type, normalized_points))
    
    return normalized_commands


