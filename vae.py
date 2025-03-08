import torch
from utils import detect_boxes, font_supports_all_chars, render_char
import os
import string
import matplotlib.font_manager as fm
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np


