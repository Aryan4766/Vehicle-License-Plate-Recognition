import cv2
import numpy as np
from collections import Counter

# Define basic color mapping
COLOR_NAMES = {
    'Red': ([0, 0, 100], [80, 80, 255]),
    'Green': ([0, 100, 0], [80, 255, 80]),
    'Blue': ([100, 0, 0], [255, 80, 80]),
    'Yellow': ([0, 100, 100], [80, 255, 255]),
    'White': ([200, 200, 200], [255, 255, 255]),
    'Black': ([0, 0, 0], [50, 50, 50]),
    'Orange': ([0, 50, 200], [100, 150, 255]),
    'Purple': ([50, 0, 50], [150, 80, 150]),
    'Gray': ([100, 100, 100], [200, 200, 200]),
}

def get_dominant_color(image):
    if image is None or image.size == 0:
        return "Unknown"
    
    image = cv2.resize(image, (50, 50))  # Resize for faster processing
    pixels = image.reshape((-1, 3))
    pixels = [tuple(p) for p in pixels]

    common = Counter(pixels).most_common(1)[0][0]
    b, g, r = common

    for color_name, (lower, upper) in COLOR_NAMES.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        if (lower_np <= [b, g, r]).all() and ([b, g, r] <= upper_np).all():
            return color_name

    return "Unknown"
