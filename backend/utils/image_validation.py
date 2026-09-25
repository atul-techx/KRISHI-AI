import os
from PIL import Image, ImageStat

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def is_valid_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_size(filepath):
    if not os.path.exists(filepath):
        return False
    return os.path.getsize(filepath) <= MAX_FILE_SIZE_BYTES

def _load_image(filepath):
    if CV2_AVAILABLE:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Could not read image file. It might be corrupted.")
        return img
    else:
        return Image.open(filepath).convert('RGB')

def is_blurry(img, threshold=30.0):
    if not CV2_AVAILABLE:
        return False # Bypass if cv2 is not available
    try:
        # Check if img is PIL image
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold
    except Exception:
        return False

def check_brightness(img, lower_thresh=30, upper_thresh=225):
    """
    Returns True if the image is within an acceptable brightness range.
    Reject very dark or very bright images.
    """
    if CV2_AVAILABLE and not isinstance(img, Image.Image):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
    else:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gray = img.convert('L')
        stat = ImageStat.Stat(gray)
        mean_brightness = stat.mean[0]
        
    if mean_brightness < lower_thresh:
        return False, "Very dark image"
    if mean_brightness > upper_thresh:
        return False, "Very bright image"
    return True, "OK"

def is_valid_leaf_image(img):
    """
    Colour-based pre-check for Crop Disease.
    Rejects very bright images and blue-dominant images.
    Requires a meaningful green presence.
    """
    if isinstance(img, Image.Image):
        img_rgb = np.array(img)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else np.array(Image.fromarray(img).convert('RGB'))
    img_float = img_rgb.astype(np.float32) / 255.0
    
    avg_r = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_b = np.mean(img_float[:, :, 2])
    total = avg_r + avg_g + avg_b + 1e-7

    if np.mean(img_float) > 0.88:
        return False

    if avg_b > avg_g * 1.30 and avg_b > avg_r * 1.20:
        return False

    green_ratio = avg_g / total
    if green_ratio < 0.28:
        return False

    return True

def is_valid_pest_image(img):
    """
    Colour-based pre-check for Pest Detection.
    Similar to leaf image since pests are usually on plants.
    More relaxed than crop disease as pest can be on soil/wood sometimes.
    """
    if isinstance(img, Image.Image):
        img_rgb = np.array(img)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else np.array(Image.fromarray(img).convert('RGB'))
    img_float = img_rgb.astype(np.float32) / 255.0
    
    avg_r = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_b = np.mean(img_float[:, :, 2])
    
    if np.mean(img_float) > 0.90:
        return False

    if avg_b > avg_r * 1.50 and avg_b > avg_g * 1.50:
        return False
        
    return True

def is_valid_soil_image(img):
    """
    Colour-based pre-check for Soil Analysis.
    Rejects very bright/white images, blue-dominant, or green-dominant (grass).
    """
    if isinstance(img, Image.Image):
        img_rgb = np.array(img)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else np.array(Image.fromarray(img).convert('RGB'))
    img_float = img_rgb.astype(np.float32) / 255.0
    
    avg_r = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_b = np.mean(img_float[:, :, 2])
    total = avg_r + avg_g + avg_b + 1e-7

    if np.mean(img_float) > 0.85:
        return False

    if avg_b > avg_r * 1.25 and avg_b > avg_g * 1.10:
        return False

    green_ratio = avg_g / total
    if green_ratio > 0.42 and avg_g > avg_r * 1.15:
        return False

    return True

def validate_image_for_mode(filepath, mode):
    """
    Main entry point for validation.
    Returns (True, None) if valid, or (False, "Error message") if invalid.
    """
    try:
        if not is_valid_extension(filepath):
            return False, "Invalid image extension. Only JPG, JPEG, PNG, WEBP are allowed."
            
        if not is_valid_size(filepath):
            return False, f"Image size too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
            
        img = _load_image(filepath)
        
        # Check blur
        if is_blurry(img, threshold=30.0): # Relaxed threshold to avoid false positives
            return False, "Image is too blurry. Please upload a clear image."
            
        # Check brightness
        is_bright_ok, bright_msg = check_brightness(img)
        if not is_bright_ok:
            return False, f"Invalid brightness: {bright_msg}. Please upload a clear image."
            
        # Mode specific checks
        if mode == 'disease' and not is_valid_leaf_image(img):
            return False, "Invalid image. Please upload a clear picture of a crop leaf."
        elif mode == 'pest' and not is_valid_pest_image(img):
            return False, "Invalid image. Please upload a clear picture of a pest or infected plant."
        elif mode == 'soil' and not is_valid_soil_image(img):
            return False, "Invalid image. Please upload a clear picture of soil."
            
        return True, None
    except Exception as e:
        return False, f"Error validating image: {str(e)}"
