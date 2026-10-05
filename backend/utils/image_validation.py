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

_classifier_bundle = None

def _get_leaf_classifier():
    global _classifier_bundle
    if _classifier_bundle is not None:
        return _classifier_bundle
    try:
        import torch
        import torchvision.models as models
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights).eval()
        transform = weights.transforms()
        categories = weights.meta.get('categories', [])
        _classifier_bundle = (model, transform, categories)
        return _classifier_bundle
    except Exception:
        return (None, None, None)

def is_valid_leaf_image(img):
    """
    Multi-layer validation to verify if the uploaded image contains a genuine plant leaf:
    1. Size and dimension check.
    2. Texture variance (rejects flat synthetic graphics, screenshots, or blank solid images).
    3. Organic plant tissue (chlorophyll green, chlorosis yellow, necrotic brown) in HSV.
    4. Deep learning classification check using MobileNetV3 (rejects confident non-plant objects).
    Returns (True, None) or (False, "reason").
    """
    if isinstance(img, Image.Image):
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else img_rgb
    else:
        img_bgr = img
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if CV2_AVAILABLE else img_bgr

    if CV2_AVAILABLE:
        h_img, w_img = img_bgr.shape[:2]
        if h_img < 50 or w_img < 50:
            return False, "फ़ोटो बहुत छोटी है। कृपया साफ़ और स्पष्ट फ़ोटो अपलोड करें।"

        # 1. Texture variance check (anti-blank / anti-solid synthetic graphic)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 10.0:
            return False, "अमान्य फ़ोटो: यह कोई स्पष्ट या असली फ़ोटो नहीं लग रही है (ब्लैंक या अमान्य ग्राफ़िक)।"

        # 2. Organic plant tissue coverage (green, chlorosis yellow, necrotic brown) in HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        plant_mask = (
            ((h >= 25) & (h <= 95) & (s > 22) & (v > 25)) |
            ((h >= 15) & (h < 25) & (s > 35) & (v > 40)) |
            (((h < 15) | (h > 165)) & (s > 30) & (v > 25) & (v < 210))
        )
        plant_ratio = float(np.sum(plant_mask)) / (h_img * w_img)
        if plant_ratio < 0.08:
            return False, "अमान्य फ़ोटो: इसमें किसी पौधे या पत्ती की पहचान नहीं हो सकी। कृपया केवल फसल या पौधे की पत्ती की साफ़ फ़ोटो अपलोड करें।"

    # 3. Deep learning MobileNet check if available
    try:
        model, transform, categories = _get_leaf_classifier()
        if model is not None and transform is not None and categories:
            import torch
            pil_img = Image.fromarray(img_rgb) if not isinstance(img, Image.Image) else img
            batch = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                out = model(batch).squeeze(0).softmax(0)
            top_prob, top_idx = torch.topk(out, 5)
            top_cats = [categories[i].lower() for i in top_idx]

            plant_keywords = [
                'cabbage', 'broccoli', 'cauliflower', 'zucchini', 'cucumber', 'pepper', 'corn', 'ear',
                'pot', 'vase', 'acorn', 'plant', 'tree', 'flower', 'leaf', 'rose', 'daisy', 'sunflower',
                'strawberry', 'apple', 'orange', 'lemon', 'banana', 'pineapple', 'grape', 'mushroom',
                'fungus', 'hay', 'rapeseed', 'cardoon', 'artichoke', 'head cabbage', 'buckeye', 'herb'
            ]
            has_plant_signal = any(any(k in c for k in plant_keywords) for c in top_cats)

            non_plant_objects = [
                'car', 'motorcycle', 'truck', 'bus', 'telephone', 'cellular', 'screen', 'laptop',
                'computer', 'chair', 'sofa', 'table', 'suit', 'jersey', 'bikini', 'shoe', 'dog',
                'cat', 'bird', 'horse', 'revolver', 'rifle', 'wallet', 'envelope'
            ]
            top_name = top_cats[0]
            is_definite_non_plant = any(np_k in top_name for np_k in non_plant_objects) and top_prob[0].item() > 0.40

            if is_definite_non_plant and not has_plant_signal:
                return False, f"अमान्य फ़ोटो: इसमें {categories[top_idx[0]]} जैसी वस्तु पाई गई, कोई पौधा या पत्ती नहीं। कृपया केवल फसल या पौधे की पत्ती अपलोड करें।"
    except Exception:
        pass

    return True, None

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
        if mode == 'disease':
            is_valid, leaf_msg = is_valid_leaf_image(img)
            if not is_valid:
                return False, leaf_msg or "अमान्य फ़ोटो: कृपया किसी फसल या पौधे की पत्ती की स्पष्ट फ़ोटो अपलोड करें।"
        elif mode == 'pest' and not is_valid_pest_image(img):
            return False, "अमान्य फ़ोटो: कृपया किसी कीट या संक्रमित पौधे की स्पष्ट फ़ोटो अपलोड करें।"
        elif mode == 'soil' and not is_valid_soil_image(img):
            return False, "अमान्य फ़ोटो: कृपया केवल मिट्टी (Soil) की स्पष्ट फ़ोटो अपलोड करें।"
            
        return True, None
    except Exception as e:
        return False, f"Error validating image: {str(e)}"
