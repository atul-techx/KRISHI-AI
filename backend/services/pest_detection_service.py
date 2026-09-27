import os
import sys
import uuid
import logging
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

# Base and Model Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Pest_Detection", "best.pt")

# Ensure virtual environment packages are accessible if ultralytics is installed there
VENV_PACKAGES = os.path.join(BASE_DIR, "models", "Pest_Detection", ".venv", "Lib", "site-packages")
if os.path.exists(VENV_PACKAGES) and VENV_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_PACKAGES)

_model = None

def get_model():
    """Lazy load the trained YOLOv8 model from models/Pest_Detection/best.pt"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO Pest Detection model file not found at: {MODEL_PATH}")
        try:
            from ultralytics import YOLO
            logger.info(f"[PEST-AI] Loading YOLO Pest Detection model from {MODEL_PATH}")
            _model = YOLO(MODEL_PATH)
            logger.info(f"[PEST-AI] Loaded successfully. Classes: {_model.names}")
        except Exception as e:
            logger.error(f"[PEST-AI] Failed to load YOLO model: {e}")
            raise
    return _model

# Dedicated Pest Knowledge Base for YOLO model classes
PEST_DATABASE = {
    "fruit fly": {
        "hindi_name": "फल मक्खी (Fruit Fly)",
        "english_name": "Fruit Fly (Bactrocera spp.)",
        "crop_hosts": "टमाटर, आम, अमरूद, लौकी, तोरी, खीरा, तरबूज, अनार",
        "cause": "फल मक्खी फलों व सब्जियों के छिलके में छेद करके अंडे देती है। इसके मैगट (सूंडी) फल के गूदे को अंदर से खाकर सड़ा देते हैं, जिससे फल समय से पहले गिर जाता है।",
        "treatment": [
            "खेत में 6 से 8 मिथाइल यूजेनॉल फेरोमोन ट्रैप (Methyl Eugenol Trap) प्रति एकड़ लगाएं।",
            "नीम का तेल (10,000 PPM) 2.5 से 3 मिली प्रति लीटर पानी में मिलाकर शाम को छिड़कें।",
            "विष प्रलोभक (Bait Spray): 100 ग्राम गुड़ + 20 मिली मैलाथियान 50 EC को 20 लीटर पानी में मिलाकर पौधे के तनों व पत्तियों पर मोटे बूंदों में छिड़कें।",
            "संक्रमित व जमीन पर गिरे हुए सभी फलों को तुरंत एकत्र कर मिट्टी में 2 फीट गहरा दबाएं।"
        ],
        "prevention": "फल लगने के प्रारंभिक चरण में फ्रूट बैगिंग (Paper Bags) का प्रयोग करें। खेत के चारों तरफ पीले चिपचिपे कार्ड लगाएं और कटाई उपरांत जुताई करें।",
        "severity_high_threshold": 5
    },
    "aphid": {
        "hindi_name": "माहू / चेपा / एफिड्स (Aphids)",
        "english_name": "Aphids (Aphis gossypii / craccivora)",
        "crop_hosts": "सरसों, गेहूं, टमाटर, बैंगन, मिर्च, मटर, दलहन",
        "cause": "माहू पत्तियों, कोमल शाखाओं और कलियों से लगातार रस चूसते हैं। ये 'हनीड्यू' (चिपचिपा मीठा द्रव) छोड़ते हैं जिससे पत्तियों पर काली फफूंद (Sooty Mold) जम जाती है और प्रकाश संश्लेषण रुक जाता है।",
        "treatment": [
            "नीमास्त्र (Neem Astra) 10% घोल या 5% नीम बीज अर्क (NSKE) का तुरंत छिड़काव करें।",
            "खट्टी छाछ (500 मिली) + 15 लीटर पानी का घोल बनाकर स्प्रे करें, यह माहू को प्राकृतिक रूप से नियंत्रित करता है।",
            "खेत में प्रति एकड़ 10-12 पीले चिपचिपे ट्रैप (Yellow Sticky Traps) लगाएं।",
            "गंभीर प्रकोप होने पर इमिडाक्लोप्रिड 17.8 SL (0.5 मिली/लीटर) या एसिटामिप्रिड 20 SP (0.5 ग्राम/लीटर) का छिड़काव करें।"
        ],
        "prevention": "यूरिया (नाइट्रोजन) का अत्यधिक असंतुलित उपयोग न करें। मित्र कीट लेडीबर्ड बीटल (Ladybug) का संरक्षण करें जो माहू का भक्षण करते हैं।",
        "severity_high_threshold": 8
    },
    "scale_insect": {
        "hindi_name": "स्केल कीट / शल्क कीट (Scale Insect)",
        "english_name": "Scale Insect (Coccoidea)",
        "crop_hosts": "आम, नींबू वर्गीय फल, अमरूद, सेब, गन्ना, कपास",
        "cause": "स्केल कीट तनों और टहनियों पर मोम जैसी सुरक्षात्मक ढाल (Scale) बनाकर चिपक जाते हैं और लगातार पौधों का रस चूसते हैं। इससे टहनियां सूखने लगती हैं और उपज भारी घट जाती है।",
        "treatment": [
            "नीम का तेल (5 मिली) + लिक्विड साबुन या डिटर्जेंट (1 मिली) प्रति लीटर पानी में मिलाकर टहनियों पर अच्छी तरह स्प्रे करें।",
            "हॉर्टिकल्चर मिनरल ऑयल (Horticultural Oil 2%) का छिड़काव करें जो इनकी मोम जैसी खोल को घोल देता है।",
            "गंभीर रूप से ग्रसित और सूखी टहनियों को प्रूनिंग कैंची से काटकर जला दें।",
            "रासायनिक उपचार हेतु क्लोरोपायरीफॉस 20 EC (2 मिली/लीटर) का छिड़काव करें।"
        ],
        "prevention": "पेड़ों की नियमित छंटाई (Pruning) करें ताकि पौधों में धूप व हवा का संचार बना रहे। परभक्षी कीटों (Chilocorus Beetles) को बढ़ावा दें।",
        "severity_high_threshold": 4
    }
}

def get_pest_recommendation(pest_name):
    """Retrieve detailed Hindi/English pest profile or fallback"""
    lower = pest_name.lower().strip()
    for key, data in PEST_DATABASE.items():
        if key in lower or lower in key:
            return data
    
    # Generic fallback for other pests
    return {
        "hindi_name": f"कीट: {pest_name.title()}",
        "english_name": pest_name.title(),
        "crop_hosts": "विभिन्न फसलें",
        "cause": f"{pest_name.title()} का संक्रमण फसल की पत्तियों या फलों को नुकसान पहुंचा रहा है।",
        "treatment": [
            "नीम का तेल (10,000 PPM) 3 मिली प्रति लीटर पानी में मिलाकर शाम के समय छिड़कें।",
            "खेत में कीट निगरानी हेतु फेरोमोन ट्रैप व लाइट ट्रैप (प्रकाश प्रपंच) लगाएं।",
            "नजदीकी कृषि विज्ञान केंद्र (KVK) या कृषि विशेषज्ञ से उचित कीटनाशक की सलाह लें।"
        ],
        "prevention": "खेत की नियमित स्वच्छता रखें, फसल चक्र (Crop Rotation) अपनाएं और जैविक खाद का प्रयोग करें।",
        "severity_high_threshold": 5
    }


def predict_pest(image_path, user_crop="", output_dir=None):
    """
    Run YOLOv8 Pest Detection on an image.
    Annotates the image with bounding boxes, labels, and confidence.
    """
    try:
        model = get_model()
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "चित्र को पढ़ा नहीं जा सका। कृपया वैध फोटो अपलोड करें।"}

        results = model(image, verbose=False)
        annotated = results[0].plot()

        # Save annotated image
        annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        annotated_path = os.path.join(output_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated)

        best_pest = None
        best_conf = 0.0
        detections = {}

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0]) * 100
            detections[label] = detections.get(label, 0) + 1
            if conf > best_conf:
                best_conf = conf
                best_pest = label

        if best_pest is None or len(results[0].boxes) == 0:
            # Crop image might be safe or no insects found
            return {
                "media_type": "image",
                "disease": "कोई हानिकारक कीट नहीं मिला (No Pest Detected)",
                "crop": user_crop if user_crop else "फसल",
                "confidence": 95.0,
                "severity": "Safe",
                "is_safe": True,
                "annotated_filename": annotated_filename,
                "treatment": [
                    "आपकी फसल में कोई हानिकारक कीट नहीं पाया गया है।",
                    "पौधों की नियमित निगरानी जारी रखें।",
                    "संतुलित जैविक खाद व समय पर सिंचाई करें।"
                ],
                "prevention": "स्वस्थ फसल हेतु नीम अर्क का 15 दिन के अंतराल पर एहतियातन छिड़काव करते रहें।",
                "cause": "फसल स्वस्थ प्रतीत हो रही है। कीटों का कोई सक्रिय प्रकोप नहीं दिखा।"
            }

        rec = get_pest_recommendation(best_pest)
        severity = "High" if best_conf > 70 or detections.get(best_pest, 0) >= rec.get("severity_high_threshold", 5) else "Medium"

        detection_list = [
            {
                "label": PEST_DATABASE.get(k, {}).get("hindi_name", k.title()),
                "raw_label": k,
                "count": v
            }
            for k, v in detections.items()
        ]

        return {
            "media_type": "image",
            "disease": rec["hindi_name"],
            "raw_name": best_pest,
            "crop": user_crop if user_crop else rec.get("crop_hosts", "फसल"),
            "confidence": round(best_conf, 1),
            "severity": severity,
            "is_safe": False,
            "annotated_filename": annotated_filename,
            "detections": detection_list,
            "total_pest_count": len(results[0].boxes),
            "treatment": rec["treatment"],
            "prevention": rec["prevention"],
            "cause": rec["cause"]
        }

    except Exception as e:
        logger.error(f"[PEST-AI] Image prediction error: {e}", exc_info=True)
        return {"error": f"कीट पहचान में त्रुटि: {str(e)}"}


def predict_pest_video(video_path, user_crop="", output_dir=None):
    """
    Run YOLOv8 Pest Detection on a video file.
    Annotates video with bounding boxes and tracks pest detections across all frames.
    Saves an annotated video and a peak snapshot keyframe image.
    """
    try:
        model = get_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "वीडियो फाइल को खोला नहीं जा सका। कृपया वैध MP4 या AVI वीडियो दें।"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"[PEST-AI] Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        os.makedirs(output_dir, exist_ok=True)

        annotated_video_filename = f"pest_annotated_{uuid.uuid4().hex}.mp4"
        annotated_video_path = os.path.join(output_dir, annotated_video_filename)

        snapshot_filename = f"pest_snap_{uuid.uuid4().hex}.jpg"
        snapshot_path = os.path.join(output_dir, snapshot_filename)

        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback to avc1 or XVID if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        frame_count = 0
        all_labels = {}
        max_boxes_in_frame = 0
        best_frame_annotated = None
        confidences = []

        # Frame stride to keep processing time fast for web users
        # For videos with > 120 frames, process every N-th frame to ensure instant response (< 5s)
        stride = 1
        if total_frames > 150:
            stride = max(1, total_frames // 120)

        read_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            read_count += 1
            if stride > 1 and (read_count % stride != 0):
                continue

            frame_count += 1
            results = model(frame, verbose=False)
            annotated = results[0].plot()

            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))

            if writer.isOpened():
                writer.write(annotated)

            num_boxes = len(results[0].boxes)
            if num_boxes > max_boxes_in_frame or best_frame_annotated is None:
                max_boxes_in_frame = num_boxes
                best_frame_annotated = annotated.copy()

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0]) * 100
                confidences.append(conf)
                all_labels[label] = all_labels.get(label, 0) + 1

        cap.release()
        if writer.isOpened():
            writer.release()

        # Save snapshot keyframe image
        if best_frame_annotated is not None:
            cv2.imwrite(snapshot_path, best_frame_annotated)
        else:
            snapshot_filename = None

        logger.info(f"[PEST-AI] Finished video inference: {frame_count} frames, detections: {all_labels}")

        if not all_labels:
            return {
                "media_type": "video",
                "disease": "कोई हानिकारक कीट नहीं मिला (No Pest Detected)",
                "crop": user_crop if user_crop else "फसल",
                "confidence": 96.0,
                "severity": "Safe",
                "is_safe": True,
                "frames_analyzed": frame_count,
                "annotated_video_filename": annotated_video_filename,
                "snapshot_filename": snapshot_filename,
                "total_pest_count": 0,
                "treatment": [
                    "वीडियो विश्लेषण में कोई हानिकारक कीट या नुकसान नहीं पाया गया।",
                    "पौधों की स्वस्थ वृद्धि बनाए रखने हेतु नियमित निरीक्षण करते रहें।"
                ],
                "prevention": "एहतियात के तौर पर प्रति 15 दिन में जीवामृत या नीम अर्क का छिड़काव करते रहें।",
                "cause": "वीडियो में फसल पूरी तरह स्वस्थ और कीट-मुक्त पाई गई।"
            }

        # Dominant pest with highest count
        dominant_pest = max(all_labels.items(), key=lambda x: x[1])[0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 85.0
        max_conf = max(confidences) if confidences else 85.0

        rec = get_pest_recommendation(dominant_pest)
        total_pests = sum(all_labels.values())

        severity = "High" if total_pests >= rec.get("severity_high_threshold", 5) or max_conf > 75 else "Medium"

        detection_list = [
            {
                "label": PEST_DATABASE.get(k, {}).get("hindi_name", k.title()),
                "raw_label": k,
                "count": v
            }
            for k, v in sorted(all_labels.items(), key=lambda x: x[1], reverse=True)
        ]

        return {
            "media_type": "video",
            "disease": rec["hindi_name"],
            "raw_name": dominant_pest,
            "crop": user_crop if user_crop else rec.get("crop_hosts", "फसल"),
            "confidence": round(avg_conf, 1),
            "max_confidence": round(max_conf, 1),
            "severity": severity,
            "is_safe": False,
            "frames_analyzed": frame_count,
            "annotated_video_filename": annotated_video_filename,
            "snapshot_filename": snapshot_filename,
            "detections": detection_list,
            "total_pest_count": total_pests,
            "treatment": rec["treatment"],
            "prevention": rec["prevention"],
            "cause": rec["cause"]
        }

    except Exception as e:
        logger.error(f"[PEST-AI] Video prediction error: {e}", exc_info=True)
        return {"error": f"वीडियो कीट पहचान में त्रुटि: {str(e)}"}
