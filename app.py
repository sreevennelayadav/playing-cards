import csv
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Playing Card Classification", layout="centered")

st.title("Playing Card Classification")
st.write("Upload a playing card image to predict its class.")

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "cards-image-datasetclassification" / "cards.csv"
DEFAULT_IMAGE_SIZE = (200, 200)
LOW_CONFIDENCE_THRESHOLD = 30.0


class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Older exports may serialize `groups=1` for DepthwiseConv2D.
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


CUSTOM_OBJECTS = {
    "DepthwiseConv2D": DepthwiseConv2DCompat,
}


@contextmanager
def dense_input_compat_patch():
    original_call = tf.keras.layers.Dense.__call__

    def patched_call(self, inputs, *args, **kwargs):
        # Some older/corrupted exports pass duplicate tensors into Dense.
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = inputs[0]
        return original_call(self, inputs, *args, **kwargs)

    tf.keras.layers.Dense.__call__ = patched_call
    try:
        yield
    finally:
        tf.keras.layers.Dense.__call__ = original_call


def discover_model_candidates():
    candidates = [
        BASE_DIR / "cards_model.keras",
        BASE_DIR / "cards_model.h5",
        BASE_DIR / "cards-image-datasetclassification" / "53cards-53-(200 X 200)-100.00.h5",
        BASE_DIR / "cards-image-datasetclassification" / "14card types-14-(200 X 200)-94.61.h5",
    ]

    for model_path in sorted((BASE_DIR / "cards-image-datasetclassification").glob("*.h5")):
        if model_path.name not in [p.name for p in candidates]:
            candidates.append(model_path)

    # Remove duplicates while preserving order and skip temporary office files.
    deduped = []
    seen = set()
    for path in candidates:
        if path.name.startswith("~$"):
            continue
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            deduped.append(path)
    return deduped


def run_inference(model, processed_img):
    try:
        return model.predict(processed_img, verbose=0)
    except Exception:
        output = model(processed_img, training=False)
        if isinstance(output, (list, tuple)):
            output = output[0]
        if hasattr(output, "numpy"):
            return output.numpy()
        return np.array(output)


def get_model_image_size(model):
    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], tuple):
        input_shape = input_shape[0]

    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        h, w = input_shape[1], input_shape[2]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (h, w), True

    # These model files are commonly trained at 200x200.
    return DEFAULT_IMAGE_SIZE, False


def model_has_rescaling_layer(model):
    def walk_layers(layers):
        for layer in layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
            if hasattr(layer, "layers") and walk_layers(layer.layers):
                return True
        return False

    return walk_layers(getattr(model, "layers", []))


def smoke_test_model(model):
    size, _ = get_model_image_size(model)
    sample = np.zeros((1, size[0], size[1], 3), dtype="float32")
    prediction = run_inference(model, sample)
    if prediction is None:
        raise RuntimeError("Model returned no output during smoke test.")


@st.cache_resource
def load_trained_model():
    load_errors = []
    model_candidates = discover_model_candidates()

    for model_path in model_candidates:
        if not model_path.exists():
            continue

        try:
            if model_path.suffix.lower() == ".keras":
                # `safe_mode=False` improves compatibility with older Keras exports.
                model = load_model(
                    model_path,
                    compile=False,
                    safe_mode=False,
                    custom_objects=CUSTOM_OBJECTS,
                )
            else:
                model = load_model(
                    model_path,
                    compile=False,
                    custom_objects=CUSTOM_OBJECTS,
                )
            smoke_test_model(model)
            return model, model_path.name
        except Exception as exc:
            first_error = str(exc)
            try:
                with dense_input_compat_patch():
                    if model_path.suffix.lower() == ".keras":
                        model = load_model(
                            model_path,
                            compile=False,
                            safe_mode=False,
                            custom_objects=CUSTOM_OBJECTS,
                        )
                    else:
                        model = load_model(
                            model_path,
                            compile=False,
                            custom_objects=CUSTOM_OBJECTS,
                        )
                smoke_test_model(model)
                return model, f"{model_path.name} (compat mode)"
            except Exception as compat_exc:
                load_errors.append(
                    f"{model_path.name}: {first_error}\n{model_path.name} (compat mode): {compat_exc}"
                )

    if not load_errors:
        searched = ", ".join(str(path.name) for path in model_candidates)
        raise FileNotFoundError(f"No model file found. Looked for: {searched}")

    details = "\n".join(load_errors)
    raise RuntimeError(f"Failed to load available model files:\n{details}")


@st.cache_data
def load_class_names():
    if CSV_PATH.exists():
        class_lookup = {}
        with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    idx = int(row["class index"])
                except (TypeError, ValueError, KeyError):
                    continue
                label = str(row.get("labels", "")).strip()
                if label and idx not in class_lookup:
                    class_lookup[idx] = label

        if class_lookup:
            return [class_lookup[i] for i in sorted(class_lookup)]

    train_dir = BASE_DIR / "cards-image-datasetclassification" / "train"
    if train_dir.exists():
        return sorted(
            [entry.name for entry in train_dir.iterdir() if entry.is_dir()]
        )

    return []


def preprocess_image(image: Image.Image, target_size, normalize_to_unit=True):
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, target_size)
    img = img.astype("float32")
    if normalize_to_unit:
        img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def detect_playing_card(image: Image.Image):
    frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w

    def is_card_like_contour(contour):
        area = cv2.contourArea(contour)
        if area < 0.02 * img_area or area > 0.95 * img_area:
            return False

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            return False

        aspect_ratio = w / float(h)
        # Card can appear portrait or landscape depending on rotation/crop.
        ratio_ok = (0.45 <= aspect_ratio <= 0.85) or (1.15 <= aspect_ratio <= 2.2)
        if not ratio_ok:
            return False

        rect_area = w * h
        extent = area / float(rect_area) if rect_area > 0 else 0.0
        if extent < 0.55:
            return False

        return True

    # Pass 1: edge-based detection.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if is_card_like_contour(contour):
            return True

    # Pass 2: foreground mask using border-color background estimate.
    border_pixels = np.concatenate(
        [
            frame[0, :, :],
            frame[-1, :, :],
            frame[:, 0, :],
            frame[:, -1, :],
        ],
        axis=0,
    )
    bg_color = np.median(border_pixels, axis=0)
    color_dist = np.linalg.norm(frame.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    # Lower threshold to handle subtle card-vs-background differences.
    fg_mask = (color_dist > 8).astype(np.uint8) * 255
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if is_card_like_contour(contour):
            return True

    # Pass 3: bright-card-body detection (useful for clean renders on light backgrounds).
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    card_mask = cv2.inRange(hsv, (0, 0, 130), (180, 90, 255))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if not is_card_like_contour(contour):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Ignore candidates touching the frame edge; true card is usually fully inside.
        if x <= 1 or y <= 1 or (x + w) >= (img_w - 1) or (y + h) >= (img_h - 1):
            continue

        roi_gray = gray[y : y + h, x : x + w]
        dark_ratio = float(np.mean(roi_gray < 80))
        if dark_ratio >= 0.005:
            return True

    return False


def label_for_index(class_index, class_names, num_classes):
    if class_names and num_classes == len(class_names) and class_index < len(class_names):
        return class_names[class_index]
    return f"Class {class_index}"


def predict_with_fallback_sizes(
    model, image: Image.Image, primary_size, has_explicit_size, normalize_to_unit
):
    errors = []
    candidate_sizes = [primary_size]
    if (not has_explicit_size) and primary_size != DEFAULT_IMAGE_SIZE:
        candidate_sizes.append(DEFAULT_IMAGE_SIZE)

    for size in candidate_sizes:
        try:
            processed_img = preprocess_image(image, size, normalize_to_unit=normalize_to_unit)
            prediction = run_inference(model, processed_img)
            return prediction, size, candidate_sizes
        except Exception as exc:
            errors.append(f"{size}: {exc}")

    raise RuntimeError(
        "Inference failed.\n" + "\n".join(errors)
    )

try:
    model, loaded_model_name = load_trained_model()
    class_names = load_class_names()
    model_image_size, has_explicit_size = get_model_image_size(model)
    normalize_to_unit = not model_has_rescaling_layer(model)
    st.success(f"Model loaded successfully ({loaded_model_name}).")
    st.caption(
        f"Preprocess mode: {'scaled 0..1' if normalize_to_unit else 'raw 0..255'} | Input size: {model_image_size}"
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a card image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        card_detected = detect_playing_card(image)

        prediction, used_size, tried_sizes = predict_with_fallback_sizes(
            model, image, model_image_size, has_explicit_size, normalize_to_unit
        )
        class_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction) * 100)
        if len(tried_sizes) > 1 and used_size != tried_sizes[0]:
            st.info(f"Used fallback inference size: {used_size}")
        output_shape = model.output_shape
        if isinstance(output_shape, (list, tuple)) and output_shape and isinstance(output_shape[0], tuple):
            num_classes = output_shape[0][-1]
        else:
            num_classes = output_shape[-1]

        predicted_label = label_for_index(class_index, class_names, num_classes)
        if class_names and num_classes != len(class_names):
            st.warning(
                f"Label file has {len(class_names)} classes, but model outputs {num_classes}."
            )

        st.subheader("Prediction Result")
        if (not card_detected) and confidence < LOW_CONFIDENCE_THRESHOLD:
            st.error("No playing card detected in the uploaded image.")
            st.caption("Reason: detector did not find a card-like region and confidence is low.")
            st.stop()

        if not card_detected:
            st.warning(
                "Card detector is uncertain, but classification confidence is sufficient to show a best guess."
            )

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"Low confidence ({confidence:.2f}%). The model is uncertain and this may be wrong."
            )
            st.write("**Predicted Class (best guess):** " + predicted_label)
        else:
            st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        top_k = min(5, prediction.shape[-1] if prediction.ndim > 1 else 1)
        if prediction.ndim > 1 and top_k > 1:
            probs = prediction[0]
            top_indices = np.argsort(probs)[::-1][:top_k]
            st.write("**Top Predictions:**")
            for idx in top_indices:
                label = label_for_index(int(idx), class_names, num_classes)
                st.write(f"- {label}: {float(probs[idx] * 100):.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")