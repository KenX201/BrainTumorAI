import numpy as np
import cv2
import os, uuid
from keras.models import load_model, Model
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf

model_path = os.path.join(os.path.dirname(__file__), 'brain_tumor_densenet_adam_model.h5')
class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
input_sizes = (224,224)
results_dir = 'static/results'

# Load model
model = load_model(model_path, compile=False)

def _ensure_3ch(img):
    # Load and preprocess image
    if img.ndim == 2:
        img = np.stack([img, img, img], axis =-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img

def _preprocess(img_path):
    # read original for overlay later
    orig = cv2.imread(img_path) # BGR
    if orig is None:
        raise ValueError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    rgb = _ensure_3ch(rgb)
    resized = cv2.resize(rgb, input_sizes)
    arr = resized.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis = 0)
    return orig, arr

def _last_conv_layer(m):
    # try to find the last Conv2D layer automatically
    for layer in reversed(m.layers):
        try:
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def _grad_cam(model, img_array, target_index):
    # Find last conv
    last_conv_name = _last_conv_layer(model)
    if not last_conv_name:
        return None # cannot compute CAM
    
    conv_layer = model.get_layer(last_conv_name)

    grad_model = Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, target_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0].numpy()
    pooled = pooled_grads.numpy()
    for c in range(conv_out.shape[-1]):
        conv_out[:, :, c] *= pooled[c]

    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap

def _overlay_heatmap_on(orig_bgr, heatmap):
    h, w = orig_bgr.shape[:2]
    cam = cv2.resize(heatmap, (w, h))
    cam = np.uint8(255 * cam)
    cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # heatmap
    overlay = cv2.addWeighted(orig_bgr, 0.6, cam_color, 0.4, 0)
    return overlay

def run_inference(img_path):
    orig_bgr, arr = _preprocess(img_path)

    # forward pass
    preds = model.predict(arr)[0]  # shape (num_classes,) or scalar
    if preds.ndim == 0:
        preds = np.array([1 - preds, preds])  # binary fallback
    probs = preds / (np.sum(preds) + 1e-8)

    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

    # Grad-CAM (best-effort; may be None if model has no conv layers)
    heatmap = _grad_cam(model, arr, idx)
    if heatmap is not None:
        overlay = _overlay_heatmap_on(orig_bgr, heatmap)
    else:
        overlay = orig_bgr.copy()  # fallback: just show original

    # save overlay
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"overlay_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(results_dir, out_name)
    cv2.imwrite(out_path, overlay)

    # pack probabilities into (label -> percent) dict
    prob_dict = {}
    for i, p in enumerate(probs):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        prob_dict[name] = round(float(p) * 100, 2)

    return {
        "label": label,
        "overlay_path": out_path.replace("\\", "/"),
        "probs": prob_dict
    }

