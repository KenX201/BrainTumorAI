from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model, Model

# Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOAD_DIR = 'static/uploads'
RESULTS_DIR = 'static/results'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Config
model_path = 'brain_tumor_densenet_adam_model.h5'
class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
input_sizes = (224,224)
enable_cam = True

_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model(model_path, compile=False)
    return _model

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
    arr = (resized.astype(np.float32) / 255.0)[None, ...]
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

    heatmap = np.maximum(np.mean(conv_out, axis=-1), 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

def _overlay_heatmap_on(orig_bgr, heatmap):
    # Resize heatmap to original size and blend with JET colormap
    h, w = orig_bgr.shape[:2]
    cam = cv2.resize(heatmap, (w, h))
    cam = np.uint8(255 * cam)
    cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # heatmap
    overlay = cv2.addWeighted(orig_bgr, 0.6, cam_color, 0.4, 0)
    return overlay

def run_inference(img_path):
    # Preprocess -> predict -> Grad-CAM -> save overlay -> return dict
    model = get_model()
    orig_bgr, x = _preprocess(img_path)

    # forward pass
    preds = model.predict(x)[0]  # shape (num_classes,) or scalar
    preds = np.array(preds, dtype=np.float32)
    probs = preds / (np.sum(preds) + 1e-8)

    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

    # Grad-CAM (best-effort; may be None if model has no conv layers)
    heatmap = _grad_cam(model, x, idx)
    if heatmap is not None:
        overlay = _overlay_heatmap_on(orig_bgr, heatmap)
    else:
        overlay = orig_bgr.copy()  # fallback: just show original

    # save overlay
    out_name = f"overlay_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(RESULTS_DIR, out_name)
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

@app.route('/',methods=['GET'])
def index():
    # Main page
    return render_template('index.html', config=app.config)

@app.route('/predict', methods=['POST'])
def predict_route():
    file = request.files.get('file')
    if not file or file.filename.strip() == '':
        return jsonify(successs=False, error="Please choose an image."), 400
    
    # Save upload with a short random prefix to avoid collisions

    name = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex[:6]}_{name}"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    try:
        result = run_inference(upload_path) # {label, overlay_path, probs}
        # Ensure returned paths are browser-severable URLs
        origina_url = '/' + upload_path.replace('\\','/')
        overlay_url = '/' + result["overlay_path"].lstrip('/').replace('\\','/')
        return jsonify(
            success=True,           
            label= result["label"],
            probs=result["probs"],
            original_path=origina_url,
            overlay_path=overlay_url)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    
if __name__ == '__main__':
    app.run(debug=True)
