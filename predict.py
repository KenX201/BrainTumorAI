import numpy as np
import cv2
import os, uuid
from keras.models import load_model, Model
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf

model = load_model('brain_tumor_densenet_adam_model.h5')
class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
input_sizes = (224,224)
results_dir = 'static/results'

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


