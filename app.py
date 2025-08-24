from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid
import numpy as np
import cv2
import base64
import tensorflow as tf
from keras.models import load_model, Model
from keras.preprocessing import image

# Flask
app = Flask(__name__)
UPLOAD_DIR = 'static/uploads'
RESULTS_DIR = 'static/results'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Config
model_path = 'model/brain_tumor_densenet_adam_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
input_sizes = (224,224)
enable_cam = True

# Load model
try:
    model = load_model(model_path, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, input_sizes  = (224,224)):
    img = image.load_img(img_path, target_size=input_sizes)

    # Preprocess image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

def predict_tumor(img_path):
    """
    Use trained model to predict tumor type
    """
    if model is None:
        return "Error: Model not loaded", None
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        # Get class name
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error during prediction", None

def generate_heatmap(img_path):
    """
    Generate a heatmap visualization for the image
    This is a simplified version - you might want to implement
    Grad-CAM or another visualization technique specific to your model
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    
    # Create a simple heatmap based on the prediction
    # Replace this with your actual model's heatmap generation
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', heatmap)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    # Main page
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get tumor diagnosis
        diagnosis, confidence = predict_tumor(filepath)
        
        # Generate heatmap
        heatmap_base64 = generate_heatmap(filepath, diagnosis)
        
        # Process the image for display
        img = cv2.imread(filepath)
        img = cv2.resize(img, (300, 300))
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'diagnosis': diagnosis,
            'confidence': float(confidence) if confidence else 0,
            'image': img_base64,
            'heatmap': heatmap_base64
        }
        
        return jsonify(response)
    
    return jsonify({'error': 'Invalid file format'})
    
if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)
