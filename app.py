from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid
from predict import run_inference

app = Flask(__name__)

# Folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html', config=app.config)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify(successs=False, error="Please choose an image."), 400
    
    # Save upload with a short random prefix to avoid collisions

    name = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex[:6]}_{name}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    try:
        result = run_inference(upload_path) # {label, overlay_path, probs}
        # Ensure returned paths are browser-severable URLs
        origina_url = '/' + upload_path.replace('\\','/')
        overlay_url = '/' + result["overlay_path"].lstrip('/').replace('\\','/')
        return jsonify(success=True,
                       label= result["label"],
                       probs=result["probs"],
                       original_path=origina_url,
                       overlay_path=overlay_url)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    
if __name__ == '__main__':
    app.run(debug=True)
