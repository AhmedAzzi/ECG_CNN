from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import shutil

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

import torch
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from main import LSTM
from utils import hierarchical_kalman_filter
import json
from collections import Counter

app = Flask(__name__)

# Configure static folder for images
app.static_folder = 'static'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LSTM()
model_path = os.path.join(current_dir, 'models', 'heart_signal_model.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Class mapping
CLASS_MAPPING = {
    0: 'Normal Beat (N)',
    1: 'Atrial Premature Beat (A)',
    2: 'Ventricular Premature Beat (V)',
    3: 'Left Bundle Branch Block Beat (L)',
    4: 'Right Bundle Branch Block Beat (R)'
}

def process_ecg_data(record_name):
    """Process ECG data file."""
    try:
        # Read the record
        record = wfdb.rdrecord(record_name, channel_names=['MLII'])
        data = record.p_signal.flatten()
        
        # Apply Kalman filter denoising
        Q1, R1 = 0.001, 10
        Q2, R2 = 0.001, 1
        denoised_data = hierarchical_kalman_filter(data, Q1, R1, Q2, R2)
        
        # Plot original and denoised signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(data[:1000])
        plt.title('Original ECG Signal')
        plt.subplot(2, 1, 2)
        plt.plot(denoised_data[:1000])
        plt.title('Denoised ECG Signal')
        
        # Save plot to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()  # Close the figure to free memory
        
        # Encode the image
        graph = base64.b64encode(image_png).decode('utf-8')
        
        return denoised_data, graph
    except Exception as e:
        print(f"Error processing ECG data: {str(e)}")
        return None, None

def predict_segment(segment):
    """Make prediction on a single ECG segment."""
    try:
        # Ensure the model is on the same device as the segment tensor
        model.to(device)
        
        with torch.no_grad():
            # Convert the segment to a tensor and move it to the correct device
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make the prediction
            prediction = model(segment_tensor)
            
            # Get the predicted class and probabilities
            predicted_class = torch.argmax(prediction, dim=1).item()
            probabilities = torch.nn.functional.softmax(prediction, dim=1)[0]
            
            return predicted_class, probabilities.cpu().numpy()
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None

def calculate_final_prediction(predictions):
    """Calculate the most likely class, distribution, and average probabilities."""
    class_counts = Counter(pred['prediction'] for pred in predictions)
    final_class = max(class_counts, key=class_counts.get)
    confidence = (class_counts[final_class] / len(predictions)) * 100

    # Aggregate probabilities
    total_probabilities = {cls: 0 for cls in CLASS_MAPPING.values()}
    for prediction in predictions:
        for cls, prob in prediction['probabilities'].items():
            total_probabilities[cls] += prob

    average_probabilities = {cls: prob / len(predictions) for cls, prob in total_probabilities.items()}

    return {
        'class': final_class,
        'confidence': round(confidence, 2),
        'distribution': class_counts,
        'average_probabilities': average_probabilities
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Create a temporary directory for the record
        record_dir = os.path.join(UPLOAD_FOLDER, 'temp_record')
        if os.path.exists(record_dir):
            shutil.rmtree(record_dir)
        os.makedirs(record_dir)
        
        # Save all uploaded files
        record_name = None
        for file in files:
            if file and file.filename:
                filename = file.filename
                file_path = os.path.join(record_dir, filename)
                file.save(file_path)
                
                # Get the record name from the first file (without extension)
                if record_name is None:
                    record_name = os.path.splitext(filename)[0]
        
        if not record_name:
            return jsonify({'error': 'Invalid record files'}), 400
        
        # Full path to the record (without extension)
        record_path = os.path.join(record_dir, record_name)
        
        # Process the ECG data
        denoised_data, signal_plot = process_ecg_data(record_path)
        if denoised_data is None:
            return jsonify({'error': 'Error processing ECG data'}), 400
        
        # Make predictions on segments
        results = []
        for i in range(0, len(denoised_data) - 300, 300):
            segment = denoised_data[i:i+300]
            if len(segment) == 300:
                pred_class, probs = predict_segment(segment)
                if pred_class is not None:
                    results.append({
                        'segment': i // 300 + 1,
                        'prediction': CLASS_MAPPING[pred_class],
                        'probabilities': {CLASS_MAPPING[i]: float(prob) for i, prob in enumerate(probs)}
                    })
        
        # Calculate final prediction summary
        final_prediction = calculate_final_prediction(results)
        
        return jsonify({
            'signal_plot': signal_plot,
            'predictions': results,
            'final_prediction': final_prediction
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    finally:
        # Clean up uploaded files
        if os.path.exists(record_dir):
            shutil.rmtree(record_dir)

@app.route('/metrics')
def metrics():
    return render_template('metrics.html',
                         accuracy_img='accuracy.png',
                         loss_img='loss.png',
                         confusion_matrix_img='confusion_matrix.png')

if __name__ == '__main__':
    # Ensure all necessary directories exist
    for directory in ['uploads', 'static', 'logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    app.run(debug=True)
