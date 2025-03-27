import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

with open('logistic_model_l2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        df = pd.read_csv(file, header=None)
        if df.shape[1] != 54:
            return jsonify({'error': 'Uploaded file must contain exactly 54 columns'})

        
        predictions = model.predict(df.values)
        results = ['Spam' if pred == 1 else 'Not Spam' for pred in predictions]

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
