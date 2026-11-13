import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load Model, Scaler, dan Metadata
MODEL_PATH = 'burnout_model.pkl'
SCALER_PATH = 'burnout_scaler.pkl'
META_PATH = 'burnout_metadata.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    print("✅ AntiBurnout System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading AntiBurnout system: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AntiBurnout AI - Deteksi Risiko Burnout</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 700px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 8px; font-size: 26px; }
        p.subtitle { text-align: center; color: #7f8c8d; margin-top: 0; margin-bottom: 25px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 13px; font-weight: 600; margin-bottom: 5px; color: #34495e; }
        input, select { padding: 9px 12px; border: 1px solid #ccd1d9; border-radius: 6px; font-size: 14px; transition: border 0.2s; }
        input:focus, select:focus { border-color: #3498db; outline: none; }
        .full-width { grid-column: span 2; }
        button { background-color: #2980b9; color: white; border: none; padding: 12px; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.2s; margin-top: 10px; }
        button:hover { background-color: #1f618d; }
        .result-box { margin-top: 25px; padding: 20px; border-radius: 8px; text-align: center; }
        .risk-low { background-color: #e8f8f5; border: 1px solid #a3e4d7; color: #117864; }
        .risk-moderate { background-color: #fef9e7; border: 1px solid #f9e79f; color: #b7950b; }
        .risk-high { background-color: #fdedec; border: 1px solid #f5b7b1; color: #922b21; }
        .recommendation { margin-top: 10px; font-size: 14px; color: #555; }
        .probabilities { margin-top: 12px; font-size: 12px; color: #777; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AntiBurnout AI</h1>
        <p class="subtitle">Sistem Deteksi Dini Tingkat Risiko Burnout Berbasis Machine Learning</p>
        
        <form action="/predict" method="post">
            <div class="form-grid">
                <div class="form-group">
                    <label>Usia (Tahun):</label>
                    <input type="number" name="Age" min="18" max="70" value="28" required>
                </div>
                <div class="form-group">
                    <label>Jam Kerja per Minggu:</label>
                    <input type="number" name="Work_Hours_Per_Week" min="10" max="100" value="45" required>
                </div>
                <div class="form-group">
                    <label>Jam Tidur per Hari:</label>
                    <input type="number" step="0.5" name="Sleep_Hours" min="2" max="14" value="6.5" required>
                </div>
                <div class="form-group">
                    <label>Tingkat Stres Kerja:</label>
                    <select name="Stress_Level">
                        <option value="Low">Low (Rendah)</option>
                        <option value="Moderate" selected>Moderate (Sedang)</option>
                        <option value="High">High (Tinggi)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Kepuasan Kerja (1 - 10):</label>
                    <input type="number" name="Job_Satisfaction" min="1" max="10" value="6" required>
                </div>
                <div class="form-group">
                    <label>Keseimbangan Kerja-Hidup (1 - 10):</label>
                    <input type="number" name="Work_Life_Balance" min="1" max="10" value="5" required>
                </div>
                <div class="form-group">
                    <label>Kualitas Tidur:</label>
                    <select name="Sleep_Quality">
                        <option value="Excellent">Excellent</option>
                        <option value="Good" selected>Good</option>
                        <option value="Average">Average</option>
                        <option value="Poor">Poor</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Skor Kecemasan (1 - 10):</label>
                    <input type="number" name="Anxiety_Score" min="1" max="10" value="5" required>
                </div>
                <div class="form-group">
                    <label>Skor Depresi (1 - 10):</label>
                    <input type="number" name="Depression_Score" min="1" max="10" value="4" required>
                </div>
                <div class="form-group">
                    <label>Skor Produktivitas (1 - 10):</label>
                    <input type="number" name="Productivity_Score" min="1" max="10" value="6" required>
                </div>
                <div class="form-group">
                    <label>Model Kerja (Remote/WFO):</label>
                    <select name="Remote_Work">
                        <option value="No">On-site (WFO)</option>
                        <option value="Hybrid" selected>Hybrid</option>
                        <option value="Yes">Remote (WFH)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Status Kesehatan Mental:</label>
                    <select name="Mental_Health_Status">
                        <option value="Healthy">Healthy (Sehat)</option>
                        <option value="Needs Attention" selected>Needs Attention (Perlu Perhatian)</option>
                        <option value="Critical">Critical (Kritis)</option>
                    </select>
                </div>
                <div class="form-group full-width">
                    <button type="submit">Analisis Risiko Burnout</button>
                </div>
            </div>
        </form>

        {% if prediction %}
        <div class="result-box {{ result_class }}">
            <h2 style="margin: 0;">Hasil Prediksi: {{ prediction }}</h2>
            <p class="recommendation">{{ recommendation }}</p>
            <p class="probabilities">Distribusi Probabilitas: Low: {{ prob_low }}% | Moderate: {{ prob_mod }}% | High: {{ prob_high }}%</p>
            <a href="/" style="display:inline-block; margin-top:10px; color:#2980b9; text-decoration:none; font-weight:600;">← Analisis Ulang</a>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

def prepare_feature_vector(input_dict):
    feature_names = metadata['feature_names']
    feature_means = metadata['feature_means']
    encoding_maps = metadata['encoding_maps']

    vector = []
    for col in feature_names:
        if col in input_dict and input_dict[col] is not None and input_dict[col] != '':
            val = input_dict[col]
            if col in encoding_maps:
                val = encoding_maps[col].get(str(val), encoding_maps[col].get(val, 0))
            else:
                try:
                    val = float(val)
                except ValueError:
                    val = feature_means.get(col, 0.0)
            vector.append(val)
        else:
            vector.append(feature_means.get(col, 0.0))
            
    return np.array(vector).reshape(1, -1)

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        feat_vector = prepare_feature_vector(input_data)
        scaled_vector = scaler.transform(feat_vector)
        
        pred_idx = model.predict(scaled_vector)[0]
        probs = model.predict_proba(scaled_vector)[0]
        
        target_labels = metadata['target_labels']
        label = target_labels[pred_idx]
        
        if label == 'Low':
            result_class = 'risk-low'
            recommendation = "Kondisi Anda saat ini seimbang. Pertahankan pola istirahat, manajemen waktu, dan aktivitas fisik secara teratur."
        elif label == 'Moderate':
            result_class = 'risk-moderate'
            recommendation = "Terdapat indikasi kelelahan kerja sedang. Kurangi lembur, luangkan waktu relaksasi, dan batasi beban kerja tambahan."
        else:
            result_class = 'risk-high'
            recommendation = "Tingkat risiko burnout tinggi! Sangat disarankan mengambil cuti, berkonsultasi dengan profesional atau konselor, serta mengatur ulang prioritas kerja."

        return render_template_string(
            HTML_TEMPLATE,
            prediction=label,
            result_class=result_class,
            recommendation=recommendation,
            prob_low=round(probs[0] * 100, 1),
            prob_mod=round(probs[1] * 100, 1),
            prob_high=round(probs[2] * 100, 1)
        )
    except Exception as e:
        return f"Terjadi kesalahan pemrosesan: {e}", 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        feat_vector = prepare_feature_vector(data)
        scaled_vector = scaler.transform(feat_vector)
        
        pred_idx = int(model.predict(scaled_vector)[0])
        probs = model.predict_proba(scaled_vector)[0].tolist()
        target_labels = metadata['target_labels']
        label = target_labels[pred_idx]
        
        return jsonify({
            'status': 'success',
            'prediction': label,
            'prediction_index': pred_idx,
            'probabilities': {
                target_labels[i]: round(probs[i], 4) for i in range(len(target_labels))
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
