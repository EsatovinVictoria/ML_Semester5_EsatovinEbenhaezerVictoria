---
title: AntiBurnout Predictor
emoji: 🧘
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---

# AntiBurnout AI - Burnout Risk Predictor

Aplikasi Machine Learning berbasis Flask & Docker untuk memprediksi tingkat risiko burnout kerja (Low, Moderate, High) menggunakan model Gaussian Naive Bayes yang telah dilatih dan distandarisasi pada proyek PBL AntiBurnout.

## Panduan Deployment ke Hugging Face Spaces

1. Buka [Hugging Face Spaces](https://huggingface.co/new-space).
2. Isi nama Space (misal: `antiburnout-classifier`).
3. Pilih Space SDK: **Docker** (Blank template).
4. Buat Access Token (Write) melalui menu Settings > Access Tokens pada akun Hugging Face Anda.
5. Clone repository Space kosong tersebut ke komputer Anda:
   ```bash
   git clone https://huggingface.co/spaces/<USERNAME>/<NAMA_SPACE>
   ```
6. Salin seluruh file dari folder ini (`app.py`, `Dockerfile`, `requirements.txt`, `burnout_model.pkl`, `burnout_scaler.pkl`, `burnout_metadata.pkl`) ke dalam folder hasil clone.
7. Lakukan commit dan push ke Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy aplikasi AntiBurnout ke Hugging Face"
   git push
   ```
8. Masukkan username Hugging Face Anda dan Access Token (Write) sebagai password.
9. Buka tab App di halaman Space Anda di browser dan tunggu status berubah menjadi *Running*.
