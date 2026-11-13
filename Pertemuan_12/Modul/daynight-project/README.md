---
title: Day Night Classifier
emoji: ☀️
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
---

# Day vs Night Classifier

Aplikasi web deployment model Machine Learning untuk klasifikasi citra siang (Day) dan malam (Night) menggunakan ekstraksi fitur Histogram of Oriented Gradients (HOG) dan Neural Network.

## Panduan Deployment ke Hugging Face Spaces

1. Buat Space baru di [Hugging Face](https://huggingface.co/new-space).
2. Berikan nama space (misal `daynight-classifier`).
3. Pilih SDK: **Docker** (Blank template).
4. Buat Access Token (Write) di menu Settings > Access Tokens pada akun Hugging Face Anda.
5. Clone repository Space tersebut ke komputer Anda:
   ```bash
   git clone https://huggingface.co/spaces/<USERNAME>/<NAMA_SPACE>
   ```
6. Salin seluruh file dari folder ini (`app.py`, `Dockerfile`, `requirements.txt`, `day_night_model.h5`, `scaler.pkl`) ke dalam folder hasil clone.
7. Lakukan push ke Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy aplikasi Day vs Night ke Hugging Face"
   git push
   ```
8. Masukkan username Hugging Face Anda dan Access Token (Write) sebagai password saat diminta.
9. Buka tab App di halaman Space Hugging Face Anda dan tunggu proses *building* selesai hingga berstatus *Running*.
