# 💼 Clustering Gaji dengan Unsupervised Learning

Proyek ini menggunakan algoritma *unsupervised learning* untuk melakukan **clustering gaji** berdasarkan fitur-fitur tertentu dalam dataset. Tujuannya adalah untuk mengelompokkan data individu ke dalam beberapa klaster berdasarkan kesamaan pola gaji.

## 🧠 Algoritma yang Digunakan

- **K-Means Clustering**
- (Opsional) PCA (Principal Component Analysis) untuk reduksi dimensi
- Elbow Method & Silhouette Score untuk menentukan jumlah klaster optimal


## 📌 Tujuan

- Mengidentifikasi pola atau kelompok individu berdasarkan gaji
- Mendeteksi kemungkinan segmentasi gaji berdasarkan fitur seperti jabatan, pendidikan, usia, dll
- Menyajikan visualisasi hasil klaster untuk interpretasi yang lebih mudah

## 🚀 Cara Menjalankan

1. Clone Repo:
   ```bash
   git clone https://github.com/Dionisius951/unsupervised.git
   cd unsupervised
2. Install Dependency:
   ```bash
   pip install -r requirements.txt
3. Jalankan Server:
   ```bash
   streamlit app.py
 
## Demo Web 
https://clusteringgaji24.streamlit.app/
