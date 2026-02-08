# Decission-tree-tutorial
Tutorial lengkap Decision Tree menggunakan Python, mencakup konsep dasar, proses training, evaluasi model, serta visualisasi pohon keputusan untuk kasus klasifikasi dan regresi.

## Cara Kerja Decision Tree

Decision Tree adalah model prediktif yang menyerupai struktur pohon untuk mengambil keputusan.
- Node internal: mewakili pengujian pada satu fitur (mis. apakah `petal length < 2.5`).
- Cabang (branch): hasil dari pengujian tersebut (benar/salah atau rentang nilai).
- Daun (leaf): berisi prediksi kelas (klasifikasi) atau nilai kontinu (regresi).

Algoritma secara rekursif memilih fitur dan titik pemisahan (split) yang terbaik untuk memisahkan data menurut ukuran impurity (mis. Gini atau Entropy). Pemilihan split didasarkan pada pengurangan impurity terbesar (information gain). Proses berhenti ketika: semua sampel di node memiliki label sama, tidak ada fitur tersisa, atau tercapai kriteria berhenti (kedalaman maksimal, jumlah sampel minimal, dll.). Untuk menghindari overfitting, biasanya dilakukan pruning atau disetel hyperparameter seperti `max_depth` dan `min_samples_leaf`.

## Komentar di dalam kode

File `main_decission_tree.py` sudah diperbarui dengan komentar berbahasa Indonesia yang menjelaskan setiap bagian utama: persiapan data, pemisahan train/test, pembuatan model, pelatihan, prediksi, dan evaluasi. Buka file tersebut untuk melihat penjelasan baris demi baris.

Untuk menjalankan contoh:

```bash
python main_decission_tree.py
```

Output akan menampilkan akurasi model pada data uji.

## Langkah demi langkah membangun Decision Tree

1. Pahami data
	- Lihat fitur, tipe data, nilai hilang, dan distribusi target.
2. Pra-pemrosesan
	- Tangani nilai hilang, encoding kategori (jika ada), dan normalisasi jika diperlukan.
3. Pilih fitur / engineering
	- Hilangkan fitur yang tidak relevan, buat fitur baru bila perlu.
4. Pisahkan data
	- Bagi menjadi data training dan testing (mis. 70/30 atau 80/20).
5. Pilih model dan parameter awal
	- Gunakan `DecisionTreeClassifier` atau `DecisionTreeRegressor`.
6. Latih model
	- Panggil `.fit()` pada data training.
7. Evaluasi model
	- Untuk klasifikasi gunakan akurasi, precision, recall, F1; untuk regresi gunakan MSE, MAE, R2.
8. Visualisasi dan interpretasi
	- Visualisasikan pohon untuk memahami aturan keputusan (contoh: `sklearn.tree.plot_tree`).
	  Contoh singkat visualisasi:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# model: DecisionTreeClassifier yang sudah dilatih
plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
plt.show()
```

9. Tuning dan Validasi
	- Gunakan cross-validation dan GridSearch/RandomizedSearch untuk menemukan hyperparameter terbaik (`max_depth`, `min_samples_split`, dll.).
10. Pruning / Regularisasi
	- Batasi kedalaman (`max_depth`) atau minimal sample per leaf untuk mengurangi overfitting.
11. Deploy dan Monitor
	- Setelah puas, simpan model (mis. dengan `joblib`) dan pantau performa di data produksi.

Jika Anda ingin, saya bisa:
- Menambahkan contoh visualisasi lengkap ke repo.
- Menambahkan pipeline kecil dengan GridSearchCV untuk tuning.

