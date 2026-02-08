import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# --- Persiapan data ---
# Kita menggunakan dataset Iris dari scikit-learn sebagai contoh.
iris = datasets.load_iris()
# Buat DataFrame agar lebih mudah dimanipulasi dan dibaca
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# `X` berisi fitur (input), `y` berisi target/label (output)
X = data.drop('target', axis=1)
# sklearn lebih suka array 1-d untuk target, jadi kita ambil sebagai Series
y = data['target']

from sklearn.model_selection import train_test_split
# Pisah data menjadi train dan test. test_size=0.3 => 30% data untuk evaluasi
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# --- Membangun model Decision Tree ---
# Tanpa parameter eksplisit, DecisionTreeClassifier memakai pengaturan default.
model = DecisionTreeClassifier()
# Melatih/pasang model ke data training
model.fit(X_train, y_train)

# --- Prediksi dan Evaluasi ---
y_pred_dt = model.predict(X_test)

print('Akurasi model Decision Tree pada data test:', accuracy_score(y_test, y_pred_dt))