import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk membuat data dummy dengan pola churn yang kuat
def generate_customer_data(n_samples=1000):
    np.random.seed(42)  # Untuk reproduksibilitas
    
    # Data umur (customer yang lebih muda dan lebih tua cenderung churn)
    age = np.random.randint(18, 70, n_samples)
    
    # Data jenis kelamin (0: perempuan, 1: laki-laki)
    gender = np.random.randint(0, 2, n_samples)
    gender_str = np.where(gender == 1, 'laki-laki', 'perempuan')
    
    # Data revenue dengan distribusi yang lebih realistis
    # Korelasi dengan churn: customer dengan revenue rendah lebih cenderung churn
    base_revenue = np.random.lognormal(mean=15, sigma=1, size=n_samples)
    revenue = np.round(base_revenue, -3)  # Bulatkan ke ribuan terdekat
    
    # Revenue threshold untuk penentuan churn
    revenue_threshold = np.percentile(revenue, 20)  # 20% customer dengan revenue terendah
    
    # Buat pola churn berdasarkan variabel
    # Base probability
    base_prob = 0.2
    
    # Probabilitas churn sangat dipengaruhi oleh revenue
    revenue_factor = np.zeros(n_samples)
    # Customer dengan revenue rendah memiliki kemungkinan churn yang sangat tinggi
    revenue_factor[revenue < revenue_threshold] = 0.8
    # Customer dengan revenue sangat tinggi memiliki kemungkinan churn sangat rendah
    revenue_factor[revenue > np.percentile(revenue, 80)] = -0.15
    
    # Umur juga berpengaruh (efek lebih kecil dari revenue)
    age_factor = np.zeros(n_samples)
    # Customer muda (<25) dan tua (>60) lebih mungkin churn
    age_factor[(age < 25) | (age > 60)] = 0.1
    
    # Gender hanya sedikit berpengaruh
    gender_factor = np.zeros(n_samples)
    gender_factor[gender == 0] = 0.05  # Misalnya, perempuan sedikit lebih mungkin churn
    
    # Final churn probability
    churn_prob = base_prob + revenue_factor + age_factor + gender_factor
    churn_prob = np.clip(churn_prob, 0.05, 0.95)  # Batasi probabilitas
    
    # Tentukan churn berdasarkan probabilitas
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Buat DataFrame
    df = pd.DataFrame({
        'umur': age,
        'jenis_kelamin': gender_str,
        'total_revenue': revenue,
        'churn': churn
    })
    
    return df

# Fungsi untuk menganalisis dan memvisualisasikan korelasi
def analyze_data(df):
    print(f"Shape data: {df.shape}")
    print("\nDistribusi churn:")
    print(df['churn'].value_counts(normalize=True))
    
    # Copy df untuk analisis
    df_analysis = df.copy()
    # Convert jenis_kelamin menjadi numerik untuk analisis korelasi
    df_analysis['jenis_kelamin_num'] = df_analysis['jenis_kelamin'].map({'laki-laki': 1, 'perempuan': 0})
    
    # Hitung korelasi
    corr_columns = ['umur', 'jenis_kelamin_num', 'total_revenue', 'churn']
    correlation = df_analysis[corr_columns].corr()
    print("\nKorelasi dengan churn:")
    print(correlation['churn'].sort_values(ascending=False))
    
    # Visualisasi distribusi dan korelasi
    plt.figure(figsize=(15, 10))
    
    # Plot distribusi revenue by churn
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='total_revenue', hue='churn', bins=30, element='step')
    plt.title('Distribusi Revenue berdasarkan Churn')
    plt.xlabel('Total Revenue')
    plt.xlim(0, np.percentile(df['total_revenue'], 95))  # Limit x axis for better visualization
    
    # Plot distribusi umur by churn
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='umur', hue='churn', bins=30, element='step')
    plt.title('Distribusi Umur berdasarkan Churn')
    plt.xlabel('Umur')
    
    # Plot churn rate by revenue group
    plt.subplot(2, 2, 3)
    df['revenue_group'] = pd.qcut(df['total_revenue'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    churn_by_revenue = df.groupby('revenue_group')['churn'].mean()
    sns.barplot(x=churn_by_revenue.index, y=churn_by_revenue.values)
    plt.title('Churn Rate berdasarkan Revenue Group')
    plt.xlabel('Revenue Group')
    plt.ylabel('Churn Rate')
    
    # Plot correlation heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('churn_analysis.png')
    print("\nVisualisasi disimpan ke churn_analysis.png")
    
    return df_analysis

# Membuat data
print("Membuat dataset dummy...")
customer_data = generate_customer_data(10000)

# Analisis data
analyze_data(customer_data)

# Simpan data ke CSV untuk referensi
customer_data.to_csv("customer_churn_data.csv", index=False)
print("Dataset disimpan ke customer_churn_data.csv")

# Persiapan data untuk pemodelan
X = customer_data.drop('churn', axis=1)
y = customer_data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline preprocessing dan model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['umur', 'total_revenue']),
        ('cat', OneHotEncoder(drop='first'), ['jenis_kelamin'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Latih model
print("Melatih model...")
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Akurasi model: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred))

# Feature importance
if hasattr(model[-1], 'feature_importances_'):
    # Get feature names after preprocessing
    cat_features = model[0].transformers_[1][1].get_feature_names_out(['jenis_kelamin'])
    feature_names = ['umur', 'total_revenue'] + list(cat_features)
    
    importances = model[-1].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nPentingnya Fitur:")
    for i, idx in enumerate(indices):
        if i < len(feature_names):
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# Simpan model
joblib.dump(model, 'customer_churn_model.joblib')
print("Model disimpan ke customer_churn_model.joblib")

# Contoh prediksi
print("\nContoh prediksi:")
sample = pd.DataFrame({
    'umur': [25, 65], 
    'jenis_kelamin': ['laki-laki', 'perempuan'], 
    'total_revenue': [1000000, 8000000]
})
print(sample)
print("Prediksi churn:", model.predict(sample))
print("Probabilitas churn:", model.predict_proba(sample)[:, 1])