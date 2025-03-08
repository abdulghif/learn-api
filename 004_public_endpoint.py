from fastapi import FastAPI, HTTPException, Depends, Path, Query, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import json
import os
import joblib
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyngrok import ngrok

# Your ngrok token - replace with your actual token
print("Put NGROX token here: ",end='')
NGROK_TOKEN = input()

app = FastAPI(title="Customer Management API")

# Path for data storage
DATA_FILE = "customers.txt"

# Model untuk customer
class Customer(BaseModel):
    id: Optional[int] = None
    nama: str = Field(..., min_length=1, max_length=100)
    umur: int = Field(..., gt=0, lt=150)
    jenis_kelamin: str = Field(..., pattern="^(laki-laki|perempuan)$")
    total_revenue: float = Field(..., ge=0)

class CustomerResponse(Customer):
    id: int

class CustomerUpdate(BaseModel):
    nama: Optional[str] = Field(None, min_length=1, max_length=100)
    umur: Optional[int] = Field(None, gt=0, lt=150)
    jenis_kelamin: Optional[str] = Field(None, pattern="^(laki-laki|perempuan)$")
    total_revenue: Optional[float] = Field(None, ge=0)

# Model untuk prediksi churn
class ChurnPredictionInput(BaseModel):
    umur: int = Field(..., gt=0, lt=150)
    jenis_kelamin: str = Field(..., pattern="^(laki-laki|perempuan)$")
    total_revenue: float = Field(..., ge=0)

# Update model untuk hasil prediksi
class ChurnPredictionResult(BaseModel):
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None
    churn_probability: float
    churn_prediction: bool
    message: str

class BulkPredictionResult(BaseModel):
    results: List[ChurnPredictionResult]
    summary: dict

# Fungsi untuk membaca data dari file
def read_customers():
    if not os.path.exists(DATA_FILE):
        return []
    
    try:
        with open(DATA_FILE, "r") as file:
            content = file.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []

# API endpoints
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Fungsi untuk menulis data ke file
def write_customers(customers):
    with open(DATA_FILE, "w") as file:
        file.write(json.dumps(customers, indent=2))

# Mendapatkan ID baru untuk customer
def get_new_id(customers):
    if not customers:
        return 1
    return max(customer["id"] for customer in customers) + 1

# Load model prediksi churn
def get_model():
    model_path = "customer_churn_model.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model belum dilatih. Jalankan modelling.py terlebih dahulu.")
    return joblib.load(model_path)

# Endpoint untuk menambahkan customer baru (POST)
@app.post("/customers/", response_model=CustomerResponse, status_code=201)
def create_customer(customer: Customer):
    customers = read_customers()
    
    # Buat ID baru
    new_customer = customer.dict()
    new_customer["id"] = get_new_id(customers)
    
    # Tambahkan ke daftar dan simpan
    customers.append(new_customer)
    write_customers(customers)
    
    return new_customer

# Endpoint untuk mendapatkan semua nama customer (GET)
@app.get("/customers/", response_model=List[str])
def get_customer_names():
    customers = read_customers()
    return [customer["nama"] for customer in customers]

# Endpoint untuk mendapatkan detail customer berdasarkan ID (GET)
@app.get("/customers/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: int = Path(..., gt=0)):
    customers = read_customers()
    
    for customer in customers:
        if customer["id"] == customer_id:
            return customer
    
    raise HTTPException(status_code=404, detail="Customer tidak ditemukan")

# Endpoint untuk mengupdate customer (PUT)
@app.put("/customers/{customer_id}", response_model=CustomerResponse)
def update_customer(update_data: CustomerUpdate, customer_id: int = Path(..., gt=0)):
    customers = read_customers()
    
    for i, customer in enumerate(customers):
        if customer["id"] == customer_id:
            # Update hanya field yang ada dalam request
            update_dict = update_data.dict(exclude_unset=True)
            customers[i].update(update_dict)
            write_customers(customers)
            return customers[i]
    
    raise HTTPException(status_code=404, detail="Customer tidak ditemukan")

# Endpoint untuk menghapus customer (DELETE)
@app.delete("/customers/{customer_id}", status_code=204)
def delete_customer(customer_id: int = Path(..., gt=0)):
    customers = read_customers()
    
    initial_count = len(customers)
    customers = [c for c in customers if c["id"] != customer_id]
    
    if len(customers) == initial_count:
        raise HTTPException(status_code=404, detail="Customer tidak ditemukan")
    
    write_customers(customers)
    return None

# ===== TAMBAHAN: ENDPOINT PREDIKSI CHURN =====

# Endpoint untuk prediksi churn satu customer (POST)
@app.post("/predict/churn", response_model=ChurnPredictionResult)
def predict_churn(data: ChurnPredictionInput):
    model = get_model()
    
    # Persiapkan data input
    input_df = pd.DataFrame([{
        'umur': data.umur,
        'jenis_kelamin': data.jenis_kelamin,
        'total_revenue': data.total_revenue
    }])
    
    # Lakukan prediksi
    churn_prob = model.predict_proba(input_df)[0, 1]
    churn_pred = churn_prob >= 0.5
    
    # Buat pesan berdasarkan hasil
    if churn_pred:
        message = "Customer berisiko tinggi churn. Pertimbangkan untuk melakukan retensi."
    else:
        message = "Customer memiliki risiko rendah untuk churn."
    
    return {
        "churn_probability": float(churn_prob),
        "churn_prediction": bool(churn_pred),
        "message": message
    }

# Tambahkan endpoint baru di main.py

# Endpoint untuk prediksi churn bulk dari customers yang sudah tersimpan
@app.get("/predict/churn/all-customers", response_model=BulkPredictionResult)
def predict_all_customers_churn():
    # Baca data customer dari file
    customers = read_customers()
    
    if not customers:
        raise HTTPException(status_code=404, detail="Tidak ada customer yang tersimpan")
    
    model = get_model()
    
    # Siapkan data untuk prediksi
    customer_data = []
    for customer in customers:
        # Pastikan customer memiliki semua field yang diperlukan
        if all(field in customer for field in ['umur', 'jenis_kelamin', 'total_revenue']):
            customer_data.append({
                'id': customer['id'],
                'nama': customer['nama'],
                'umur': customer['umur'],
                'jenis_kelamin': customer['jenis_kelamin'],
                'total_revenue': customer['total_revenue']
            })
    
    if not customer_data:
        raise HTTPException(status_code=400, detail="Tidak ada customer dengan data lengkap")
    
    # Buat DataFrame untuk prediksi
    df = pd.DataFrame(customer_data)
    
    # Lakukan prediksi
    input_features = df[['umur', 'jenis_kelamin', 'total_revenue']]
    predictions_prob = model.predict_proba(input_features)[:, 1]
    predictions = predictions_prob >= 0.5
    
    # Buat hasil
    results = []
    for i, customer in enumerate(customer_data):
        prob = predictions_prob[i]
        pred = predictions[i]
        
        if pred:
            message = f"Customer {customer['nama']} (ID: {customer['id']}) berisiko tinggi churn. Pertimbangkan untuk melakukan retensi."
        else:
            message = f"Customer {customer['nama']} (ID: {customer['id']}) memiliki risiko rendah untuk churn."
        
        results.append({
            "customer_id": customer['id'],
            "customer_name": customer['nama'],
            "churn_probability": float(prob),
            "churn_prediction": bool(pred),
            "message": message
        })
    
    # Hitung ringkasan
    summary = {
        "total_customers": len(results),
        "predicted_churn_count": int(sum(predictions)),
        "churn_rate": float(sum(predictions) / len(predictions)),
        "avg_churn_probability": float(predictions_prob.mean()),
        "high_risk_customers": [r["customer_name"] for i, r in enumerate(results) if predictions[i]]
    }
    
    return {
        "results": results,
        "summary": summary
    }

# At the bottom, modify the code to run uvicorn and ngrok
if __name__ == "__main__":
    # Define port
    port = 8000
    
    # Set ngrok auth token
    ngrok.set_auth_token(NGROK_TOKEN)
    
    # Open a ngrok tunnel to the HTTP server
    public_url = ngrok.connect(port).public_url
    print(f"ngrok tunnel active at: {public_url}")
    print(f"API Swagger UI available at: {public_url}/docs")
    
    # Start the FastAPI server
    print(f"Starting FastAPI server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=8000)