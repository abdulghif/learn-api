from fastapi import FastAPI, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Customer Management API")

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua method
    allow_headers=["*"],  # Izinkan semua header
)

# Path file untuk penyimpanan data
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

# Fungsi untuk menulis data ke file
def write_customers(customers):
    with open(DATA_FILE, "w") as file:
        file.write(json.dumps(customers, indent=2))

# Mendapatkan ID baru untuk customer
def get_new_id(customers):
    if not customers:
        return 1
    return max(customer["id"] for customer in customers) + 1

# API endpoints
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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

# Endpoint untuk mencari customer berdasarkan nama (GET)
@app.get("/customers/search/", response_model=List[CustomerResponse])
def search_customers(nama: str = Query(..., min_length=1)):
    customers = read_customers()
    results = [c for c in customers if nama.lower() in c["nama"].lower()]
    return results

# Endpoint untuk mendapatkan statistik customers (GET)
@app.get("/stats/")
def get_stats():
    customers = read_customers()
    
    if not customers:
        return {
            "total_customers": 0,
            "average_age": 0,
            "total_revenue": 0
        }
    
    total = len(customers)
    avg_age = sum(c["umur"] for c in customers) / total
    total_revenue = sum(c["total_revenue"] for c in customers)
    
    return {
        "total_customers": total,
        "average_age": round(avg_age, 2),
        "total_revenue": total_revenue
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)