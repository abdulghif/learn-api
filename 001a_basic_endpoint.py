from fastapi import FastAPI

app = FastAPI(title="Customer Management API")

# API endpoints
@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)