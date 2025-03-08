import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load the iris dataset and train a model
iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model for future use (optional)
joblib.dump(model, 'iris_model.pkl')

# Load the model (for production use, you may load it once at the start)
model = joblib.load('iris_model.pkl')

# Define the FastAPI app
app = FastAPI()

# Define the request body using Pydantic
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# API endpoints
@app.get("/")
async def root():
    return {"message": "Iris Predictions"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Define the prediction endpoint
@app.post("/predict")
def predict(iris_request: IrisRequest):
    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(
        [[iris_request.sepal_length, iris_request.sepal_width,
          iris_request.petal_length, iris_request.petal_width]],
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the predicted species
    species = iris.target_names[prediction][0]
    return {"predicted_species": species}

# Add this at the end to run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)