# Customer Churn Prediction API

This repository contains a machine learning API for predicting customer churn based on customer data.

## Project Structure

```
├── .gitignore                      # Git ignore file
├── 000_basic_api_scrape.py         # Basic API introduction: scraping Google Images
├── 001_basic_endpoint.py           # Basic API endpoint introduction
├── 002_create_endpoint.py          # API endpoint creation script
├── 003_model_endpoint.py           # Model endpoint implementation
├── 004_public_endpoint.py          # Public API endpoint setup
├── Accessing Public API.ipynb      # Jupyter notebook for accessing the API
├── churn_analysis.png              # Visualization of churn analysis
├── customer_churn_data.csv         # Customer churn dataset
├── customer_churn_model.joblib     # Saved machine learning model
├── customers.txt                   # Customer data text file
├── LICENSE                         # License file
├── modeling.py                     # Model training script
├── README.md                       # This file
├── requirements.txt                # Required Python packages
```

## Getting Started

### Prerequisites

To use this project, you'll need:
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/customer-churn-api.git
   cd customer-churn-api
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### API Development Progression

This repository shows the step-by-step development of an API:

1. **Web Scraping API** (`000_basic_api_scrape.py`): Initial API introduction to scrape Google Images
2. **Basic Endpoint** (`001_basic_endpoint.py`): Introduction to API endpoints
3. **Creating Endpoints** (`002_create_endpoint.py`): How to create API endpoints
4. **Model Integration** (`003_model_endpoint.py`): Integrating ML models into API endpoints
5. **Public Deployment** (`004_public_endpoint.py`): Setting up public-facing endpoints

### Training a Model

To train the customer churn prediction model:

```
python modeling.py
```

This script will:
- Create customer data and then generate `customer_churn_data.csv`
- Train a machine learning model
- Save the model to `customer_churn_model.joblib`
- Generate analysis visualization (`churn_analysis.png`)

### Running the API

To run the basic API:

```
python 000_basic_api_scrape.py
```

For the complete API with the model integration:

```
python 004_public_endpoint.py
```

### Accessing the API

You can access the API using the Jupyter notebook `Accessing Public API.ipynb` which demonstrates how to make requests to the API endpoints.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request