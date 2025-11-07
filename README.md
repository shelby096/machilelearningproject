# Diet Recommendation ML Project 🍎

## Overview
A production-level machine learning project that provides personalized diet recommendations based on individual health metrics, lifestyle factors, and dietary preferences.

## Dataset
The project uses a comprehensive diet dataset containing:
- **Patient Demographics**: Age, Gender, Height, Weight, BMI
- **Health Metrics**: Blood Pressure, Cholesterol, Blood Sugar, Chronic Diseases
- **Lifestyle Factors**: Exercise, Sleep, Daily Steps, Alcohol, Smoking
- **Dietary Information**: Current intake, preferences, allergies
- **Target Variables**: Recommended calories, macronutrients, and meal plans

## Project Structure
```
my_ds_project/
├── data/                    # Data storage and processing
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Production-ready source code
├── configs/                 # Configuration files
├── models/                  # Saved model artifacts
├── dashboards/              # Streamlit application
├── tests/                   # Testing suite
├── scripts/                 # Utility scripts
└── reports/                 # Final deliverables
```

## Features
- **Data Pipeline**: Automated ETL with validation
- **ML Models**: Multiple algorithms for diet recommendation
- **Web Interface**: Interactive Streamlit dashboard
- **Model Monitoring**: Performance tracking and validation
- **Production Ready**: Comprehensive testing and deployment

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run data pipeline: `python src/pipelines/etl_pipeline.py`
3. Train models: `python src/pipelines/ml_pipeline.py`
4. Launch dashboard: `streamlit run dashboards/streamlit_app.py`

## Deployment
The application is deployed on Streamlit Cloud for easy access and scalability.

## Technologies Used
- **ML**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: Streamlit
- **Testing**: pytest
- **Deployment**: Streamlit Cloud
