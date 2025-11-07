# Diet Recommendation System - Deployment Guide

## 🚀 Quick Start

### Local Development
1. **Setup Environment**
   ```bash
   python setup.py
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the App**
   - Open your browser to `http://localhost:8501`

### Manual Setup (Alternative)
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   python run_pipeline.py
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run app.py
   ```

## 🌐 Streamlit Cloud Deployment

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account (free at share.streamlit.io)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Diet Recommendation ML Project"
   git branch -M main
   git remote add origin https://github.com/yourusername/diet-recommendation.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration**
   - The app will automatically use the `.streamlit/config.toml` file
   - Models will be loaded from the `models/` directory
   - Ensure all required files are in the repository

### Required Files for Deployment
```
my_ds_project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Model training script
├── models/                         # Trained models (auto-generated)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── preprocessors.joblib
├── data/
│   └── raw/curadiet-g.csv         # Dataset
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── README.md                      # Project documentation
```

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.streamlit/secrets.toml` file for sensitive configurations:
```toml
# Add any API keys or sensitive data here
# This file should not be committed to version control
```

### Model Configuration
The application automatically loads:
- Best performing model: `models/logistic_regression.joblib`
- Preprocessors: `models/preprocessors.joblib`
- Feature configuration from training pipeline

## 📊 Performance Metrics

Current model performance:
- **Best Model**: Logistic Regression
- **Accuracy**: 25.7%
- **Dataset Size**: 5,000 samples
- **Features**: 33 engineered features

### Model Improvement Suggestions
1. **Data Quality**: Collect more diverse and balanced data
2. **Feature Engineering**: Add more domain-specific features
3. **Model Tuning**: Implement hyperparameter optimization
4. **Ensemble Methods**: Combine multiple models for better performance

## 🧪 Testing

### Run Test Suite
```bash
python -m pytest tests/ -v
```

### Test Coverage
- Data quality validation
- Model accuracy verification
- Pipeline integrity checks

## 📁 Project Structure

```
my_ds_project/
├── 📄 README.md                    # Project overview
├── 📦 requirements.txt             # Dependencies
├── 🧹 .gitignore                   # Git ignore rules
├── ⚙️ setup.py                     # Setup script
├── 🚀 run_pipeline.py              # Main pipeline
├── 🌐 app.py                       # Streamlit app
├── 📊 DEPLOYMENT.md                # This file
│
├── 📂 data/                        # Data storage
│   ├── raw/                        # Original dataset
│   ├── interim/                    # Intermediate data
│   └── processed/                  # Final datasets
│
├── 📂 src/                         # Source code
│   ├── data/                       # Data processing
│   ├── features/                   # Feature engineering
│   ├── models/                     # ML models
│   ├── pipelines/                  # ETL & ML pipelines
│   └── utils/                      # Utilities
│
├── 📂 models/                      # Trained models
├── 📂 reports/                     # Analysis reports
├── 📂 tests/                       # Test suite
├── 📂 logs/                        # Application logs
└── 📂 .streamlit/                  # Streamlit config
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure models are trained: `python run_pipeline.py`
   - Check file paths in `models/` directory

2. **Import Errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

3. **Data Loading Issues**
   - Verify dataset exists: `data/raw/curadiet-g.csv`
   - Check file permissions

4. **Streamlit Deployment Issues**
   - Ensure all files are committed to Git
   - Check Streamlit Cloud logs for errors
   - Verify requirements.txt includes all dependencies

### Performance Optimization

1. **Model Caching**
   - Models are cached using `@st.cache_data`
   - Clear cache if models are updated

2. **Memory Usage**
   - Large datasets are processed in chunks
   - Consider data sampling for faster development

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review application logs in `logs/` directory
3. Run the test suite to identify issues
4. Check model performance in `reports/` directory

## 🎯 Next Steps

1. **Model Improvement**
   - Collect more training data
   - Implement advanced feature engineering
   - Try ensemble methods

2. **Feature Enhancement**
   - Add user feedback system
   - Implement meal planning features
   - Add nutritional analysis

3. **Production Readiness**
   - Add monitoring and logging
   - Implement A/B testing
   - Add user authentication

---

**🍎 Diet Recommendation System** - Powered by Machine Learning & Streamlit
