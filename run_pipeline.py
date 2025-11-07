#!/usr/bin/env python3
"""
Main script to run the complete diet recommendation ML pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw', 'data/interim', 'data/processed', 'data/external',
        'models', 'reports', 'reports/figures', 'logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Directories created successfully")

def run_etl_pipeline():
    """Run the ETL pipeline."""
    print("\n🔄 Starting ETL Pipeline...")
    
    try:
        # Import after path setup
        from utils.logger import get_logger
        from utils.config_loader import ConfigLoader
        
        logger = get_logger("etl_pipeline")
        config_loader = ConfigLoader()
        
        # Load and process data
        print("📊 Loading dataset...")
        df = pd.read_csv('data/raw/curadiet-g.csv')
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic data cleaning
        print("🧹 Cleaning data...")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Feature engineering
        print("⚙️ Engineering features...")
        
        # BMI categories
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 30, 45, 60, float('inf')], 
                                labels=['Young', 'Middle_Age', 'Senior', 'Elderly'])
        
        # Health risk score
        risk_score = 0
        if 'Chronic_Disease' in df.columns:
            risk_score += (df['Chronic_Disease'] != 'None').astype(int)
        if 'Smoking_Habit' in df.columns:
            risk_score += (df['Smoking_Habit'] == 'Yes').astype(int)
        if 'Genetic_Risk_Factor' in df.columns:
            risk_score += (df['Genetic_Risk_Factor'] == 'Yes').astype(int)
        
        df['Health_Risk_Score'] = risk_score
        
        # Save processed data
        print("💾 Saving processed data...")
        df.to_csv('data/processed/processed_diet_data.csv', index=False)
        
        print(f"✅ ETL Pipeline completed successfully!")
        print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        print(f"❌ ETL Pipeline failed: {str(e)}")
        raise

def run_ml_pipeline(df):
    """Run the ML pipeline."""
    print("\n🤖 Starting ML Pipeline...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import joblib
        
        # Prepare data for modeling
        print("📋 Preparing data for modeling...")
        
        target_column = 'Recommended_Meal_Plan'
        
        # Select features (exclude target and ID columns)
        feature_columns = [col for col in df.columns if col not in [
            target_column, 'Patient_ID', 'Recommended_Calories', 
            'Recommended_Protein', 'Recommended_Carbs', 'Recommended_Fats'
        ]]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # Scale numeric features
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🏋️ Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}")
            
            # Save model
            model_path = f'models/{name.lower().replace(" ", "_")}.joblib'
            joblib.dump(model, model_path)
            print(f"💾 Model saved: {model_path}")
        
        # Save preprocessors
        preprocessors = {
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        joblib.dump(preprocessors, 'models/preprocessors.joblib')
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Generate classification report for best model
        best_predictions = results[best_model_name]['predictions']
        class_names = target_encoder.classes_
        
        print("\n📊 Classification Report (Best Model):")
        print(classification_report(y_test, best_predictions, target_names=class_names))
        
        # Save results summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'models_trained': list(results.keys()),
            'dataset_shape': df.shape,
            'features_used': len(feature_columns)
        }
        
        import json
        with open('reports/ml_pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ ML Pipeline completed successfully!")
        
        return results, preprocessors
        
    except Exception as e:
        print(f"❌ ML Pipeline failed: {str(e)}")
        raise

def main():
    """Main function to run the complete pipeline."""
    print("🍎 Diet Recommendation ML Pipeline")
    print("=" * 50)
    
    try:
        # Setup
        setup_directories()
        
        # Run ETL Pipeline
        df = run_etl_pipeline()
        
        # Run ML Pipeline
        results, preprocessors = run_ml_pipeline(df)
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        print("📁 Generated files:")
        print("  - data/processed/processed_diet_data.csv")
        print("  - models/*.joblib (trained models)")
        print("  - reports/ml_pipeline_summary.json")
        print("\n🚀 Ready for deployment!")
        
    except Exception as e:
        print(f"\n💥 Pipeline failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
