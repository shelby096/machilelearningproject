#!/usr/bin/env python3
"""
Simplified training script for Streamlit deployment.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

def simple_train():
    """Simple training function that works reliably."""
    try:
        print("🔄 Starting simple training... (v2)")
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Load data
        print("📊 Loading dataset...")
        df = pd.read_csv('data/raw/curadiet-g.csv')
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic preprocessing
        print("🧹 Preprocessing data...")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Create simple features
        if 'BMI' in df.columns:
            df['BMI_Category'] = pd.cut(df['BMI'], 
                                       bins=[0, 18.5, 25, 30, float('inf')], 
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], 
                                    bins=[0, 30, 45, 60, float('inf')], 
                                    labels=['Young', 'Middle_Age', 'Senior', 'Elderly'])
        
        # Prepare features and target
        target_column = 'Recommended_Meal_Plan'
        
        # Select basic features
        feature_columns = []
        for col in df.columns:
            if col != target_column and col not in ['Patient_ID']:
                feature_columns.append(col)
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        print(f"Features: {len(feature_columns)}")
        print(f"Target classes: {y.nunique()}")
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # Scale features
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
        print("🤖 Training models...")
        
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        best_model = None
        best_accuracy = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} accuracy: {accuracy:.4f}")
            
            # Save model
            joblib.dump(model, f'models/{name}.joblib')
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
        
        # Save preprocessors
        preprocessors = {
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        
        joblib.dump(preprocessors, 'models/preprocessors.joblib')
        
        print(f"✅ Training completed!")
        print(f"Best model: {best_name} (Accuracy: {best_accuracy:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_train()
    exit(0 if success else 1)
