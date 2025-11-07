import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Dict, Tuple, Any
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class FeatureBuilder:
    """Class for building and engineering features for diet recommendation."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.data_config = self.config_loader.get_data_config()
        self.feature_importance = {}
        self.selected_features = []
    
    def create_health_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive health metrics from basic measurements.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional health metrics
        """
        logger.info("Creating health metrics...")
        df_features = df.copy()
        
        # BMI Categories (more detailed)
        if 'BMI' in df_features.columns:
            df_features['BMI_Category_Detailed'] = pd.cut(
                df_features['BMI'],
                bins=[0, 16, 17, 18.5, 25, 30, 35, 40, float('inf')],
                labels=['Severe_Underweight', 'Moderate_Underweight', 'Mild_Underweight', 
                       'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']
            )
        
        # Blood Pressure Categories
        if 'Blood_Pressure_Systolic' in df_features.columns and 'Blood_Pressure_Diastolic' in df_features.columns:
            df_features['BP_Category'] = df_features.apply(self._categorize_blood_pressure, axis=1)
            df_features['BP_Risk_Score'] = (
                (df_features['Blood_Pressure_Systolic'] - 120) / 20 +
                (df_features['Blood_Pressure_Diastolic'] - 80) / 10
            ).clip(0, 10)
        
        # Cholesterol Risk
        if 'Cholesterol_Level' in df_features.columns:
            df_features['Cholesterol_Risk'] = pd.cut(
                df_features['Cholesterol_Level'],
                bins=[0, 200, 240, float('inf')],
                labels=['Normal', 'Borderline', 'High']
            )
        
        # Blood Sugar Categories
        if 'Blood_Sugar_Level' in df_features.columns:
            df_features['Blood_Sugar_Category'] = pd.cut(
                df_features['Blood_Sugar_Level'],
                bins=[0, 100, 126, float('inf')],
                labels=['Normal', 'Prediabetes', 'Diabetes']
            )
        
        logger.info("Health metrics created successfully")
        return df_features
    
    def create_lifestyle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lifestyle-based features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with lifestyle features
        """
        logger.info("Creating lifestyle features...")
        df_features = df.copy()
        
        # Activity Level Score
        if 'Daily_Steps' in df_features.columns and 'Exercise_Frequency' in df_features.columns:
            df_features['Activity_Score'] = (
                (df_features['Daily_Steps'] / 10000) * 0.6 +
                (df_features['Exercise_Frequency'] / 7) * 0.4
            ).clip(0, 2)
        
        # Sleep Quality Score
        if 'Sleep_Hours' in df_features.columns:
            df_features['Sleep_Quality'] = df_features['Sleep_Hours'].apply(self._categorize_sleep)
        
        # Lifestyle Risk Factors
        risk_columns = ['Alcohol_Consumption', 'Smoking_Habit']
        risk_score = 0
        
        for col in risk_columns:
            if col in df_features.columns:
                risk_score += (df_features[col] == 'Yes').astype(int)
        
        df_features['Lifestyle_Risk_Score'] = risk_score
        
        # Age-Activity Interaction
        if 'Age' in df_features.columns and 'Exercise_Frequency' in df_features.columns:
            df_features['Age_Activity_Ratio'] = df_features['Age'] / (df_features['Exercise_Frequency'] + 1)
        
        logger.info("Lifestyle features created successfully")
        return df_features
    
    def create_dietary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dietary and nutritional features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with dietary features
        """
        logger.info("Creating dietary features...")
        df_features = df.copy()
        
        # Macronutrient Ratios
        if all(col in df_features.columns for col in ['Protein_Intake', 'Carbohydrate_Intake', 'Fat_Intake']):
            total_macros = df_features['Protein_Intake'] + df_features['Carbohydrate_Intake'] + df_features['Fat_Intake']
            
            df_features['Protein_Ratio'] = df_features['Protein_Intake'] / (total_macros + 1e-8)
            df_features['Carb_Ratio'] = df_features['Carbohydrate_Intake'] / (total_macros + 1e-8)
            df_features['Fat_Ratio'] = df_features['Fat_Intake'] / (total_macros + 1e-8)
        
        # Caloric Density
        if 'Caloric_Intake' in df_features.columns and 'Weight_kg' in df_features.columns:
            df_features['Calories_Per_Kg'] = df_features['Caloric_Intake'] / df_features['Weight_kg']
        
        # Dietary Pattern Score
        dietary_patterns = {
            'Vegetarian': 1,
            'Vegan': 2,
            'Keto': 3,
            'Regular': 0
        }
        
        if 'Dietary_Habits' in df_features.columns:
            df_features['Dietary_Pattern_Score'] = df_features['Dietary_Habits'].map(dietary_patterns).fillna(0)
        
        # Cuisine Diversity (if multiple cuisines are preferred)
        if 'Preferred_Cuisine' in df_features.columns:
            cuisine_diversity = {
                'Western': 1,
                'Asian': 2,
                'Mediterranean': 3,
                'Indian': 2
            }
            df_features['Cuisine_Diversity_Score'] = df_features['Preferred_Cuisine'].map(cuisine_diversity).fillna(1)
        
        logger.info("Dietary features created successfully")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different domains.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        df_features = df.copy()
        
        # BMI-Age Interaction
        if 'BMI' in df_features.columns and 'Age' in df_features.columns:
            df_features['BMI_Age_Interaction'] = df_features['BMI'] * np.log(df_features['Age'] + 1)
        
        # Health Risk Composite Score
        health_risk_factors = []
        
        if 'Chronic_Disease' in df_features.columns:
            health_risk_factors.append((df_features['Chronic_Disease'] != 'None').astype(int) * 3)
        
        if 'Genetic_Risk_Factor' in df_features.columns:
            health_risk_factors.append((df_features['Genetic_Risk_Factor'] == 'Yes').astype(int) * 2)
        
        if 'Smoking_Habit' in df_features.columns:
            health_risk_factors.append((df_features['Smoking_Habit'] == 'Yes').astype(int) * 2)
        
        if 'Alcohol_Consumption' in df_features.columns:
            health_risk_factors.append((df_features['Alcohol_Consumption'] == 'Yes').astype(int) * 1)
        
        if health_risk_factors:
            df_features['Composite_Health_Risk'] = sum(health_risk_factors)
        
        # Activity-Health Interaction
        if 'Exercise_Frequency' in df_features.columns and 'Composite_Health_Risk' in df_features.columns:
            df_features['Activity_Health_Balance'] = df_features['Exercise_Frequency'] / (df_features['Composite_Health_Risk'] + 1)
        
        logger.info("Interaction features created successfully")
        return df_features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 20) -> List[str]:
        """
        Select top k features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method ('mutual_info' or 'f_classif')
            k: Number of features to select
        
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using {method}...")
        
        # Prepare data for feature selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_numeric.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X_numeric.shape[1]))
        
        selector.fit(X_numeric, y)
        
        # Get selected features
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        # Store feature importance scores
        feature_scores = dict(zip(X_numeric.columns, selector.scores_))
        self.feature_importance = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Selected {len(selected_features)} features")
        self.selected_features = selected_features
        
        return selected_features
    
    def _categorize_blood_pressure(self, row) -> str:
        """Categorize blood pressure based on systolic and diastolic values."""
        systolic = row['Blood_Pressure_Systolic']
        diastolic = row['Blood_Pressure_Diastolic']
        
        if systolic < 120 and diastolic < 80:
            return 'Normal'
        elif systolic < 130 and diastolic < 80:
            return 'Elevated'
        elif systolic < 140 or diastolic < 90:
            return 'Stage_1_Hypertension'
        elif systolic < 180 or diastolic < 120:
            return 'Stage_2_Hypertension'
        else:
            return 'Hypertensive_Crisis'
    
    def _categorize_sleep(self, hours: float) -> str:
        """Categorize sleep quality based on hours."""
        if hours < 6:
            return 'Poor'
        elif hours < 7:
            return 'Fair'
        elif hours <= 9:
            return 'Good'
        else:
            return 'Excessive'
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all engineered features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Building all engineered features...")
        
        # Start with original dataframe
        df_engineered = df.copy()
        
        # Create different types of features
        df_engineered = self.create_health_metrics(df_engineered)
        df_engineered = self.create_lifestyle_features(df_engineered)
        df_engineered = self.create_dietary_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        logger.info(f"Feature engineering completed. Shape: {df_engineered.shape}")
        return df_engineered
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Get feature importance report.
        
        Returns:
            Dictionary containing feature importance information
        """
        return {
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'total_features_created': len(self.feature_importance),
            'features_selected': len(self.selected_features)
        }
