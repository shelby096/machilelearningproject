import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List, Union, Tuple
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DietPredictor:
    """Class for making diet recommendations using trained models."""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        self.model = None
        self.preprocessors = None
        self.model_loaded = False
        self.config_loader = ConfigLoader()
        
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessors(preprocessor_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_preprocessors(self, preprocessor_path: str) -> None:
        """
        Load preprocessors (scalers, encoders, etc.) from file.
        
        Args:
            preprocessor_path: Path to the saved preprocessors
        """
        try:
            if not os.path.exists(preprocessor_path):
                logger.warning(f"Preprocessor file not found: {preprocessor_path}")
                return
            
            self.preprocessors = joblib.load(preprocessor_path)
            logger.info(f"Preprocessors loaded successfully from {preprocessor_path}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Raw input data
        
        Returns:
            Preprocessed data ready for prediction
        """
        try:
            processed_data = input_data.copy()
            
            if self.preprocessors:
                # Apply saved preprocessors
                if 'scalers' in self.preprocessors:
                    for scaler_name, scaler in self.preprocessors['scalers'].items():
                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            processed_data[numeric_cols] = scaler.transform(processed_data[numeric_cols])
                
                if 'encoders' in self.preprocessors:
                    for col_name, encoder in self.preprocessors['encoders'].items():
                        if col_name in processed_data.columns:
                            processed_data[col_name] = encoder.transform(processed_data[col_name].astype(str))
            
            logger.info("Input data preprocessed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing input data: {str(e)}")
            raise
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single input.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Dictionary containing prediction results
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Convert input to DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Preprocess if preprocessors are available
            if self.preprocessors:
                df_processed = self.preprocess_input(df_input)
            else:
                df_processed = df_input
            
            # Make prediction
            prediction = self.model.predict(df_processed)[0]
            
            # Get prediction probabilities if available
            prediction_proba = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df_processed)[0]
                classes = self.model.classes_ if hasattr(self.model, 'classes_') else None
                if classes is not None:
                    prediction_proba = dict(zip(classes, proba))
            
            result = {
                'recommended_meal_plan': prediction,
                'confidence_scores': prediction_proba,
                'input_data': input_data
            }
            
            logger.info(f"Prediction made: {prediction}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple inputs.
        
        Args:
            input_data: DataFrame containing input features
        
        Returns:
            DataFrame with predictions added
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please load a model first.")
        
        try:
            # Preprocess if preprocessors are available
            if self.preprocessors:
                df_processed = self.preprocess_input(input_data)
            else:
                df_processed = input_data.copy()
            
            # Make predictions
            predictions = self.model.predict(df_processed)
            
            # Add predictions to original data
            result_df = input_data.copy()
            result_df['predicted_meal_plan'] = predictions
            
            # Add prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df_processed)
                classes = self.model.classes_ if hasattr(self.model, 'classes_') else None
                
                if classes is not None:
                    for i, class_name in enumerate(classes):
                        result_df[f'confidence_{class_name}'] = proba[:, i]
            
            logger.info(f"Batch predictions made for {len(input_data)} samples")
            return result_df
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the loaded model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please load a model first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature importance information")
            return {}
        
        try:
            # This would need feature names from training
            # For now, return generic feature importance
            importance_scores = self.model.feature_importances_
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
            
            importance_dict = dict(zip(feature_names, importance_scores))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide explanation for a prediction.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Dictionary containing prediction explanation
        """
        try:
            # Get basic prediction
            prediction_result = self.predict_single(input_data)
            
            # Add explanation based on input features
            explanation = {
                'prediction': prediction_result['recommended_meal_plan'],
                'reasoning': self._generate_reasoning(input_data, prediction_result['recommended_meal_plan']),
                'confidence_scores': prediction_result.get('confidence_scores', {}),
                'key_factors': self._identify_key_factors(input_data)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            raise
    
    def _generate_reasoning(self, input_data: Dict[str, Any], prediction: str) -> List[str]:
        """
        Generate human-readable reasoning for the prediction.
        
        Args:
            input_data: Input features
            prediction: Predicted meal plan
        
        Returns:
            List of reasoning statements
        """
        reasoning = []
        
        # BMI-based reasoning
        if 'BMI' in input_data:
            bmi = input_data['BMI']
            if bmi < 18.5:
                reasoning.append("Low BMI indicates need for calorie-dense nutrition")
            elif bmi > 30:
                reasoning.append("High BMI suggests need for calorie-controlled diet")
        
        # Health condition reasoning
        if 'Chronic_Disease' in input_data and input_data['Chronic_Disease'] != 'None':
            disease = input_data['Chronic_Disease']
            reasoning.append(f"Chronic condition ({disease}) requires specialized dietary approach")
        
        # Activity level reasoning
        if 'Exercise_Frequency' in input_data:
            exercise = input_data['Exercise_Frequency']
            if exercise >= 5:
                reasoning.append("High activity level supports higher protein requirements")
            elif exercise <= 1:
                reasoning.append("Low activity level suggests moderate calorie intake")
        
        # Age-based reasoning
        if 'Age' in input_data:
            age = input_data['Age']
            if age >= 65:
                reasoning.append("Senior age group requires nutrient-dense, easily digestible foods")
            elif age <= 30:
                reasoning.append("Young age group can handle varied dietary approaches")
        
        # Dietary preference reasoning
        if 'Dietary_Habits' in input_data:
            diet_type = input_data['Dietary_Habits']
            if diet_type == 'Vegetarian':
                reasoning.append("Vegetarian preference requires plant-based protein focus")
            elif diet_type == 'Keto':
                reasoning.append("Ketogenic preference indicates low-carb, high-fat approach")
        
        return reasoning
    
    def _identify_key_factors(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Identify key factors that influenced the prediction.
        
        Args:
            input_data: Input features
        
        Returns:
            List of key factors
        """
        key_factors = []
        
        # High-impact factors based on common diet recommendation logic
        high_impact_features = [
            'BMI', 'Chronic_Disease', 'Dietary_Habits', 'Exercise_Frequency',
            'Age', 'Allergies', 'Blood_Sugar_Level', 'Cholesterol_Level'
        ]
        
        for feature in high_impact_features:
            if feature in input_data:
                key_factors.append(f"{feature}: {input_data[feature]}")
        
        return key_factors[:5]  # Return top 5 key factors
    
    def get_nutrition_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed nutrition recommendations based on prediction.
        
        Args:
            input_data: Dictionary containing input features
        
        Returns:
            Dictionary containing detailed nutrition recommendations
        """
        try:
            prediction_result = self.predict_single(input_data)
            meal_plan = prediction_result['recommended_meal_plan']
            
            # Basic nutrition guidelines based on meal plan type
            nutrition_guidelines = {
                'High-Protein Diet': {
                    'protein_percentage': '25-30%',
                    'carb_percentage': '40-45%',
                    'fat_percentage': '25-30%',
                    'focus': 'Lean meats, fish, eggs, legumes, dairy',
                    'avoid': 'Processed foods, excessive sugars'
                },
                'Low-Carb Diet': {
                    'protein_percentage': '20-25%',
                    'carb_percentage': '20-25%',
                    'fat_percentage': '50-60%',
                    'focus': 'Healthy fats, moderate protein, low-carb vegetables',
                    'avoid': 'Grains, sugars, starchy vegetables'
                },
                'Balanced Diet': {
                    'protein_percentage': '15-20%',
                    'carb_percentage': '45-55%',
                    'fat_percentage': '25-35%',
                    'focus': 'Variety of all food groups in moderation',
                    'avoid': 'Excessive processed foods'
                },
                'Low-Fat Diet': {
                    'protein_percentage': '15-20%',
                    'carb_percentage': '55-65%',
                    'fat_percentage': '15-25%',
                    'focus': 'Fruits, vegetables, whole grains, lean proteins',
                    'avoid': 'High-fat foods, fried foods'
                }
            }
            
            recommendations = {
                'meal_plan': meal_plan,
                'nutrition_breakdown': nutrition_guidelines.get(meal_plan, {}),
                'personalized_notes': self._get_personalized_notes(input_data),
                'meal_timing': self._get_meal_timing_advice(input_data),
                'hydration': 'Aim for 8-10 glasses of water daily',
                'supplements': self._get_supplement_recommendations(input_data)
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating nutrition recommendations: {str(e)}")
            raise
    
    def _get_personalized_notes(self, input_data: Dict[str, Any]) -> List[str]:
        """Generate personalized dietary notes."""
        notes = []
        
        if 'Allergies' in input_data and input_data['Allergies'] != 'None':
            notes.append(f"Avoid {input_data['Allergies']} due to allergies")
        
        if 'Exercise_Frequency' in input_data and input_data['Exercise_Frequency'] >= 5:
            notes.append("Increase protein intake to support high activity level")
        
        if 'Age' in input_data and input_data['Age'] >= 65:
            notes.append("Focus on calcium and vitamin D for bone health")
        
        return notes
    
    def _get_meal_timing_advice(self, input_data: Dict[str, Any]) -> str:
        """Generate meal timing advice."""
        if 'Exercise_Frequency' in input_data and input_data['Exercise_Frequency'] >= 4:
            return "Eat protein-rich snack within 30 minutes post-workout"
        else:
            return "Maintain regular meal times with 3-4 hours between meals"
    
    def _get_supplement_recommendations(self, input_data: Dict[str, Any]) -> List[str]:
        """Generate supplement recommendations."""
        supplements = []
        
        if 'Dietary_Habits' in input_data:
            if input_data['Dietary_Habits'] == 'Vegan':
                supplements.extend(['Vitamin B12', 'Iron', 'Omega-3'])
            elif input_data['Dietary_Habits'] == 'Vegetarian':
                supplements.extend(['Vitamin B12', 'Iron'])
        
        if 'Age' in input_data and input_data['Age'] >= 50:
            supplements.extend(['Vitamin D', 'Calcium'])
        
        return list(set(supplements))  # Remove duplicates
