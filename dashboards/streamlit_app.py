import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.predict_model import DietPredictor
    from utils.logger import get_logger
except ImportError:
    # Fallback for deployment
    pass

# Page configuration
st.set_page_config(
    page_title="Diet Recommendation System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class DietRecommendationApp:
    """Streamlit application for diet recommendations."""
    
    def __init__(self):
        self.predictor = None
        self.model_loaded = False
        self.initialize_session_state()
        self.load_model()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
    
    def load_model(self):
        """Load the trained model and preprocessors."""
        try:
            # Try to load the best model
            model_path = os.path.join("models", "random_forest.joblib")  # Default to random forest
            preprocessor_path = os.path.join("models", "preprocessors.joblib")
            
            if os.path.exists(model_path):
                self.predictor = DietPredictor(model_path, preprocessor_path)
                self.model_loaded = True
            else:
                st.error("Model files not found. Please train the model first.")
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🍎 AI Diet Recommendation System</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Get personalized diet recommendations based on your health profile and lifestyle
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with user inputs."""
        st.sidebar.header("👤 Personal Information")
        
        # Basic Information
        age = st.sidebar.slider("Age", 18, 100, 30)
        gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.sidebar.slider("Height (cm)", 140, 220, 170)
        weight = st.sidebar.slider("Weight (kg)", 40, 200, 70)
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        st.sidebar.metric("BMI", f"{bmi:.1f}")
        
        st.sidebar.header("🏥 Health Information")
        
        # Health metrics
        chronic_disease = st.sidebar.selectbox(
            "Chronic Disease", 
            ["None", "Diabetes", "Hypertension", "Heart Disease", "Obesity"]
        )
        
        blood_pressure_sys = st.sidebar.slider("Blood Pressure (Systolic)", 90, 200, 120)
        blood_pressure_dia = st.sidebar.slider("Blood Pressure (Diastolic)", 60, 120, 80)
        cholesterol = st.sidebar.slider("Cholesterol Level", 150, 350, 200)
        blood_sugar = st.sidebar.slider("Blood Sugar Level", 70, 300, 100)
        
        genetic_risk = st.sidebar.selectbox("Genetic Risk Factor", ["No", "Yes"])
        allergies = st.sidebar.selectbox(
            "Allergies", 
            ["None", "Nut Allergy", "Gluten Intolerance", "Lactose Intolerance"]
        )
        
        st.sidebar.header("🏃 Lifestyle")
        
        # Lifestyle factors
        daily_steps = st.sidebar.slider("Daily Steps", 1000, 20000, 8000)
        exercise_freq = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)
        sleep_hours = st.sidebar.slider("Sleep Hours", 4, 12, 8)
        
        alcohol = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
        smoking = st.sidebar.selectbox("Smoking Habit", ["No", "Yes"])
        
        st.sidebar.header("🍽️ Dietary Preferences")
        
        # Dietary information
        dietary_habits = st.sidebar.selectbox(
            "Dietary Habits", 
            ["Regular", "Vegetarian", "Vegan", "Keto"]
        )
        
        caloric_intake = st.sidebar.slider("Current Caloric Intake", 1200, 4000, 2000)
        protein_intake = st.sidebar.slider("Protein Intake (g)", 50, 300, 100)
        carb_intake = st.sidebar.slider("Carbohydrate Intake (g)", 100, 500, 250)
        fat_intake = st.sidebar.slider("Fat Intake (g)", 30, 200, 80)
        
        preferred_cuisine = st.sidebar.selectbox(
            "Preferred Cuisine", 
            ["Western", "Asian", "Mediterranean", "Indian"]
        )
        
        food_aversions = st.sidebar.selectbox(
            "Food Aversions", 
            ["None", "Spicy", "Sweet", "Salty"]
        )
        
        # Compile user data
        user_data = {
            'Age': age,
            'Gender': gender,
            'Height_cm': height,
            'Weight_kg': weight,
            'BMI': bmi,
            'Chronic_Disease': chronic_disease,
            'Blood_Pressure_Systolic': blood_pressure_sys,
            'Blood_Pressure_Diastolic': blood_pressure_dia,
            'Cholesterol_Level': cholesterol,
            'Blood_Sugar_Level': blood_sugar,
            'Genetic_Risk_Factor': genetic_risk,
            'Allergies': allergies,
            'Daily_Steps': daily_steps,
            'Exercise_Frequency': exercise_freq,
            'Sleep_Hours': sleep_hours,
            'Alcohol_Consumption': alcohol,
            'Smoking_Habit': smoking,
            'Dietary_Habits': dietary_habits,
            'Caloric_Intake': caloric_intake,
            'Protein_Intake': protein_intake,
            'Carbohydrate_Intake': carb_intake,
            'Fat_Intake': fat_intake,
            'Preferred_Cuisine': preferred_cuisine,
            'Food_Aversions': food_aversions
        }
        
        return user_data
    
    def render_health_dashboard(self, user_data):
        """Render health metrics dashboard."""
        st.subheader("📊 Health Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bmi_status = self.get_bmi_status(user_data['BMI'])
            st.metric("BMI Status", bmi_status, f"{user_data['BMI']:.1f}")
        
        with col2:
            bp_status = self.get_bp_status(user_data['Blood_Pressure_Systolic'], user_data['Blood_Pressure_Diastolic'])
            st.metric("Blood Pressure", bp_status, f"{user_data['Blood_Pressure_Systolic']}/{user_data['Blood_Pressure_Diastolic']}")
        
        with col3:
            cholesterol_status = self.get_cholesterol_status(user_data['Cholesterol_Level'])
            st.metric("Cholesterol", cholesterol_status, f"{user_data['Cholesterol_Level']} mg/dL")
        
        with col4:
            activity_level = self.get_activity_level(user_data['Exercise_Frequency'])
            st.metric("Activity Level", activity_level, f"{user_data['Exercise_Frequency']} days/week")
        
        # Health risk visualization
        self.render_health_risk_chart(user_data)
    
    def render_health_risk_chart(self, user_data):
        """Render health risk assessment chart."""
        risk_factors = []
        risk_scores = []
        
        # BMI risk
        bmi_risk = max(0, abs(user_data['BMI'] - 22.5) / 10)
        risk_factors.append('BMI')
        risk_scores.append(min(bmi_risk, 1))
        
        # Blood pressure risk
        bp_risk = max(0, (user_data['Blood_Pressure_Systolic'] - 120) / 60)
        risk_factors.append('Blood Pressure')
        risk_scores.append(min(bp_risk, 1))
        
        # Cholesterol risk
        chol_risk = max(0, (user_data['Cholesterol_Level'] - 200) / 100)
        risk_factors.append('Cholesterol')
        risk_scores.append(min(chol_risk, 1))
        
        # Lifestyle risk
        lifestyle_risk = 0
        if user_data['Smoking_Habit'] == 'Yes':
            lifestyle_risk += 0.3
        if user_data['Alcohol_Consumption'] == 'Yes':
            lifestyle_risk += 0.2
        if user_data['Exercise_Frequency'] < 3:
            lifestyle_risk += 0.3
        risk_factors.append('Lifestyle')
        risk_scores.append(min(lifestyle_risk, 1))
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=risk_scores,
            theta=risk_factors,
            fill='toself',
            name='Risk Level',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Health Risk Assessment"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_bmi_status(self, bmi):
        """Get BMI status category."""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def get_bp_status(self, systolic, diastolic):
        """Get blood pressure status."""
        if systolic < 120 and diastolic < 80:
            return "Normal"
        elif systolic < 130 and diastolic < 80:
            return "Elevated"
        elif systolic < 140 or diastolic < 90:
            return "High Stage 1"
        else:
            return "High Stage 2"
    
    def get_cholesterol_status(self, cholesterol):
        """Get cholesterol status."""
        if cholesterol < 200:
            return "Normal"
        elif cholesterol < 240:
            return "Borderline"
        else:
            return "High"
    
    def get_activity_level(self, exercise_freq):
        """Get activity level."""
        if exercise_freq == 0:
            return "Sedentary"
        elif exercise_freq < 3:
            return "Low"
        elif exercise_freq < 5:
            return "Moderate"
        else:
            return "High"
    
    def make_prediction(self, user_data):
        """Make diet recommendation prediction."""
        if not self.model_loaded:
            st.error("Model not loaded. Cannot make predictions.")
            return None
        
        try:
            # Make prediction using the loaded model
            prediction_result = self.predictor.predict_single(user_data)
            
            # Get detailed recommendations
            nutrition_recommendations = self.predictor.get_nutrition_recommendations(user_data)
            
            # Get explanation
            explanation = self.predictor.explain_prediction(user_data)
            
            return {
                'prediction': prediction_result,
                'nutrition': nutrition_recommendations,
                'explanation': explanation
            }
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def render_prediction_results(self, results, user_data):
        """Render prediction results."""
        if not results:
            return
        
        st.subheader("🎯 Your Personalized Diet Recommendation")
        
        # Main recommendation
        meal_plan = results['prediction']['recommended_meal_plan']
        
        st.markdown(f"""
        <div class="recommendation-box">
            <h2 style="color: #2E8B57; margin-bottom: 1rem;">Recommended: {meal_plan}</h2>
            <p style="font-size: 1.1rem;">Based on your health profile and lifestyle, this diet plan is most suitable for you.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence scores
        if 'confidence_scores' in results['prediction'] and results['prediction']['confidence_scores']:
            st.subheader("📈 Confidence Scores")
            confidence_data = results['prediction']['confidence_scores']
            
            # Create confidence chart
            fig = px.bar(
                x=list(confidence_data.keys()),
                y=list(confidence_data.values()),
                title="Model Confidence for Each Diet Type",
                labels={'x': 'Diet Type', 'y': 'Confidence Score'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Nutrition breakdown
        if 'nutrition' in results:
            self.render_nutrition_recommendations(results['nutrition'])
        
        # Explanation
        if 'explanation' in results:
            self.render_explanation(results['explanation'])
    
    def render_nutrition_recommendations(self, nutrition):
        """Render detailed nutrition recommendations."""
        st.subheader("🥗 Detailed Nutrition Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Macronutrient Breakdown")
            if 'nutrition_breakdown' in nutrition and nutrition['nutrition_breakdown']:
                breakdown = nutrition['nutrition_breakdown']
                
                macros = ['Protein', 'Carbohydrates', 'Fats']
                percentages = [
                    breakdown.get('protein_percentage', '20%'),
                    breakdown.get('carb_percentage', '50%'),
                    breakdown.get('fat_percentage', '30%')
                ]
                
                # Clean percentage values
                clean_percentages = []
                for p in percentages:
                    if isinstance(p, str):
                        # Extract first number from range like "25-30%"
                        import re
                        match = re.search(r'(\d+)', p)
                        if match:
                            clean_percentages.append(int(match.group(1)))
                        else:
                            clean_percentages.append(0)
                    else:
                        clean_percentages.append(p)
                
                fig = px.pie(
                    values=clean_percentages,
                    names=macros,
                    title="Recommended Macronutrient Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display breakdown details
                for macro, percentage in zip(macros, percentages):
                    st.write(f"**{macro}:** {percentage}")
        
        with col2:
            st.markdown("### 🎯 Focus Foods")
            if 'nutrition_breakdown' in nutrition and 'focus' in nutrition['nutrition_breakdown']:
                st.write(f"**Emphasize:** {nutrition['nutrition_breakdown']['focus']}")
            
            if 'nutrition_breakdown' in nutrition and 'avoid' in nutrition['nutrition_breakdown']:
                st.write(f"**Limit:** {nutrition['nutrition_breakdown']['avoid']}")
            
            st.markdown("### 💧 Hydration")
            st.write(nutrition.get('hydration', 'Aim for 8-10 glasses of water daily'))
            
            st.markdown("### ⏰ Meal Timing")
            st.write(nutrition.get('meal_timing', 'Maintain regular meal times'))
        
        # Personalized notes
        if 'personalized_notes' in nutrition and nutrition['personalized_notes']:
            st.markdown("### 📝 Personalized Notes")
            for note in nutrition['personalized_notes']:
                st.write(f"• {note}")
        
        # Supplements
        if 'supplements' in nutrition and nutrition['supplements']:
            st.markdown("### 💊 Recommended Supplements")
            for supplement in nutrition['supplements']:
                st.write(f"• {supplement}")
    
    def render_explanation(self, explanation):
        """Render prediction explanation."""
        st.subheader("🔍 Why This Recommendation?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Reasoning")
            if 'reasoning' in explanation:
                for reason in explanation['reasoning']:
                    st.write(f"• {reason}")
        
        with col2:
            st.markdown("### 🔑 Key Factors")
            if 'key_factors' in explanation:
                for factor in explanation['key_factors']:
                    st.write(f"• {factor}")
    
    def render_history(self):
        """Render prediction history."""
        if st.session_state.prediction_history:
            st.subheader("📚 Prediction History")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Download history
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download History as CSV",
                data=csv,
                file_name=f"diet_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        
        # Main layout
        user_data = self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendation", "📊 Health Dashboard", "📚 History"])
        
        with tab1:
            st.subheader("Get Your Personalized Diet Recommendation")
            
            if st.button("🚀 Generate Recommendation", type="primary", use_container_width=True):
                with st.spinner("Analyzing your profile and generating recommendations..."):
                    results = self.make_prediction(user_data)
                    
                    if results:
                        self.render_prediction_results(results, user_data)
                        
                        # Save to history
                        history_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'recommendation': results['prediction']['recommended_meal_plan'],
                            'bmi': user_data['BMI'],
                            'age': user_data['Age'],
                            'exercise_frequency': user_data['Exercise_Frequency']
                        }
                        st.session_state.prediction_history.append(history_entry)
        
        with tab2:
            self.render_health_dashboard(user_data)
        
        with tab3:
            self.render_history()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🍎 AI Diet Recommendation System | Built with Streamlit & Machine Learning</p>
            <p><small>⚠️ This tool provides general recommendations. Please consult with healthcare professionals for personalized medical advice.</small></p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the app."""
    app = DietRecommendationApp()
    app.run()


if __name__ == "__main__":
    main()
