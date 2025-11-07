import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

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
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_preprocessors():
    """Load the trained model and preprocessors."""
    try:
        # Check if models exist, if not train them
        if not os.path.exists('models/logistic_regression.joblib'):
            st.info("🔄 Models not found. Training models now... This may take a few minutes.")
            
            # Run the training pipeline
            import subprocess
            import sys
            
            with st.spinner("Training ML models..."):
                result = subprocess.run([sys.executable, 'run_pipeline.py'], 
                                      capture_output=True, text=True)
                
                if result.returncode != 0:
                    st.error(f"Training failed: {result.stderr}")
                    return None, None
                
                st.success("✅ Models trained successfully!")
        
        # Load the trained models
        model = joblib.load('models/logistic_regression.joblib')
        preprocessors = joblib.load('models/preprocessors.joblib')
        return model, preprocessors
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_input(input_data, preprocessors):
    """Preprocess user input for prediction."""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Get feature columns used during training
        feature_columns = preprocessors['feature_columns']
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in df.columns:
                # Set default values for missing engineered features
                if col == 'BMI_Category':
                    bmi = input_data.get('BMI', 25)
                    if bmi < 18.5:
                        df[col] = 'Underweight'
                    elif bmi < 25:
                        df[col] = 'Normal'
                    elif bmi < 30:
                        df[col] = 'Overweight'
                    else:
                        df[col] = 'Obese'
                elif col == 'Age_Group':
                    age = input_data.get('Age', 30)
                    if age < 30:
                        df[col] = 'Young'
                    elif age < 45:
                        df[col] = 'Middle_Age'
                    elif age < 60:
                        df[col] = 'Senior'
                    else:
                        df[col] = 'Elderly'
                elif col == 'Health_Risk_Score':
                    risk = 0
                    if input_data.get('Chronic_Disease', 'None') != 'None':
                        risk += 1
                    if input_data.get('Smoking_Habit', 'No') == 'Yes':
                        risk += 1
                    if input_data.get('Genetic_Risk_Factor', 'No') == 'Yes':
                        risk += 1
                    df[col] = risk
                else:
                    df[col] = 0  # Default value
        
        # Select only the feature columns used during training
        df = df[feature_columns]
        
        # Encode categorical variables
        label_encoders = preprocessors['label_encoders']
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Scale numeric features
        scaler = preprocessors['scaler']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        return df
        
    except Exception as e:
        st.error(f"Error preprocessing input: {str(e)}")
        return None

def make_prediction(input_data, model, preprocessors):
    """Make diet recommendation prediction."""
    try:
        # Preprocess input
        processed_data = preprocess_input(input_data, preprocessors)
        
        if processed_data is None:
            return None
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(processed_data)[0]
        
        # Decode prediction
        target_encoder = preprocessors['target_encoder']
        predicted_diet = target_encoder.inverse_transform([prediction])[0]
        
        # Create confidence scores
        confidence_scores = {}
        for i, class_name in enumerate(target_encoder.classes_):
            confidence_scores[class_name] = probabilities[i]
        
        return {
            'prediction': predicted_diet,
            'confidence_scores': confidence_scores
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def render_sidebar():
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

def render_health_dashboard(user_data):
    """Render health metrics dashboard."""
    st.subheader("📊 Health Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bmi_status = get_bmi_status(user_data['BMI'])
        st.metric("BMI Status", bmi_status, f"{user_data['BMI']:.1f}")
    
    with col2:
        bp_status = get_bp_status(user_data['Blood_Pressure_Systolic'], user_data['Blood_Pressure_Diastolic'])
        st.metric("Blood Pressure", bp_status, f"{user_data['Blood_Pressure_Systolic']}/{user_data['Blood_Pressure_Diastolic']}")
    
    with col3:
        cholesterol_status = get_cholesterol_status(user_data['Cholesterol_Level'])
        st.metric("Cholesterol", cholesterol_status, f"{user_data['Cholesterol_Level']} mg/dL")
    
    with col4:
        activity_level = get_activity_level(user_data['Exercise_Frequency'])
        st.metric("Activity Level", activity_level, f"{user_data['Exercise_Frequency']} days/week")

def get_bmi_status(bmi):
    """Get BMI status category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_bp_status(systolic, diastolic):
    """Get blood pressure status."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "High Stage 1"
    else:
        return "High Stage 2"

def get_cholesterol_status(cholesterol):
    """Get cholesterol status."""
    if cholesterol < 200:
        return "Normal"
    elif cholesterol < 240:
        return "Borderline"
    else:
        return "High"

def get_activity_level(exercise_freq):
    """Get activity level."""
    if exercise_freq == 0:
        return "Sedentary"
    elif exercise_freq < 3:
        return "Low"
    elif exercise_freq < 5:
        return "Moderate"
    else:
        return "High"

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🍎 AI Diet Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Get personalized diet recommendations based on your health profile and lifestyle
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, preprocessors = load_model_and_preprocessors()
    
    if model is None or preprocessors is None:
        st.error("❌ Could not load the trained model. Please ensure the model files exist.")
        return
    
    # Sidebar inputs
    user_data = render_sidebar()
    
    # Main content
    tab1, tab2 = st.tabs(["🎯 Get Recommendation", "📊 Health Dashboard"])
    
    with tab1:
        st.subheader("Get Your Personalized Diet Recommendation")
        
        if st.button("🚀 Generate Recommendation", type="primary", use_container_width=True):
            with st.spinner("Analyzing your profile and generating recommendations..."):
                result = make_prediction(user_data, model, preprocessors)
                
                if result:
                    # Main recommendation
                    meal_plan = result['prediction']
                    
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h2 style="color: #2E8B57; margin-bottom: 1rem;">Recommended: {meal_plan}</h2>
                        <p style="font-size: 1.1rem;">Based on your health profile and lifestyle, this diet plan is most suitable for you.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence scores
                    st.subheader("📈 Confidence Scores")
                    confidence_data = result['confidence_scores']
                    
                    # Create confidence chart
                    fig = px.bar(
                        x=list(confidence_data.keys()),
                        y=list(confidence_data.values()),
                        title="Model Confidence for Each Diet Type",
                        labels={'x': 'Diet Type', 'y': 'Confidence Score'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Diet plan information
                    st.subheader("📋 Diet Plan Information")
                    
                    diet_info = {
                        'Balanced Diet': {
                            'description': 'A well-rounded approach including all food groups in moderation',
                            'focus': 'Variety, portion control, and nutrient balance',
                            'benefits': 'Sustainable, nutritionally complete, easy to follow'
                        },
                        'High-Protein Diet': {
                            'description': 'Emphasizes protein-rich foods to support muscle growth and satiety',
                            'focus': 'Lean meats, fish, eggs, legumes, and dairy products',
                            'benefits': 'Muscle building, weight management, increased satiety'
                        },
                        'Low-Carb Diet': {
                            'description': 'Restricts carbohydrates to promote fat burning and weight loss',
                            'focus': 'Healthy fats, moderate protein, minimal carbohydrates',
                            'benefits': 'Weight loss, blood sugar control, reduced cravings'
                        },
                        'Low-Fat Diet': {
                            'description': 'Limits fat intake while emphasizing fruits, vegetables, and whole grains',
                            'focus': 'Fruits, vegetables, whole grains, lean proteins',
                            'benefits': 'Heart health, weight management, cholesterol control'
                        }
                    }
                    
                    if meal_plan in diet_info:
                        info = diet_info[meal_plan]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Description:** {info['description']}")
                            st.write(f"**Focus:** {info['focus']}")
                        
                        with col2:
                            st.write(f"**Benefits:** {info['benefits']}")
    
    with tab2:
        render_health_dashboard(user_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🍎 AI Diet Recommendation System | Built with Streamlit & Machine Learning</p>
        <p><small>⚠️ This tool provides general recommendations. Please consult with healthcare professionals for personalized medical advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
