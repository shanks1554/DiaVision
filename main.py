import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

# Function to load LabelEncoders correctly
def load_label_encoder(file_path):
    obj = joblib.load(file_path)
    if isinstance(obj, LabelEncoder):  
        return obj
    elif isinstance(obj, np.ndarray):  
        le = LabelEncoder()
        le.classes_ = obj  
        return le
    else:
        raise ValueError(f"Unknown encoder format for {file_path}. Expected LabelEncoder or numpy.ndarray.")

# Function to load MinMaxScalers
def load_scaler(file_path):
    return joblib.load(file_path)

# Load Encoders & Scalers
try:
    gender_encoder = load_label_encoder("models/le_gender.pkl")
    diet_encoder = load_label_encoder("models/le_diet.pkl")
    food_encoder = load_label_encoder("models/le_food.pkl")
    meal_time_encoder = load_label_encoder("models/le_meal_timing.pkl")
    prakriti_encoder = load_label_encoder("models/le_prakriti.pkl")
    sleep_quality_encoder = load_label_encoder("models/le_sleep_quality.pkl")

    age_scaler = load_scaler("models/mm_age.pkl")
    sleep_hours_scaler = load_scaler("models/mm_sleep_hours.pkl")
    stress_level_scaler = load_scaler("models/mm_stress_level.pkl")
except Exception as e:
    st.error(f"Error loading encoders: {e}")

# Load Encoded Training Data & Train Model
try:
    encoded_data = pd.read_csv("data/encoded_data.csv")
    X_train = encoded_data.drop(columns=["Diabetes Diagnosis"])
    y_train = encoded_data["Diabetes Diagnosis"]

    model = GradientBoostingClassifier(
        learning_rate=0.2, 
        max_depth=3, 
        min_samples_split=5, 
        n_estimators=100
    )
    model.fit(X_train, y_train)
except Exception as e:
    st.error(f"Error training the model: {e}")

# ---- Navigation System ----
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_prediction():
    st.session_state.page = "prediction"

def go_home():
    st.session_state.page = "home"

# ---- Home Page ----
if st.session_state.page == "home":
    st.title("Welcome to the Diabetes Prediction System")
    st.write("Click the button below to start your prediction.")
    if st.button("Go to Prediction System"):
        go_to_prediction()

# ---- Prediction Page ----
elif st.session_state.page == "prediction":
    st.title("Diabetes Prediction System")

    if st.button("⬅️ Back to Home"):
        go_home()

    # User Inputs
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    prakriti = st.selectbox("Prakriti", prakriti_encoder.classes_)
    diet = st.selectbox("Diet", diet_encoder.classes_)

    # Food Preferences (Multiple Selection)
    food_options = ["cooling", "dry", "heavy", "light", "oily", "salty", "sour", "spicy", "sweet", "warming"]
    selected_foods = st.multiselect("Food Preferences", food_options)
    food_combined = "_".join(sorted(selected_foods)) if selected_foods else "none"

    # Encode food preference
    try:
        food_encoded = food_encoder.transform([food_combined])[0]
    except ValueError:
        food_encoded = 0  

    meal_timing = st.selectbox("Meal Timing", meal_time_encoder.classes_)
    exercise = st.radio("Exercise (Yes/No)", ["Yes", "No"])
    sleep_hours = st.slider("Sleep Hours/Night", 3, 12, 7)
    sleep_quality = st.selectbox("Sleep Quality", sleep_quality_encoder.classes_)
    daily_routine = st.radio("Daily Routine (Regular/Irregular)", ["Regular", "Irregular"])
    stress_level = st.slider("Stress Level (0-5)", 0, 5, 3)
    family_history = st.radio("Family History of Diabetes", ["Yes", "No"])

    # Encode Inputs
    try:
        input_data = np.array([
            age_scaler.transform([[age]])[0][0],  
            gender_encoder.transform([gender])[0],
            prakriti_encoder.transform([prakriti])[0],
            diet_encoder.transform([diet])[0],
            food_encoded,  
            meal_time_encoder.transform([meal_timing])[0],
            1 if exercise == "Yes" else 0,
            sleep_hours_scaler.transform([[sleep_hours]])[0][0],  
            sleep_quality_encoder.transform([sleep_quality])[0],
            1 if daily_routine == "Regular" else 0,
            stress_level_scaler.transform([[stress_level]])[0][0],  
            1 if family_history == "Yes" else 0
        ]).reshape(1, -1)
    except Exception as e:
        st.error(f"Error encoding input: {e}")

    # Make Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes ({prediction_proba:.2f}% confidence)")
            else:
                st.success(f"✅ Low Risk of Diabetes ({100 - prediction_proba:.2f}% confidence)")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
