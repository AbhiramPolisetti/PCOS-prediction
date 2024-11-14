import streamlit as st
import joblib
import numpy as np

# Load the trained Logistic Regression model
log_reg_model = joblib.load("/workspaces/PCOS-prediction/random_forest_balanced_model.pkl")

# Set up the Streamlit app
st.title("PCOS Prediction")
st.write("Enter the details below to check the likelihood of having PCOS.")

# Collect user input with Streamlit widgets
age = st.number_input("Age:", min_value=0.0, step=0.1)
weight = st.number_input("Weight (Kg):", min_value=0.0, step=0.1)
height = st.number_input("Height (Cm):", min_value=0.0, step=0.1)
marriage_years = st.number_input('Years of Marriage(Note: If not married please enter "0"):', min_value=0.0, step=0.1)
cycle_length = st.number_input("Cycle Length (days):", min_value=0.0, step=0.1)
cycle_regular = st.checkbox("Cycle Regularity (Regular)", value=False)
skin_darkening = st.checkbox("Skin Darkening ", value=False)
weight_gain = st.checkbox("Weight Gain", value=False)
hair_growth = st.checkbox("Hair Growth", value=False)

# Convert checkboxes to binary values for the model
skin_darkening = 1 if skin_darkening else 0
weight_gain = 1 if weight_gain else 0
hair_growth = 1 if hair_growth else 0
cycle_regular = 1 if cycle_regular else 0

# Calculate BMI
height_m = height / 100  # Convert height from cm to meters
bmi = weight / (height_m ** 2) if height_m > 0 else 0  # Handle division by zero if height is zero

# Display calculated BMI
st.write(f"BMI: {bmi:.2f}")

# Prepare the data for prediction
user_data = np.array([[skin_darkening, weight_gain, hair_growth, marriage_years, age, 
                       weight, cycle_length, cycle_regular, height, bmi]])

# Button to make the prediction
if st.button("Predict"):
    prediction = log_reg_model.predict(user_data)

    # Interpret prediction
    if prediction[0] == 1:
        st.success("The model predicts that the user might have PCOS.")
    else:
        st.success("The model predicts that the user does not have PCOS.")
