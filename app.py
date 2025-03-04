import streamlit as st
import joblib
import numpy as np

# Path to the pre-trained model
MODEL_PATH = "svm_model.pkl"

# Load the pre-trained SVM model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function for making predictions
def make_prediction(model, input_data):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit Application UI
st.set_page_config(page_title="SVM Prediction App", layout="wide", page_icon="🔍")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #2c3e50;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .description {
        text-align: center;
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .form-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    .result-card {
        background: #ecf0f1;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.markdown('<div class="main-title">SVM Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Enter your details below to find out if a purchase is likely.</div>',
    unsafe_allow_html=True
)

# Load the model
svm_model = load_model()

if svm_model:
    st.divider()
    col_space, col_form, col_space_2 = st.columns([1, 2, 1])

    with col_form:
        with st.form("prediction_form"):
            st.markdown('<div class="form-card">', unsafe_allow_html=True)

            st.markdown('<h3 style="text-align: center; color: #2c3e50;">Input Features</h3>', unsafe_allow_html=True)

            # Input fields
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)

            with col2:
                age = st.slider("Age", min_value=0, max_value=100, value=25, step=1)

            with col3:
                salary = st.number_input("Estimated Salary", min_value=0, max_value=500000, value=50000, step=1000)

            # Submit button
            st.markdown('</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict")

            if submitted:
                # Convert Gender to numerical values
                gender_encoded = 1 if gender == "Male" else 0
                user_input = [gender_encoded, age, salary]

                # Make prediction
                prediction = make_prediction(svm_model, user_input)

                if prediction is not None:
                    result = "**Purchased**" if prediction == 1 else "**Not Purchased**"

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"Prediction: {result}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.balloons()
else:
    st.error("Model could not be loaded. Please check the logs.")
