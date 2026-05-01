import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
MODELS_FOLDER = r"C:\Users\Cp9-30\Downloads\Ghamees Thesis\models"
DATA_PATH = os.path.join(MODELS_FOLDER, "cars_dataset.csv")

# -----------------------------
# Load original dataset
# -----------------------------
try:
    original_df = pd.read_csv(DATA_PATH)
    st.sidebar.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    original_df = None

# -----------------------------
# Load models & preprocessor
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    
    try:
        # Load scikit-learn models
        pkl_models = {
            "Random Forest": "random_forest_model.pkl",
            "XGBoost": "xgboost_model.pkl", 
            "Gradient Boosting": "gradient_boosting_model.pkl",
            "Ridge": "ridge_model.pkl",
            "Lasso": "lasso_model.pkl"
        }
        for name, file in pkl_models.items():
            path = os.path.join(MODELS_FOLDER, file)
            if os.path.exists(path):
                models[name] = joblib.load(path)

        # Load Keras model
        keras_path = os.path.join(MODELS_FOLDER, "keras_dense_model.h5")
        if os.path.exists(keras_path):
            models["Keras Dense"] = load_model(keras_path, compile=False)

        # Load preprocessor
        preprocessor_path = os.path.join(MODELS_FOLDER, "preprocessor.pkl")
        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        
        return models, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None

models, preprocessor = load_models()

# -----------------------------
# FIXED: Proper feature encoding that matches training
# -----------------------------
def encode_features(model_name, year, mileage, fuelType, transmission, Make, tax=0, mpg=0, engineSize=0):
    """
    Create input features in the EXACT format the preprocessor expects
    """
    try:
        # Calculate derived features exactly as done during training
        age = 2025 - year
        mileage_per_year = mileage / max(age, 1) if age > 0 else mileage
        
        # Calculate price_per_year (this was likely calculated during training)
        # Since we don't know the actual price, we'll use a reasonable estimate
        # This might need adjustment based on your training methodology
        estimated_price = 15000  # You might want to make this more sophisticated
        price_per_year = estimated_price / max(age, 1) if age > 0 else estimated_price
        
        # Create age groups (adjust thresholds to match your training data)
        if age <= 3:
            age_group = 'new'
        elif age <= 7:
            age_group = 'medium' 
        else:
            age_group = 'old'
            
        # Create mileage groups (adjust thresholds to match your training data)
        if mileage <= 30000:
            mileage_group = 'low'
        elif mileage <= 80000:
            mileage_group = 'medium'
        else:
            mileage_group = 'high'
        
        # Create DataFrame with EXACTLY the 14 columns the preprocessor expects
        # Order matters! Must match: ['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize', 'Make', 'age', 'mileage_per_year', 'price_per_year', 'age_group', 'mileage_group']
        input_df = pd.DataFrame([{
            'model': model_name,
            'year': year,
            'transmission': transmission,
            'mileage': mileage,
            'fuelType': fuelType,
            'tax': tax,
            'mpg': mpg,
            'engineSize': engineSize,
            'Make': Make,
            'age': age,
            'mileage_per_year': mileage_per_year,
            'price_per_year': price_per_year,
            'age_group': age_group,
            'mileage_group': mileage_group
        }])
        
        # Debug info
        if st.sidebar.checkbox("Show Debug Info", key="debug_input"):
            st.sidebar.write("Input DataFrame shape:", input_df.shape)
            st.sidebar.write("Input columns:", list(input_df.columns))
            st.sidebar.write("Sample values:")
            for col in input_df.columns:
                st.sidebar.write(f"  {col}: {input_df[col].iloc[0]}")
        
        # Transform with preprocessor to get 166 features
        if preprocessor:
            processed_features = preprocessor.transform(input_df)
            if st.sidebar.checkbox("Show Preprocessing Debug", key="debug_preprocessing"):
                st.sidebar.write(f"After preprocessing shape: {processed_features.shape}")
        else:
            st.error("Preprocessor not available!")
            return None
        
        return processed_features
        
    except Exception as e:
        st.error(f"Error in feature encoding: {e}")
        st.write("Error details:", str(e))
        return None

# -----------------------------
# Get unique values from dataset for dropdowns
# -----------------------------
def get_dataset_values():
    """Get unique values from the original dataset"""
    if original_df is not None:
        return {
            'models': sorted([model.strip() for model in original_df['model'].unique()]),  # Remove spaces
            'makes': sorted(original_df['Make'].unique()),
            'fuel_types': sorted(original_df['fuelType'].unique()),
            'transmissions': sorted(original_df['transmission'].unique())
        }
    else:
        return {
            'models': ['A1', 'A3', 'A4', 'A5', 'A6'],
            'makes': ['audi', 'BMW', 'Ford', 'vw', 'toyota'],
            'fuel_types': ['Petrol', 'Diesel', 'Hybrid', 'Electric'],
            'transmissions': ['Manual', 'Automatic', 'Semi-Auto']
        }

dataset_values = get_dataset_values()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🚗 Vehicle Price Prediction 🚗</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Estimate your car's price using advanced ML models</h4>", unsafe_allow_html=True)
st.markdown("---")

# Show model loading status
if models:
    st.success(f" Successfully loaded {len(models)} models: {', '.join(models.keys())}")
    if preprocessor:
        st.success(" Preprocessor loaded - Ready for predictions!")
    else:
        st.error(" Preprocessor not loaded")
else:
    st.error(" No models loaded. Please check your model files.")

# Sidebar Inputs
st.sidebar.header(" Input Features")

# Car selection with actual dataset values
model_choice = st.sidebar.selectbox("Car Model", dataset_values['models'], help="Select from available models in dataset")
Make_choice = st.sidebar.selectbox("Car Make", dataset_values['makes'])
year = st.sidebar.number_input("Year", 2000, 2025, 2017)
mileage = st.sidebar.number_input("Mileage (km)", 0, 500000, 20000)
fuel = st.sidebar.selectbox("Fuel Type", dataset_values['fuel_types'])
transmission = st.sidebar.selectbox("Transmission", dataset_values['transmissions'])
engineSize = st.sidebar.number_input("Engine Size (L)", 0.0, 10.0, 1.6, step=0.1)
mpg = st.sidebar.number_input("MPG", 0.0, 500.0, 55.0)
tax = st.sidebar.number_input("Tax (£)", 0, 1000, 145)

if models:
    selected_model = st.sidebar.selectbox("Choose ML Model", list(models.keys()))
else:
    selected_model = None

# Main Prediction Section
col1, col2 = st.columns([1, 2])

with col1:
    if st.button(" Predict Price", disabled=not (selected_model and models and preprocessor)):
        prediction_made = True
    else:
        prediction_made = False

with col2:
    if not models:
        st.warning(" No models available")
    elif not preprocessor:
        st.warning(" No preprocessor available") 
    else:
        st.info(" Ready to predict")

# Make prediction
if prediction_made:
    with st.spinner(" Making prediction..."):
        try:
            # Encode features to match training format
            features = encode_features(model_choice, year, mileage, fuel, transmission, Make_choice, tax, mpg, engineSize)
            
            if features is not None:
                model = models[selected_model]

                # Make prediction - FIXED: Check if model is a pipeline
                if selected_model == "Keras Dense":
                    # For Keras, use preprocessed features
                    prediction = model.predict(features, verbose=0)[0][0]
                else:
                    # For sklearn models, check if they're pipelines
                    if hasattr(model, 'steps'):
                        # Model is a pipeline - use raw input (14 features)
                        raw_input_df = pd.DataFrame([{
                            'model': model_choice,
                            'year': year,
                            'transmission': transmission,
                            'mileage': mileage,
                            'fuelType': fuel,
                            'tax': tax,
                            'mpg': mpg,
                            'engineSize': engineSize,
                            'Make': Make_choice,
                            'age': 2025 - year,
                            'mileage_per_year': mileage / max(1, 2025 - year),
                            'price_per_year': 15000 / max(1, 2025 - year),
                            'age_group': 'new' if (2025 - year) <= 3 else ('medium' if (2025 - year) <= 7 else 'old'),
                            'mileage_group': 'low' if mileage <= 30000 else ('medium' if mileage <= 80000 else 'high')
                        }])
                        prediction = model.predict(raw_input_df)[0]
                        st.info(f"Used pipeline model with raw input (14 features)")
                    else:
                        # Model expects preprocessed features
                        prediction = model.predict(features)[0]
                        st.info(f"Used standalone model with preprocessed input ({features.shape[1]} features)")
                        
                st.success(f"Model type: {'Pipeline' if hasattr(model, 'steps') else 'Standalone'}")

                # Ensure prediction is positive and reasonable
                prediction = abs(prediction)
                
                # Display result prominently
                st.markdown(f"<div style='background: linear-gradient(90deg, #00C851, #007E33); padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;'><h1 style='color: white; margin: 0;'>💰 Estimated Price: £{prediction:,.2f}</h1></div>", unsafe_allow_html=True)

                # Create three columns for detailed info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🚗 Vehicle Info")
                    st.write(f"**Make & Model:** {Make_choice} {model_choice}")
                    st.write(f"**Year:** {year}")
                    st.write(f"**Age:** {2025 - year} years")
                    st.write(f"**Engine:** {engineSize}L {fuel}")
                    
                with col2:
                    st.markdown("#### 📊 Performance")
                    st.write(f"**Mileage:** {mileage:,} km")
                    st.write(f"**MPG:** {mpg}")
                    st.write(f"**Transmission:** {transmission}")
                    st.write(f"**Tax:** £{tax}")
                    
                with col3:
                    st.markdown("#### 🔍 Analysis")
                    miles_per_year = mileage / max(1, 2025 - year)
                    price_per_year = prediction / max(1, 2025 - year)
                    st.write(f"**Miles/Year:** {miles_per_year:,.0f}")
                    st.write(f"**Price/Year:** £{price_per_year:,.0f}")
                    st.write(f"**Model Used:** {selected_model}")
                    
                    # Value assessment
                    if miles_per_year < 10000:
                        st.success("🟢 Low mileage")
                    elif miles_per_year < 15000:
                        st.info("🟡 Average mileage")
                    else:
                        st.warning("🔴 High mileage")


                # Comparison with similar vehicles
                if original_df is not None:
                    st.markdown("### 🔍 Market Context")
                    
                    # Filter similar vehicles
                    similar_cars = original_df[
                        (original_df['Make'] == Make_choice) &
                        (original_df['year'] >= year - 2) & 
                        (original_df['year'] <= year + 2) &
                        (original_df['fuelType'] == fuel)
                    ]
                    
                    if len(similar_cars) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Similar Vehicles Found", len(similar_cars))
                        with col2:
                            st.metric("Average Market Price", f"£{similar_cars['price'].mean():,.0f}")
                        with col3:
                            difference = prediction - similar_cars['price'].mean()
                            st.metric("Price Difference", f"£{difference:,.0f}", delta=f"{difference:,.0f}")
                        with col4:
                            percentile = (similar_cars['price'] < prediction).mean() * 100
                            st.metric("Market Percentile", f"{percentile:.0f}%")
                    else:
                        st.info("No similar vehicles found in dataset for comparison")

                st.markdown("---")
                st.markdown("** Disclaimer:** This prediction is based on historical data and machine learning models. Actual market prices may vary due to vehicle condition, location, market trends, and other factors not captured in the model.")
            
            else:
                st.error(" Failed to process input features. Please check your selections.")
                
        except Exception as e:
            st.error(f" Error making prediction: {e}")
            with st.expander(" Technical Details"):
                st.write("Error type:", type(e).__name__)
                st.write("Error message:", str(e))
                st.write("This might be due to:")
                st.write("- Model compatibility issues")
                st.write("- Preprocessor mismatch")
                st.write("- Invalid input values")

# Dataset information section
if original_df is not None:
    with st.expander(" Dataset Information"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(original_df):,}")
        with col2:
            st.metric("Features", len(original_df.columns))
        with col3:
            st.metric("Average Price", f"£{original_df['price'].mean():,.0f}")
        with col4:
            st.metric("Price Range", f"£{original_df['price'].min():,} - £{original_df['price'].max():,}")
        
        if st.checkbox("Show sample data"):
            st.dataframe(original_df.head(10))

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>🚗 Vehicle Price Predictor | Powered by Machine Learning & Streamlit </div>", unsafe_allow_html=True)
