import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained model safely
@st.cache_resource
def load_model():
    try:
        with open("model (1).pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: 'model (1).pkl' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# App header
st.title("🏠 California Housing Price Predictor")
st.write("""
This app predicts the **Median House Value** based on regional census data metrics. 
Input the characteristics below to generate a prediction.
""")

st.markdown("---")

if model is not None:
    # Organize inputs into columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input("Longitude", value=-122.23, step=0.01, format="%.2f")
        latitude = st.number_input("Latitude", value=37.88, step=0.01, format="%.2f")
        housing_median_age = st.number_input("Housing Median Age (years)", min_value=1.0, max_value=100.0, value=41.0, step=1.0)
        total_rooms = st.number_input("Total Rooms in Block", min_value=1, value=880)
        total_bedrooms = st.number_input("Total Bedrooms in Block", min_value=1, value=129)

    with col2:
        population = st.number_input("Population in Block", min_value=1, value=322)
        households = st.number_input("Households in Block", min_value=1, value=126)
        median_income = st.number_input("Median Income (in tens of thousands, e.g., 8.32 = $83,200)", min_value=0.5, value=8.3252, format="%.4f")
        
        # Categorical Selection for Ocean Proximity
        # The model specifically expects 4 binary columns: INLAND, ISLAND, NEAR BAY, NEAR OCEAN
        # Note: If it's '<1H OCEAN', all 4 of these dummy variables will be 0.
        ocean_prox = st.selectbox(
            "Ocean Proximity",
            options=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
        )

    # Map the dropdown selection back to the model's expected binary features
    ocean_proximity_INLAND = 1 if ocean_prox == "INLAND" else 0
    ocean_proximity_ISLAND = 1 if ocean_prox == "ISLAND" else 0
    ocean_proximity_NEAR_BAY = 1 if ocean_prox == "NEAR BAY" else 0
    ocean_proximity_NEAR_OCEAN = 1 if ocean_prox == "NEAR OCEAN" else 0

    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict House Value", type="primary"):
        # Match the exact feature names and order found in your .pkl file
        feature_names = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
        ]
        
        # Create a DataFrame to ensure feature names match perfectly if the model requires it
        input_data = pd.DataFrame([[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            ocean_proximity_INLAND, ocean_proximity_ISLAND,
            ocean_proximity_NEAR_BAY, ocean_proximity_NEAR_OCEAN
        ]], columns=feature_names)
        
        try:
            prediction = model.predict(input_data)[0]
            
            # Format output nicely
            if prediction < 0:
                st.warning("The model predicted a negative value based on these inputs. Try adjusting the parameters.")
            else:
                st.success(f"### Estimated Median House Value: **${prediction:,.2f}**")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
