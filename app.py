import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="ЁЯЪЧ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_xgboost_model():
    model_data = {}
    
    try:
        model_data['xgb'] = XGBRegressor()
        model_data['xgb'].load_model('xgb_model.pkl')
        model_data['xgb_scaler_X'] = joblib.load('xgb_scaler_X.pkl')
        model_data['xgb_scaler_y'] = joblib.load('xgb_scaler_y.pkl')
        model_data['xgb_feature_names'] = joblib.load('xgb_feature_names.pkl')
        model_data['xgb_poly_transformer'] = joblib.load('xgb_poly_transformer.pkl')
    except Exception as e:
        st.warning(f"тЪая╕П XGBoost model files not found: {str(e)}")
        model_data = None
    
    return model_data

def create_menu():
    menu = ["Prediction", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)
    return choice

def get_feature_encodings():
    manufacturer_mapping = {
        "BMW": 0, "Mercedes-Benz": 1, "Toyota": 2, "Lexus": 3, "Audi": 4,
        "Ford": 5, "Honda": 6, "Hyundai": 7, "Kia": 8, "Nissan": 9,
        "Volkswagen": 10, "Chevrolet": 11, "Mazda": 12, "Subaru": 13
    }
    
    model_mapping = {
        "BMW": {"3 Series": 0, "5 Series": 1, "X5": 2, "X3": 3, "Other": 4},
        "Mercedes-Benz": {"C-Class": 0, "E-Class": 1, "S-Class": 2, "GLE": 3, "Other": 4},
        "Toyota": {"Camry": 0, "Corolla": 1, "RAV4": 2, "Highlander": 3, "Other": 4},
        "Lexus": {"RX": 0, "ES": 1, "IS": 2, "NX": 3, "Other": 4},
        "Audi": {"A4": 0, "A6": 1, "Q5": 2, "Q7": 3, "Other": 4},
        "Ford": {"F-150": 0, "Escape": 1, "Explorer": 2, "Mustang": 3, "Other": 4},
        "Honda": {"Civic": 0, "Accord": 1, "CR-V": 2, "Pilot": 3, "Other": 4},
        "Hyundai": {"Elantra": 0, "Sonata": 1, "Tucson": 2, "Santa Fe": 3, "Other": 4},
        "Kia": {"Optima": 0, "Sorento": 1, "Sportage": 2, "Soul": 3, "Other": 4},
        "Nissan": {"Altima": 0, "Rogue": 1, "Sentra": 2, "Maxima": 3, "Other": 4},
        "Volkswagen": {"Jetta": 0, "Passat": 1, "Tiguan": 2, "Golf": 3, "Other": 4},
        "Chevrolet": {"Silverado": 0, "Equinox": 1, "Malibu": 2, "Tahoe": 3, "Other": 4},
        "Mazda": {"CX-5": 0, "Mazda3": 1, "Mazda6": 2, "CX-9": 3, "Other": 4},
        "Subaru": {"Outback": 0, "Forester": 1, "Impreza": 2, "Legacy": 3, "Other": 4}
    }
    
    leather_interior_mapping = {"Yes": 1, "No": 0}
    
    fuel_type_mapping = {
        "Petrol": 0, "Diesel": 1, "Hybrid": 2, "CNG": 3, 
        "LPG": 4, "Hydrogen": 5, "Electric": 6, "Plug-in Hybrid": 7
    }
    
    gearbox_mapping = {
        "Manual": 0, "Automatic": 1, "Tiptronic": 2, "Variator": 3
    }
    
    drive_wheels_mapping = {
        "Front": 0, "Rear": 1, "4x4": 2
    }
    
    color_mapping = {
        "Black": 0, "White": 1, "Silver": 2, "Gray": 3, "Blue": 4,
        "Red": 5, "Green": 6, "Yellow": 7, "Brown": 8, "Orange": 9
    }
    
    category_mapping = {
        "Sedan": 0, "SUV": 1, "Hatchback": 2, "Crossover": 3, "Coupe": 4,
        "Minivan": 5, "Pickup": 6, "Convertible": 7, "Wagon": 8
    }
    
    wheel_mapping = {
        "Left wheel": 0, "Right-hand drive": 1
    }
    
    return {
        "Manufacturer": manufacturer_mapping,
        "Model": model_mapping,
        "Leather interior": leather_interior_mapping,
        "Fuel type": fuel_type_mapping,
        "Gear box type": gearbox_mapping,
        "Drive wheels": drive_wheels_mapping,
        "Color": color_mapping,
        "Category": category_mapping,
        "Wheel": wheel_mapping
    }

def create_input_features():
    encodings = get_feature_encodings()
    
    col1, col2 = st.columns(2)
    
    with col1:
        manufacturer = st.selectbox("Manufacturer", list(encodings["Manufacturer"].keys()))
        
        model_options = list(encodings["Model"][manufacturer].keys())
        model = st.selectbox("Model", model_options)
        
        prod_year = st.slider("Production Year", 1990, 2025, 2020)
        category = st.selectbox("Category", list(encodings["Category"].keys()))
        leather_interior = st.selectbox("Leather Interior", list(encodings["Leather interior"].keys()))
        fuel_type = st.selectbox("Fuel Type", list(encodings["Fuel type"].keys()))
        engine_volume = st.number_input("Engine Volume (L)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    
    with col2:
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000, step=1000)
        cylinders = st.slider("Cylinders", 2, 12, 4)
        gearbox_type = st.selectbox("Gearbox Type", list(encodings["Gear box type"].keys()))
        drive_wheels = st.selectbox("Drive Wheels", list(encodings["Drive wheels"].keys()))
        wheel = st.selectbox("Wheel", list(encodings["Wheel"].keys()))
        color = st.selectbox("Color", list(encodings["Color"].keys()))
        airbags = st.slider("Airbags", 0, 12, 6)
        levy = st.number_input("Levy", min_value=0, max_value=10000, value=0, step=100)
    
    features = {
        "ID": 0,  
        "Levy": levy,
        "Manufacturer": encodings["Manufacturer"][manufacturer],
        "Model": encodings["Model"][manufacturer][model],
        "Prod. year": prod_year,
        "Category": encodings["Category"][category],
        "Leather interior": encodings["Leather interior"][leather_interior],
        "Fuel type": encodings["Fuel type"][fuel_type],
        "Engine volume": engine_volume,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Gear box type": encodings["Gear box type"][gearbox_type],
        "Drive wheels": encodings["Drive wheels"][drive_wheels],
        "Wheel": encodings["Wheel"][wheel],
        "Color": encodings["Color"][color],
        "Airbags": airbags
    }
    
    return features

def create_dataframe(features):
    df = pd.DataFrame([features])
    return df

def predict_price_xgboost(df, model_data):
    if model_data is None:
        return None
    
    try:
        df_xgb = df[[col for col in model_data['xgb_feature_names'] if col in df.columns]].copy()
        
        for col in model_data['xgb_feature_names']:
            if col not in df_xgb.columns:
                df_xgb[col] = 0
                
        df_xgb = df_xgb[model_data['xgb_feature_names']]
        
        X_poly = model_data['xgb_poly_transformer'].transform(df_xgb)
        X_scaled = model_data['xgb_scaler_X'].transform(X_poly)
        y_pred_scaled = model_data['xgb'].predict(X_scaled)
        prediction = model_data['xgb_scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
        
        return prediction
    except Exception as e:
        st.error(f"Error in XGBoost prediction: {str(e)}")
        return None

def prediction_page(model_data):
    st.markdown("<h1 class='main-header'>Car Price Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>Fill in the car details below to get a predicted price using XGBoost, a powerful machine learning algorithm. The more accurate information you provide, the better the prediction will be.</div>", unsafe_allow_html=True)
    
    features = create_input_features()
    
    if st.button("Predict Car Price", key="predict_button", use_container_width=True):
        with st.spinner("Calculating price..."):
            df = create_dataframe(features)
            price = predict_price_xgboost(df, model_data)
            
            if price is not None:
                st.success("Price prediction completed!")
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h2 class='sub-header'>XGBoost Price Prediction</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-result'>${price:,.2f}</div>", unsafe_allow_html=True)
            else:
                st.error("Could not generate prediction. Please check that model files are available.")

def about_page():
    st.markdown("<h1 class='main-header'>About Car Price Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    This car price prediction application uses the XGBoost machine learning model to estimate the price of a car based on its features. The model was trained on a dataset of car listings with various attributes such as manufacturer, production year, mileage, engine specifications, and more.
    
    ### About XGBoost
    
    XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm known for its:
    
    - **High Performance**: Often outperforms other algorithms in structured/tabular data
    - **Speed**: Efficiently implemented for faster training and prediction
    - **Regularization**: Built-in techniques to prevent overfitting
    - **Handling Missing Values**: Robust processing of missing data
    - **Feature Importance**: Provides insights into which features most affect the prediction
    
    ### Features Used
    
    The prediction is based on the following car attributes:
    
    - Manufacturer (brand)
    - Model
    - Production Year
    - Category (sedan, SUV, etc.)
    - Leather Interior (yes/no)
    - Fuel Type
    - Engine Volume
    - Mileage
    - Number of Cylinders
    - Gearbox Type
    - Drive Wheels
    - Wheel Position
    - Color
    - Number of Airbags
    - Levy (tax)
    
    ### Data Processing
    
    Before training the model, the data underwent several preprocessing steps:
    
    1. Cleaning and validating each feature
    2. Handling missing values
    3. Encoding categorical variables
    4. Scaling numerical features
    5. Feature engineering
    6. Outlier detection and handling
    
    ### Usage Tips
    
    For the most accurate predictions:
    
    - Provide accurate information for all fields
    - Consider that newer cars with lower mileage typically have higher values
    - Luxury brands and special features like leather interior can significantly impact price
    """)
    
    st.markdown("---")
    st.markdown("### Created with Streamlit and XGBoost")

def generate_mock_prediction():
    st.warning("тЪая╕П Using mock prediction for demonstration purposes")
    return 25500 + np.random.normal(0, 1000)

def main():
    model_data = load_xgboost_model()
    choice = create_menu()
    
    debug_mode = st.sidebar.checkbox("Use mock predictions (for testing)", value=False)
    
    if choice == "Prediction":
        prediction_page(model_data)
        
        if debug_mode or model_data is None:
            if st.button("Generate Mock Prediction", key="mock_button"):
                mock_price = generate_mock_prediction()
                
                st.success("Mock prediction generated!")
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h2 class='sub-header'>Mock XGBoost Price Prediction</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-result'>${mock_price:,.2f}</div>", unsafe_allow_html=True)
                
    elif choice == "About":
        about_page()

if __name__ == "__main__":
    main()
