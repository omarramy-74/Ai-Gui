# AI-Gui XGBoost Car Price Predictor

This repository contains a Streamlit web application that predicts car prices using XGBoost models.

## Project Structure

```
.
├── .devcontainer/          # Development container configuration
├── .github/                # GitHub workflows
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── xgb_feature_names.pkl   # Pickle file containing feature names
├── xgb_model.json           # Main XGBoost model
├── xgb_poly_transformer.pkl # Polynomial features transformer
├── xgb_scaler_X.pkl        # Feature scaler
└── xgb_scaler_y.pkl        # Target scaler
```

## How to Run Locally

1. Clone this repository:
   ```
   git clone https://github.com/omarramy-74/Ai-Gui.git
   cd Ai-Gui
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your browser and go to `http://localhost:8501`

## Deployment

This app can be deployed on Streamlit Cloud by linking this GitHub repository.

## Requirements

See `requirements.txt` for all dependencies.
