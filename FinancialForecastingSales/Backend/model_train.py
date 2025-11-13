"""
walmart_sarimax_retrain.py
----------------------------
Re-train or update SARIMAX model whenever new Walmart data arrives.
This script:
- Loads master CSV + new batch data
- Merges and cleans
- Retrains SARIMAX with saved best params
- Generates forecasts for future weeks
- Saves updated master and model files
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# === Data Preprocessing Function ===
def prepare_weekly_data(df):
    """Aggregate and prepare exogenous variables for SARIMAX"""
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_weekly = df.groupby('Date').agg({
        'Weekly_Sales': 'sum',
        'Holiday_Flag': 'max',
        'Temperature': 'mean',
        'Fuel_Price': 'mean',
        'CPI': 'mean',
        'Unemployment': 'mean'
    }).sort_index()

    df_weekly = df_weekly.asfreq('W-FRI')  # Weekly frequency
    exog_vars = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    exog = df_weekly[exog_vars]
    return df_weekly, exog


# === Model Retraining Function ===
def retrain_sarimax(new_data_csv, master_csv, model_save_dir, forecast_weeks=12):
    """
    Re-trains SARIMAX using master + new data, then forecasts future sales.
    """

    # 1️⃣ Load master and new batch data
    print("📥 Loading master and new data...")
    master_df = pd.read_csv(master_csv)
    new_df = pd.read_csv(new_data_csv)
    combined_df = pd.concat([master_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)

    # 2️⃣ Prepare weekly aggregated data
    df_weekly, exog = prepare_weekly_data(combined_df)

    # 3️⃣ Define exogenous features and split
    exog_train = exog
    y_train = df_weekly['Weekly_Sales']

    # 4️⃣ Load or define best model parameters (update if needed)
    # These can be saved once from your initial grid search
    best_order = (1, 1, 1)
    best_seasonal_order = (1, 1, 1, 52)

    print(f"🔁 Re-training SARIMAX with order={best_order}, seasonal_order={best_seasonal_order}...")

    model = SARIMAX(y_train, exog=exog_train,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)

    # 5️⃣ Evaluate performance on last 10% data (optional)
    test_size = int(len(y_train) * 0.1)
    y_test = y_train[-test_size:]
    exog_test = exog_train[-test_size:]
    forecast = results.get_forecast(steps=test_size, exog=exog_test)
    forecast_mean = forecast.predicted_mean

    rmse = np.sqrt(mean_squared_error(y_test, forecast_mean))
    mape = np.mean(np.abs((y_test - forecast_mean) / y_test)) * 100
    print(f"📊 RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # 6️⃣ Forecast future sales
    print(f"🔮 Forecasting next {forecast_weeks} weeks...")
    future_forecast = results.get_forecast(steps=forecast_weeks, exog=None)
    forecast_mean = future_forecast.predicted_mean
    conf_int = future_forecast.conf_int()

    future_dates = pd.date_range(
        start=df_weekly.index[-1] + pd.Timedelta(weeks=1),
        periods=forecast_weeks,
        freq='W-FRI'
    )

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Sales': forecast_mean,
        'Lower_CI': conf_int.iloc[:, 0].values,
        'Upper_CI': conf_int.iloc[:, 1].values
    })

    # 7️⃣ Save updated master CSV and model
    combined_df.to_csv(master_csv, index=False)
    model_path = os.path.join(model_save_dir, "sarimax_model.pkl")
    joblib.dump(results, model_path)

    forecast_path = os.path.join(model_save_dir, "walmart_future_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)

    print(f"✅ Model re-trained and saved: {model_path}")
    print(f"✅ Future forecast saved: {forecast_path}")

    return forecast_df


# === Run Example ===
if __name__ == "__main__":
    forecast_df = retrain_sarimax(
        new_data_csv="new_walmart_data.csv",
        master_csv="walmart_master.csv",
        model_save_dir="models",
        forecast_weeks=12
    )
    print(forecast_df.head())
