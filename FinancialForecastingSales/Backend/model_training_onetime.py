# walmart_sarimax_forecast.py
# ---------------------------------------
# Walmart Sales Forecasting using SARIMAX
# ---------------------------------------
import os
import pandas as pd
import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# -------------------------------
# STEP 1: Load and Prepare Dataset
# -------------------------------
print("📥 Loading dataset...")
df = pd.read_csv("Data/base_data.csv")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df_weekly = df.groupby('Date').agg({
    'Weekly_Sales': 'sum',
    'Holiday_Flag': 'max',
    'Temperature': 'mean',
    'Fuel_Price': 'mean',
    'CPI': 'mean',
    'Unemployment': 'mean'
}).sort_index()

df_weekly = df_weekly.asfreq('W-FRI')

exog_vars = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
exog = df_weekly[exog_vars]

# -------------------------------
# STEP 2: Train/Test Split
# -------------------------------
train = df_weekly.iloc[:-26]
test = df_weekly.iloc[-26:]
exog_train = exog.iloc[:-26]
exog_test = exog.iloc[-26:]

print(f"🗓️ Train: {train.index[0]} → {train.index[-1]}")
print(f"🧾 Test : {test.index[0]} → {test.index[-1]}")

# -------------------------------
# STEP 3: SARIMAX Grid Search
# -------------------------------
p = d = q = range(0, 2)
P = D = Q = range(0, 2)
s = 52  # Weekly data, annual seasonality

parameters = list(itertools.product(p, d, q, P, D, Q))
best_aic = np.inf
best_model = None
best_order = None
best_seasonal_order = None

print("\n🔍 Performing SARIMAX grid search...\n")

for param in parameters:
    order = (param[0], param[1], param[2])
    seasonal_order = (param[3], param[4], param[5], s)
    try:
        model = SARIMAX(train['Weekly_Sales'], exog=exog_train,
                        order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        if results.aic < best_aic:
            best_aic = results.aic
            best_model = results
            best_order = order
            best_seasonal_order = seasonal_order
    except:
        continue

print(f"✅ Best SARIMAX Order: {best_order}")
print(f"✅ Best Seasonal Order: {best_seasonal_order}")
print(f"✅ Best AIC: {best_aic:.2f}\n")

# -------------------------------
# STEP 4: Forecast and Evaluate
# -------------------------------
forecast = best_model.get_forecast(steps=len(test), exog=exog_test)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

mae = mean_absolute_error(test['Weekly_Sales'], forecast_mean)
rmse = np.sqrt(mean_squared_error(test['Weekly_Sales'], forecast_mean))
mape = np.mean(np.abs((test['Weekly_Sales'] - forecast_mean) / test['Weekly_Sales'])) * 100

print(f"📊 MAE: {mae:.2f}")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 MAPE: {mape:.2f}%\n")

# Plot forecast vs actual
plt.figure(figsize=(12,6))
plt.plot(train.index, train['Weekly_Sales'], label='Train', color='blue')
plt.plot(test.index, test['Weekly_Sales'], label='Actual', color='black')
plt.plot(test.index, forecast_mean, label='Forecast', color='orange')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2)
plt.title('Walmart Weekly Sales Forecast (SARIMAX)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# STEP 5: Time Series Decomposition
# -------------------------------
decomp = seasonal_decompose(train['Weekly_Sales'], model='additive', period=52)
decomp.plot()
plt.suptitle("Trend, Seasonality, and Residuals", fontsize=14)
plt.show()

# -------------------------------
# STEP 6: 12-Week Future Forecast (Save Inside Backend/Data)
# -------------------------------

# Create future dates
future_steps = 12
future_dates = pd.date_range(
    start=df_weekly.index[-1] + pd.Timedelta(weeks=1),
    periods=future_steps,
    freq='W-FRI'
)

# Create future exogenous features (reusing last known values)
exog_future = pd.DataFrame({
    'Holiday_Flag': [df_weekly['Holiday_Flag'].iloc[-1]] * future_steps,
    'Temperature': [df_weekly['Temperature'].iloc[-1]] * future_steps,
    'Fuel_Price': [df_weekly['Fuel_Price'].iloc[-1]] * future_steps,
    'CPI': [df_weekly['CPI'].iloc[-1]] * future_steps,
    'Unemployment': [df_weekly['Unemployment'].iloc[-1]] * future_steps
}, index=future_dates)

future_forecast = best_model.get_forecast(steps=future_steps, exog=exog_future)
future_mean = future_forecast.predicted_mean
future_ci = future_forecast.conf_int()

# Plot future forecast
plt.figure(figsize=(12,6))
plt.plot(df_weekly.index, df_weekly['Weekly_Sales'], label='Historical', color='blue')
plt.plot(future_dates, future_mean, label='Future Forecast', color='green')
plt.fill_between(future_dates, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                 color='green', alpha=0.2)
plt.title("Next 12-Week Walmart Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()

# SAVE DIRECTLY TO existing Backend/Data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of this script
data_dir = os.path.join(BASE_DIR, "Data")               # Backend/Data
os.makedirs(data_dir, exist_ok=True)

future_plot_path = os.path.join(data_dir, "future_forecast.png")
plt.savefig(future_plot_path)
plt.close()

print(f"🖼️ Future forecast saved to: {future_plot_path}")

# -------------------------------
# STEP 7: Save CSV Outputs to Backend/Data
# -------------------------------

# 1️⃣ Save test forecast results
results_path = os.path.join(data_dir, "walmart_forecast_results.csv")

results_df = pd.DataFrame({
    'Date': test.index,
    'Actual_Sales': test['Weekly_Sales'],
    'Forecast_Sales': forecast_mean,
    'Lower_CI': conf_int.iloc[:, 0],
    'Upper_CI': conf_int.iloc[:, 1]
})

results_df.to_csv(results_path, index=False)
print(f"💾 Test forecast results saved to: {results_path}")

# 2️⃣ Save full dataset (2010–2012)
full_df = pd.concat([
    pd.DataFrame({
        'Date': train.index,
        'Actual_Sales': train['Weekly_Sales'],
        'Forecast_Sales': np.nan,
        'Lower_CI': np.nan,
        'Upper_CI': np.nan
    }),
    results_df
]).reset_index(drop=True)

full_path = os.path.join(data_dir, "walmart_forecast_full.csv")
full_df.to_csv(full_path, index=False)
print(f"📘 Full dataset saved to: {full_path}")
