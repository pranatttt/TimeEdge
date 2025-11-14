ForecastIQ – Walmart SARIMAX Sales Forecasting System

ForecastIQ is a complete end-to-end time series forecasting and analytics system designed to predict Walmart weekly sales using SARIMAX, served via a FastAPI backend, stored in PostgreSQL, and visualized using an interactive HTML/CSS/JavaScript (Chart.js) dashboard.
The platform also includes an optional LLM-powered Insights Assistant for natural language querying.

This project demonstrates strong skills in machine learning engineering, backend development, MLOps, data pipelines, time-series modeling, and full-stack integration.

🚀 Project Overview

ForecastIQ automates the entire forecasting lifecycle:

✓ Data ingestion
✓ Weekly feature aggregation
✓ SARIMAX training
✓ Evaluation & metrics
✓ Exporting forecasts
✓ Uploading results to PostgreSQL
✓ Exposing real-time APIs via FastAPI
✓ Interactive visualization dashboard
✓ Natural language insights using LLM (Ollama)

This mirrors a real-world production ML system.

✅ Key Features
🔹 1. SARIMAX Forecasting Engine

Model: SARIMAX (1,1,1)(0,1,1,52)

Handles weekly seasonality

Trains on 2010–2011 data and forecasts 2012

Generates:

walmart_forecast_results.csv – test period forecasts

walmart_forecast_full.csv – full dataset (train + test)

🔹 2. FastAPI Backend

/metrics/{year} → monthly actual vs forecast

/chat → natural language Q&A

/ → health check

Cross-origin support (CORS)

🔹 3. PostgreSQL Integration

Stores forecasting output in table sarimax_full using upload_data.py.

🔹 4. Interactive Dashboard (HTML + JS + Chart.js)

Monthly actual vs forecast chart

Quarterly profit/loss

Trend visualization

Year selector (2010, 2011, 2012)

🔹 5. Insights Assistant (Optional)

Using Ollama (llama3.2):

Ask:

“What were the sales in November 2012?”

Backend answers using SQL + LLM reasoning.

🛠 Tech Stack
Layer	Technology
Model	SARIMAX (statsmodels)
Backend	FastAPI
Database	PostgreSQL + SQLAlchemy
Frontend	HTML, CSS, JavaScript, Chart.js
LLM (optional)	Ollama (llama3.2)
Environment	Python, Pandas, NumPy
📁 Project Structure
ForecastIQ/
│── Backend/
│   ├── Data/
│   │   ├── base_data.csv
│   │   ├── walmart_forecast_full.csv
│   │   └── walmart_forecast_results.csv
│   ├── models/
│   │   └── sarimax_model.pkl
│   ├── main.py
│   ├── upload_data.py
│   └── walmart_sarimax_forecast.py
│
│── Frontend/
│   ├── dashboard.html
│   ├── dashboard.css
│   ├── dashboard.js
│   ├── assistant.html
│   ├── assistant.css
│   └── assistant.js
│
│── .gitignore
│── requirements.txt
│── README.md

🧠 Time Series Modeling Workflow

Load Walmart weekly data

Aggregate into weekly frequency

Train SARIMAX with seasonal=52 weeks

Choose best model via AIC

Forecast test period

Generate future 12-week predictions

Save outputs to CSV

Upload to PostgreSQL

Serve metrics via the API

🔗 API Endpoints
GET /

Health check.

GET /metrics/{year}

Returns:

Monthly actual (12 values)

Monthly forecast (12 values)

Quarterly profit/loss

Total revenue and margin

POST /chat

LLM-powered insights based on database.

📊 Dashboard (Frontend)

Features include:

Actual vs Forecast chart

Quarterly insights

Trend analysis

Year selector

Integrated Insights Assistant

📦 Installation
1️⃣ Create environment
python -m venv env
env\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Configure database (.env)
DB_URI=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/sales_db

4️⃣ Upload data
cd Backend
python upload_data.py

5️⃣ Run API
uvicorn main:app --reload --port 8000

6️⃣ Open dashboard
Frontend/dashboard.html
Frontend/assistant.html

