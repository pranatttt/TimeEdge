# main.py
# Walmart SARIMAX API (Option A - Full Year Mode)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import pandas as pd
from sqlalchemy import create_engine

# -------------------------------
# LOAD ENVIRONMENT
# -------------------------------
load_dotenv()
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("DB_URI not found in .env")

# -------------------------------
# FASTAPI INITIALIZATION
# -------------------------------
app = FastAPI(title="Walmart SARIMAX API (Full Year Mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# AI ASSISTANT (FIXED)
# Using ChatOllama only – NO agent_executor
# -------------------------------
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434"
)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(input: ChatInput):
    """
    FIXED AI Assistant - directly uses LLaMA via Ollama.
    No SQL agent. No agent_executor.
    """
    try:
        response = llm.invoke(input.message)
        return {"response": response.content}
    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# METRICS ENDPOINT (UNCHANGED)
# -------------------------------
@app.get("/metrics/{year}")
def get_metrics(year: int):

    try:
        engine = create_engine(DB_URI)
        df = pd.read_sql("SELECT * FROM sarimax_full", engine)

        # normalize columns
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns:
            return JSONResponse({"error": "date column missing in sarimax_full"}, status_code=400)

        # actual values
        if "actual_sales" not in df.columns:
            alt = [c for c in ["weekly_sales", "actual", "actuals"] if c in df.columns]
            if alt:
                df["actual_sales"] = df[alt[0]]
            else:
                return JSONResponse({"error": "No actual_sales column found"}, status_code=400)

        # forecast values
        forecast_col = None
        for c in ["forecast_sales", "forecasted_sales", "predicted_sales"]:
            if c in df.columns:
                forecast_col = c
                break

        if forecast_col is None:
            df["forecast_sales"] = df["actual_sales"]
        else:
            df[forecast_col] = df[forecast_col].fillna(df["actual_sales"])
            df["forecast_sales"] = df[forecast_col]

        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

        df_year = df[df["year"] == year]
        if df_year.empty:
            return JSONResponse({"error": f"No data for {year}"}, status_code=404)

        # monthly aggregation
        monthly_actual = (
            df_year.groupby("month")["actual_sales"]
            .sum()
            .reindex(range(1, 13), fill_value=0)
            .tolist()
        )

        monthly_forecast = (
            df_year.groupby("month")["forecast_sales"]
            .sum()
            .reindex(range(1, 13), fill_value=0)
            .tolist()
        )

        quarterly_profit = (
            (df_year["actual_sales"] - df_year["forecast_sales"])
            .groupby(df_year["quarter"])
            .sum()
            .reindex(range(1, 5), fill_value=0)
            .tolist()
        )

        total_actual = sum(monthly_actual)
        total_forecast = sum(monthly_forecast)
        profit_loss = total_actual - total_forecast

        # MAPE
        if total_actual != 0:
            mape = round(abs(total_actual - total_forecast) / total_actual * 100, 2)
        else:
            mape = "—"

        return {
            "revenue": f"${int(total_actual)//1000}K",
            "forecasted": f"${int(total_forecast)//1000}K",
            "profitloss": f"${int(profit_loss)//1000}K",
            "margin": f"{mape}%",
            "monthly_revenue": monthly_actual,
            "monthly_forecast": monthly_forecast,
            "quarterly_profit": quarterly_profit,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {"message": "Walmart SARIMAX API Running (Full Year Mode)."}
