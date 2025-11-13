import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load DB credentials
load_dotenv()

DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError("❌ DB_URI not found in .env file. Please add it.")

engine = create_engine(DB_URI)

# -------------------------------
# Path to Backend/Data/walmart_forecast_full.csv
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

csv_path = os.path.join(DATA_DIR, "walmart_forecast_full.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found: {csv_path}")

print(f"📥 Loading CSV: {csv_path}")

df = pd.read_csv(csv_path)

# PostgreSQL prefers lowercase column names
df.columns = [col.lower() for col in df.columns]

# -------------------------------
# Upload into PostgreSQL
# -------------------------------

table_name = "sarimax_full"

print(f"🔼 Uploading into PostgreSQL table: {table_name} ...")

df.to_sql(
    table_name,
    con=engine,
    if_exists="replace",   # overwrite old data every time
    index=False
)

print(f"✅ Successfully uploaded walmart_forecast_full.csv → table: {table_name}")
