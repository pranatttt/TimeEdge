"""
main_chat_sarimax.py
--------------------
FastAPI service for natural-language chat powered by Ollama (LLaMA 3.2)
✅ CORS enabled
✅ Conversation memory preserved
✅ Compatible with SARIMAX / XGBoost dashboards
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import urllib3

# === Disable SSL warnings for local Ollama ===
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === LangChain Imports ===
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --------------------------------------------------
# 🔧 FASTAPI INITIALIZATION
# --------------------------------------------------
app = FastAPI(title="Walmart AI Chat Assistant (Ollama + SARIMAX)")

# Allow all CORS for frontend (Streamlit / React / Power BI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 🧠 LLM CONFIGURATION
# --------------------------------------------------
llm_natural = ChatOllama(
    model="llama3.2",  # ✅ Local LLaMA 3.2 model
    base_url="http://localhost:11434",  # Ollama API URL
)

# === Prompt Template (with conversational memory) ===
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent business and sales assistant. "
        "You help users analyze forecasts, sales metrics, and trends from Walmart data. "
        "Keep your tone professional, brief, and data-driven."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# === Conversation Memory & Chain ===
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

conversation_chain = ConversationChain(
    llm=llm_natural,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

# --------------------------------------------------
# 📥 Request Schema
# --------------------------------------------------
class ChatInput(BaseModel):
    message: str


# --------------------------------------------------
# 💬 /chat ENDPOINT
# --------------------------------------------------
@app.post("/chat")
async def chat(input: ChatInput):
    """
    Accepts user message, runs through LLaMA model with conversation memory,
    and returns a helpful natural-language response.
    """
    try:
        response = await run_in_threadpool(conversation_chain.run, input.message)
        return {
            "response": response,
            "source": "ollama_llm",
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


# --------------------------------------------------
# ⚙️ Root Health Check
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "✅ Walmart LLM Chat API (SARIMAX Assistant) is running."}
