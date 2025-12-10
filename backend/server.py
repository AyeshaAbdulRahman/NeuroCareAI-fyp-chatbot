from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from chatbot import ChatHandler

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your RAG model
handler = ChatHandler()

@app.post("/chat")
def chat(payload: dict):
    user_message = payload.get("message", "")
    print(f"SERVER DEBUG: Received message: {user_message}")  # Debug log
    result = handler.chat(user_message)
    print(f"SERVER DEBUG: Raw reply before any processing: {result['reply']}")  # Debug log
    return {
        "reply": result["reply"],
        "references": result["references"],
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, reload=True)