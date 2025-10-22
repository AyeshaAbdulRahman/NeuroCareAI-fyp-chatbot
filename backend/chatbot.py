import os
import sys
import faiss
import numpy as np
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# ⚙️ FIX: update deprecated import
from langchain_community.chat_models import ChatOpenAI

# ⚙️ FIX: use new Mistral client import
from mistralai import Mistral


# ──────────────────────────────
# 1. Environment + Setup
# ──────────────────────────────
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

INDEX_PATH = r"vectorstore\index.faiss"
CHUNKS_PATH = r"processed\chunks.jsonl"

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match ingestion.py

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# ──────────────────────────────
# 2. Load FAISS index + embeddings
# ──────────────────────────────
print("🔍 Loading FAISS index and chunks...")

if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    print("❌ FAISS index or chunks not found. Run ingestion.py first.")
    sys.exit(1)

# chunks = np.load(CHUNKS_PATH, allow_pickle=True)
embedder = SentenceTransformer(EMBED_MODEL)

# ──────────────────────────────
# 3. Utility: retrieve top relevant chunks
# ──────────────────────────────
def retrieve_relevant_docs(query, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    if I is None or len(I) == 0 or len(I[0]) == 0:
        print("[warn] No relevant documents found.")
        return []
    return [chunks[i] for i in I[0] if i < len(chunks)]


# ──────────────────────────────
# 4. Unified ChatHandler for OpenAI + Mistral API
# ──────────────────────────────
class ChatHandler:
    def __init__(self):
        if OPENAI_KEY:
            print("🤖 Using OpenAI GPT model...")
            self.client = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
            self.provider = "openai"
        elif MISTRAL_API_KEY:
            print("🪶 Using Mistral API model...")
            self.client = mistral_client
            self.provider = "mistral"
        else:
            raise RuntimeError("❌ No API key found. Please set either OPENAI_API_KEY or MISTRAL_API_KEY.")

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def _build_context(self, query):
        retrieved = retrieve_relevant_docs(query)
        # Extract only text field
        context = "\n\n".join([r["text"] for r in retrieved if "text" in r])
        print(f"[info] Retrieved {len(retrieved)} relevant chunks.")
        return f"Context:\n{context}\n\nQuestion: {query}"

    def chat(self, user_input):
        self.memory.chat_memory.add_user_message(user_input)
        context_prompt = self._build_context(user_input)

        if self.provider == "openai":
            response = self.client.predict(context_prompt)
        else:
            # ✅ Correct Mistral call + proper response extraction
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": context_prompt}]
            )
            response = response.choices[0].message.content

        self.memory.chat_memory.add_ai_message(response)
        return response

# ──────────────────────────────
# 5. CLI Chat Interface
# ──────────────────────────────
def main():
    print("\n💬 Caregiver Assistant Chatbot (FAISS + Mistral/OpenAI)\n")
    print("Type 'exit' to quit.\n")

    try:
        handler = ChatHandler()
    except Exception as e:
        print(f"❌ {e}")
        return

    while True:
        query = input("👤 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response = handler.chat(query)
        print(f"🤖 Bot: {response}\n")

if __name__ == "__main__":
    main()
