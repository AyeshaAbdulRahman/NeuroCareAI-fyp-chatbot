import os
import sys
import faiss
import numpy as np
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell, Verbosity
from mistralai import Mistral
from langchain_community.chat_models import ChatOpenAI
from langdetect import detect as langdetect_detect
from typing import List, Dict, Any

# ──────────────────────────────
# 1. Environment + Setup
# ──────────────────────────────
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

INDEX_PATH = r"vectorstore\index.faiss"
CHUNKS_PATH = r"processed\chunks.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Load FAISS index and chunks ──
if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    print("❌ FAISS index or chunks not found. Run ingestion.py first.")
    index = None
    chunks = []
else:
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        print(f"✅ Loaded {len(chunks)} documents from vector store.")
    except Exception as e:
        print(f"❌ Error loading FAISS/Chunks: {e}")
        index = None
        chunks = []

# Load Models
print("⏳ Loading Embedding Model...")
embedder = SentenceTransformer(EMBED_MODEL)

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    print("✅ SymSpell dictionary loaded")
except Exception as e:
    print(f"⚠️ SymSpell dictionary not found or failed to load: {e}. Spell correction may be less accurate.")

# ──────────────────────────────
# 2. RetrievalTool Class
# ──────────────────────────────
class RetrievalTool:
    def __init__(self, index, chunks, embedder):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    def retrieve(self, query: str, k=5, max_distance=1.0) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []

        query_vec = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_vec).astype("float32"), k)
        results = []

        for distance, idx in zip(D[0], I[0]):
            if idx >= len(self.chunks) or distance > max_distance:
                continue

            doc = self.chunks[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "distance": float(distance)
            })
        return results

# ──────────────────────────────
# 3. Agent Class
# ──────────────────────────────
class Agent:
    def __init__(self, retrieval_tool, llm_client, provider, max_loops=3):
        self.retrieval_tool = retrieval_tool
        self.llm_client = llm_client
        self.provider = provider
        self.max_loops = max_loops

    def _run_llm_prompt(self, prompt: str, model: str = "mistral-small-latest", max_tokens: int = 256) -> str:
        if not self.llm_client:
            return ""
        try:
            if self.provider == "openai":
                response = self.llm_client.invoke(prompt)
                response = response.content if hasattr(response, 'content') else str(response)
            else:
                resp = self.llm_client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0
                )
                response = resp.choices[0].message.content
            return response.strip()
        except Exception as e:
            return ""

    def _decide_retrieval(self, query: str, current_context: str, history: List[Dict[str, str]]) -> str:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])  # Last 5 for brevity

        prompt = f"""
        You are an agent deciding whether to retrieve information from documents for the query.

        Query: {query}
        Current Context: {current_context}
        Chat History: {history_text}

        Decide: Should I retrieve more information? Answer 'yes' or 'no'. If 'no', I will generate a response without retrieval.
        """
        decision = self._run_llm_prompt(prompt, max_tokens=10).lower().strip()
        return "retrieve" if decision == "yes" else "no_retrieval"

    def _is_sufficient(self, docs: List[Dict[str, Any]], query: str) -> bool:
        if not docs:
            return False
        context = "\n".join([d["text"] for d in docs])
        prompt = f"""
        Does the following context sufficiently answer the query: {query}?

        Context: {context}

        Answer 'yes' or 'no'.
        """
        answer = self._run_llm_prompt(prompt, max_tokens=10).lower().strip()
        return answer == "yes"

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return ""
        return "\n\n".join([f"[Source: {r['metadata'].get('source', 'Unknown')}]\n{r['text']}" for r in docs])

    def _extract_references(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        references = []
        for r in docs:
            references.append({
                "chunk_id": r["chunk_id"],
                "source": r["metadata"].get("source", "Unknown"),
                "page": r["metadata"].get("page", "N/A"),
                "chunk_index": r["metadata"].get("chunk_index", 0)
            })
        return references

    def _generate_response(self, query: str, context: str, history: List[Dict[str, str]], user_language: str) -> str:
        history_block = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-5:]])

        no_info_message = "The available documents do not contain information about this topic."
        translated_no_info = no_info_message  # Translation handled in ChatHandler

        system_prompt = f"""
You are a compassionate NeuroCare AI Assistant specializing in Alzheimer’s, Parkinson’s, Dementia, and FTD.
Your ONLY goal is to provide a helpful response.

**CRITICAL INSTRUCTIONS:**
1. **Answer ONLY using the provided [CONTEXT]**.
2. If the answer is **NOT** found in the [CONTEXT], your entire reply MUST be: "{translated_no_info}"
3. **Generate the entire response ONLY in the following language: English.**  # Agent generates in English, ChatHandler translates
4. DO NOT use other languages in the final reply. DO NOT include any introductory phrases, translation preambles, or meta-text.

[CONTEXT]
{context}

[HISTORY]
{history_block}
"""
        final_prompt = f"{system_prompt}\n\nUSER: {query}\nASSISTANT:"

        try:
            if self.provider == "openai":
                response = self.llm_client.invoke(final_prompt)
                reply = response.content if hasattr(response, 'content') else str(response)
            else:
                resp = self.llm_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": final_prompt}],
                    max_tokens=600,
                    temperature=0.3
                )
                reply = resp.choices[0].message.content
        except Exception as e:
            reply = "I'm sorry, I encountered an error connecting to the AI service."

        return reply

    def process_query(self, query: str, history: List[Dict[str, str]], user_language: str) -> tuple[str, List[Dict[str, Any]]]:
        context = ""
        loop_count = 0
        all_docs = []

        while loop_count < self.max_loops:
            decision = self._decide_retrieval(query, context, history)
            if decision == "no_retrieval":
                response = self._generate_response(query, "", history, user_language)
                return response, []

            elif decision == "retrieve":
                docs = self.retrieval_tool.retrieve(query)
                all_docs.extend(docs)
                if self._is_sufficient(docs, query):
                    context = self._format_context(all_docs)
                    response = self._generate_response(query, context, history, user_language)
                    references = self._extract_references(all_docs)
                    return response, references
                else:
                    context += self._format_context(docs)
                    loop_count += 1

        # Max loops reached
        response = self._generate_response(query, context, history, user_language)
        references = self._extract_references(all_docs)
        return response, references

# ──────────────────────────────
# 4. ChatHandler Class
# ──────────────────────────────
class ChatHandler:
    def __init__(self):
        self.custom_corrections = {
            "alzaimer": "Alzheimer's",
            "dimentia": "dementia",
            "parkinson": "Parkinson's",
            "ftd": "frontotemporal dementia"
        }

        self.provider = "mistral" if MISTRAL_API_KEY else "openai"
        if MISTRAL_API_KEY:
            print("🪶 Using Mistral API model...")
            self.llm_client = Mistral(api_key=MISTRAL_API_KEY)
            self.provider = "mistral"
        elif OPENAI_KEY:
            print("🤖 Using OpenAI GPT model...")
            self.llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=OPENAI_KEY)
            self.provider = "openai"
        else:
            print("⚠️ No API Key found. Chat will fail.")
            self.llm_client = None

        # Initialize RetrievalTool and Agent
        self.retrieval_tool = RetrievalTool(index, chunks, embedder)
        self.agent = Agent(self.retrieval_tool, self.llm_client, self.provider)

        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 10
        self.user_language = "English"

    # ── Helper: Detect Casual Messages (Greetings/Farewells) ──
    def is_casual_message(self, query: str) -> bool:
        casual_messages = [
            # Greetings
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "سلام", "ہیلو", "ہائے",
            # Farewells
            "bye", "goodbye", "see you", "see you later", "take care", "farewell", "خدا حافظ", "الوداع", "بائے", "گڈ بائے"
        ]
        return query.lower().strip() in casual_messages or len(query.split()) <= 1  # Short queries like "hi" or "bye"

    # ── Helper: Spell Check (Enhanced) ──
    def correct_query(self, query: str) -> str:
        words = query.lower().split()
        corrected_words = []
        for word in words:
            if word in self.custom_corrections:
                corrected_words.append(self.custom_corrections[word])
            else:
                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                corrected_words.append(suggestions[0].term if suggestions else word)
        return " ".join(corrected_words).capitalize()

    # ── Helper: Language Utilities ──
    def _run_llm_prompt(self, prompt: str, model: str = "mistral-small-latest", max_tokens: int = 256) -> str:
        if not self.llm_client:
            return ""
        try:
            if self.provider == "openai":
                response = self.llm_client.invoke(prompt)
                response = response.content if hasattr(response, 'content') else str(response)
            else:
                resp = self.llm_client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0
                )
                response = resp.choices[0].message.content
            return response.strip()
        except Exception as e:
            return ""

    def _detect_language(self, text: str) -> str:
        try:
            prompt = f"""
            Analyze the following text and identify the language.
            Respond with ONLY the name of the language (e.g., "Spanish", "French", "English").
            Text: "{text}"
            """
            language = self._run_llm_prompt(prompt, model="mistral-tiny" if self.provider == "mistral" else "gpt-3.5-turbo")
            if language and len(language.split()) <= 2:
                return language
            detected = langdetect_detect(text)
            # Normalize to full language names
            lang_map = {
                "en": "English",
                "ur": "Urdu",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "hi": "Hindi",
                "ar": "Arabic",
                # Add more as needed
            }
            normalized = lang_map.get(detected.lower(), detected.capitalize())
            return normalized if normalized else "English"
        except:
            return "English"

    def _translate_text(self, text: str, target_language: str) -> str:
        if target_language == "English":
            return text
        # Basic check: If text already looks like target language, skip translation
        if target_language == "Urdu" and any(ord(char) > 127 for char in text):  # Heuristic for non-English
            return text
        prompt = f"""
        Translate the following text to {target_language}.
        Provide ONLY the translated text, no explanations, comments, or additional text.
        Text to translate: "{text}"
        """
        translated = self._run_llm_prompt(prompt, max_tokens=300)  # Increased tokens for longer text
        return translated if translated else text  # Fallback to original if translation fails

    # ── Helper: Retrieve Docs (Tuned) ──
    def retrieve_relevant_docs(self, query: str, k=5, max_distance=1.0) -> List[Dict[str, Any]]:
        if index is None or not chunks:
            return []

        query_vec = embedder.encode([query])
        D, I = index.search(np.array(query_vec).astype("float32"), k)
        results = []

        for distance, idx in zip(D[0], I[0]):
            if idx >= len(chunks) or distance > max_distance:
                continue

            doc = chunks[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "distance": float(distance)
            })
        return results

    # ── Helper: Contextualize Query ──
    def _contextualize_query(self, user_input_english: str) -> str:
        if not self.chat_history or self.is_casual_message(user_input_english):  # Skip contextualization for casual messages or no history
            return user_input_english

        english_history = []
        for msg in self.chat_history[-self.max_history:]:
            content = msg['content']
            if self.user_language != "English":
                content = self._translate_text(content, "English")
            english_history.append(f"{msg['role']}: {content}")

        history_text = "\n".join(english_history)

        prompt = f"""
        Given the chat history and the latest user question in English, formulate a standalone question
        which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed.
        If the question is unrelated or too short, keep it as-is.

        Chat History (in English):
        {history_text}

        Latest User Question (English): {user_input_english}

        Standalone Question:
        """
        response = self._run_llm_prompt(prompt)
        return response if response else user_input_english

    # ── Core: Chat Logic ──
    def chat(self, user_input: str) -> Dict[str, Any]:
        user_input = user_input.strip()
        if not user_input:
            return {"reply": "Please ask a question.", "references": []}

        # 1. Detect Language
        self.user_language = self._detect_language(user_input)
        print(f"DEBUG: Detected language: {self.user_language}")

        # 2. Handle Casual Messages (Greetings/Farewells) Separately
        if self.is_casual_message(user_input):
            if "bye" in user_input.lower() or "goodbye" in user_input.lower() or "خدا حافظ" in user_input or "الوداع" in user_input:
                casual_response = "Goodbye! Take care and feel free to ask about Alzheimer's, Parkinson's, Dementia, or FTD anytime."
            else:
                casual_response = "Hello! How can I assist you with Alzheimer's, Parkinson's, Dementia, or FTD today?"
            translated_casual = self._translate_text(casual_response, self.user_language) if self.user_language != "English" else casual_response
            return {"reply": translated_casual, "references": []}  # No RAG for casual messages

        # 3. Translate and Correct
        user_input_english = self._translate_text(user_input, "English") if self.user_language != "English" else user_input
        corrected_english = self.correct_query(user_input_english)

        # 4. Contextualize
        search_query = self._contextualize_query(corrected_english)

        # 5. Use Agent for RAG Processing
        reply, references = self.agent.process_query(search_query, self.chat_history, self.user_language)

        # 6. Translate Reply to User's Language (if needed)
        if self.user_language != "English":
            reply = self._translate_text(reply, self.user_language)

        # 7. Update History (Only if docs were retrieved to avoid polluting with off-topic)
        if references:
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": reply})

        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-(self.max_history*2):]

        return {
            "reply": reply,
            "references": references
        }

# ──────────────────────────────
# 3. CLI Chat Interface
# ──────────────────────────────
def main():
    print("\n💬 Caregiver Assistant Chatbot (FAISS + Mistral/OpenAI)\n")
    print("Type 'exit' to quit. Ask questions in any language!\n")

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

        result = handler.chat(query)

        print(f"\n🤖 Answer: {result['reply']}\n")

        if result["references"]:
            print("📚 Sources:")
            for ref in result["references"]:
                print(f" - Chunk {ref['chunk_id']} | Page {ref['page']} | {ref['source']}")
            print()

if __name__ == "__main__":
    main()

