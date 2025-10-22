import os
import sys
import json
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# PDF + text utilities
from pypdf import PdfReader

# LangChain components
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import FakeEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


from tqdm import tqdm

import torch
# ------------------------------- UTILITIES ---------------------------------- #

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_text_from_pdf(path: Path) -> List[Dict[str, Any]]:
    """Extract text from each page of PDF using pypdf."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_num": i + 1, "text": text})
    return pages

def load_documents_from_dir(input_dir: str) -> List[Dict[str, Any]]:
    """Load all PDF and TXT files from input_dir."""
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    docs = []
    for f in sorted(p.iterdir()):
        if f.is_dir():
            continue
        ext = f.suffix.lower()
        if ext == ".pdf":
            pages = extract_text_from_pdf(f)
            docs.append({"source": f.name, "path": str(f.resolve()), "type": "pdf", "pages": pages})
        elif ext == ".txt":
            text = read_txt_file(f)
            docs.append({"source": f.name, "path": str(f.resolve()), "type": "txt", "pages": [{"page_num": 1, "text": text}]})
        else:
            print(f"[ingest] skipping unsupported file type: {f.name}")
    return docs

def clean_text(text: str) -> str:
    if not text:
        return ""
    txt = text.replace("\r\n", "\n")
    while "\n\n\n" in txt:
        txt = txt.replace("\n\n\n", "\n\n")
    return txt.strip()


# ------------------------------- CHUNKING ---------------------------------- #

def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int=800, chunk_overlap: int=150) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in docs:
        src = doc["source"]
        path = doc["path"]
        for page in doc["pages"]:
            page_num = page.get("page_num", 1)
            page_text = clean_text(page.get("text", "") or "")
            if not page_text:
                continue
            pieces = splitter.split_text(page_text)
            for idx, piece in enumerate(pieces):
                chunk_id = str(uuid.uuid4())
                meta = {
                    "source": src,
                    "path": path,
                    "page": page_num,
                    "chunk_index": idx,
                }
                all_chunks.append({"chunk_id": chunk_id, "text": piece, "metadata": meta})
    return all_chunks


# ------------------------------- VECTORSTORE ---------------------------------- #

def create_vectorstore_from_chunks(chunks: List[Dict[str, Any]],
                                   openai_api_key: str,
                                   vectorstore_dir: str,
                                   use_fake: bool = True,
                                   batch_size: int = 64) -> Tuple[FAISS, List[Document]]:
    """
    Create embeddings (SentenceTransformer or Fake) and store in FAISS.
    Set use_fake=True for offline/demo mode.
    """

    texts, metadatas, ids = [], [], []
    for c in chunks:
        texts.append(c["text"])
        md = c["metadata"].copy()
        md.update({"chunk_id": c["chunk_id"]})
        metadatas.append(md)
        ids.append(c["chunk_id"])

    ensure_dir(vectorstore_dir)

    # ✅ Fix: use the existing flag 'use_fake'
    if use_fake:
        print("[demo] Using local FakeEmbeddings (no API key required)...")
        embeddings = FakeEmbeddings(size=768)
    else:
        print("[local] Using SentenceTransformerEmbeddings (no API key required)...")
        # You can use a GPU-friendly model
        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"[ingest] Creating FAISS vectorstore for {len(texts)} chunks...")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas, ids=ids)

    print(f"[ingest] Saving FAISS index to: {vectorstore_dir}")
    vectorstore.save_local(vectorstore_dir)

    documents = [Document(page_content=txt, metadata=md) for txt, md in zip(texts, metadatas)]
    return vectorstore, documents

# ------------------------------- SAVING ---------------------------------- #

def save_chunks_jsonl(chunks: List[Dict[str, Any]], out_path: str):
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as fh:
        for c in chunks:
            obj = {"chunk_id": c["chunk_id"], "text": c["text"], "metadata": c["metadata"]}
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[ingest] saved chunks metadata -> {out_path}")


# ------------------------------- MAIN ---------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest docs -> chunk -> embed -> FAISS index")
    parser.add_argument("--input_dir", type=str, default="data", help="Directory with PDF and TXT docs")
    parser.add_argument("--processed_dir", type=str, default="processed", help="Directory to write chunks.jsonl")
    parser.add_argument("--vectorstore_dir", type=str, default="vectorstore", help="Directory to persist FAISS index")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--use_fake", action="store_true", help="Use FakeEmbeddings for local demo (no API key)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    print(f"[system] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[system] Using GPU: {torch.cuda.get_device_name(0)}")

    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")

    if not args.use_fake and not openai_key:
        print("[info] Running locally with SentenceTransformerEmbeddings (no API key needed).")


    print(f"[ingest] Loading documents from: {args.input_dir}")
    docs = load_documents_from_dir(args.input_dir)
    if not docs:
        print("[ingest] No documents found in input_dir - exiting.")
        sys.exit(0)

    print(f"[ingest] {len(docs)} documents found. Chunking ...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"[ingest] produced {len(chunks)} chunks.")

    chunks_jsonl_path = os.path.join(args.processed_dir, "chunks.jsonl")
    save_chunks_jsonl(chunks, chunks_jsonl_path)

    vectorstore, documents = create_vectorstore_from_chunks(
        chunks,
        openai_api_key=openai_key,
        vectorstore_dir=args.vectorstore_dir,
        use_fake=args.use_fake,
        batch_size=args.batch_size
    )

    print("[ingest] ✅ Ingestion complete!")
    print(f" - chunks file: {chunks_jsonl_path}")
    print(f" - vectorstore dir: {args.vectorstore_dir}")


if __name__ == "__main__":
    main()
