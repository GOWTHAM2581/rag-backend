from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests, pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------
# APP
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://doc-gpt-4rcx.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# EMBEDDING MODEL (LOCAL)
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

def embed(text: str):
    return model.encode(text).astype("float32")

# ----------------------------
# REQUEST MODELS
# ----------------------------
class ProcessRequest(BaseModel):
    uid: str
    url: str

class AskRequest(BaseModel):
    uid: str
    question: str

# ----------------------------
# PROCESS PDF
# ----------------------------
@app.post("/process_pdf")
def process_pdf(req: ProcessRequest):
    uid = req.uid
    base_path = f"user_data/{uid}"
    os.makedirs(base_path, exist_ok=True)

    # Download PDF
    pdf_path = f"{base_path}/doc.pdf"
    pdf_bytes = requests.get(req.url, timeout=30).content
    open(pdf_path, "wb").write(pdf_bytes)

    # Extract text
    reader = PdfReader(pdf_path)
    chunks, metadata = [], []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i in range(0, len(text), 700):
            chunk = text[i:i+700].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"page": page_no})

    # Embed
    vectors = np.vstack([embed(c) for c in chunks])
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, f"{base_path}/vectors.index")
    pickle.dump(
        {"chunks": chunks, "metadata": metadata},
        open(f"{base_path}/chunks.pkl", "wb")
    )

    return {"status": "success", "chunks": len(chunks)}

# ----------------------------
# ASK
# ----------------------------
@app.post("/ask")
def ask(req: AskRequest):
    base_path = f"user_data/{req.uid}"

    index = faiss.read_index(f"{base_path}/vectors.index")
    data = pickle.load(open(f"{base_path}/chunks.pkl", "rb"))

    q_vec = embed(req.question).reshape(1, -1)
    faiss.normalize_L2(q_vec)

    _, idxs = index.search(q_vec, 3)

    context = ""
    for i in idxs[0]:
        context += f"[Page {data['metadata'][i]['page']}]\n{data['chunks'][i]}\n\n"

    return {
        "context": context,
        "note": "Send this context to LLM (Groq/OpenAI) from frontend or separate service"
    }

