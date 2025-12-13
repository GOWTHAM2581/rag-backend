from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests, pickle, shutil
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------
# FASTAPI APP + CORS
# ----------------------------
app = FastAPI()

origins = [
    "https://doc-gpt-4rcx.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# LOCAL EMBEDDING MODEL (CORRECT)
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

def embed_text(text: str):
    return embedding_model.encode(text).astype("float32")

# ----------------------------
# GROQ CLIENT (CHAT ONLY)
# ----------------------------
groq = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

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
    user_path = f"user_data/{uid}"
    os.makedirs(user_path, exist_ok=True)

    # Download PDF
    pdf_path = f"{user_path}/doc.pdf"
    r = requests.get(req.url)
    r.raise_for_status()
    open(pdf_path, "wb").write(r.content)

    # Extract text
    reader = PdfReader(pdf_path)
    chunks, metadata = [], []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i in range(0, len(text), 800):
            chunk = text[i:i+800].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"page": page_no})

    if not chunks:
        return {"error": "No readable text found in PDF"}

    # Embed chunks
    vectors = np.array([embed_text(c) for c in chunks], dtype="float32")
    faiss.normalize_L2(vectors)

    # Create FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    faiss.write_index(index, f"{user_path}/vectors.index")
    pickle.dump(
        {"chunks": chunks, "metadata": metadata},
        open(f"{user_path}/chunks.pkl", "wb")
    )

    return {
        "status": "PDF processed successfully",
        "chunks": len(chunks)
    }

# ----------------------------
# ASK QUESTION
# ----------------------------
@app.post("/ask")
def ask(req: AskRequest):
    user_path = f"user_data/{req.uid}"

    index = faiss.read_index(f"{user_path}/vectors.index")
    data = pickle.load(open(f"{user_path}/chunks.pkl", "rb"))

    chunks = data["chunks"]
    metadata = data["metadata"]

    query_vec = embed_text(req.question).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, idxs = index.search(query_vec, 3)

    context = ""
    for idx in idxs[0]:
        context += f"[Page {metadata[idx]['page']}] {chunks[idx]}\n\n"

    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer ONLY using the context."},
            {"role": "user", "content": f"Context:\n{context}\nQuestion:\n{req.question}"}
        ]
    )

    return {"answer": completion.choices[0].message.content}

# ----------------------------
# DELETE USER DATA
# ----------------------------
@app.delete("/delete_user")
def delete_user(uid: str):
    path = f"user_data/{uid}"
    if os.path.exists(path):
        shutil.rmtree(path)
    return {"status": "deleted"}

