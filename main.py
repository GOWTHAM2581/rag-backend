from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests, pickle, faiss, numpy as np
from pypdf import PdfReader
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
# GROQ CLIENT (use env variable)
# ----------------------------
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

groq = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ----------------------------
# EMBEDDING FUNCTION (GROQ)
# ----------------------------
def embed_text(text: str):
    r = groq.embeddings.create(
        model="voyage-large-2",
        input=text
    )
    return np.array(r.data[0].embedding, dtype="float32")


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
# PROCESS PDF ENDPOINT
# ----------------------------
@app.post("/process_pdf")
def process_pdf(req: ProcessRequest):
    uid = req.uid
    user_path = f"user_data/{uid}"
    os.makedirs(user_path, exist_ok=True)

    # Download PDF from Supabase
    pdf_path = f"{user_path}/doc.pdf"
    pdf_bytes = requests.get(req.url).content
    open(pdf_path, "wb").write(pdf_bytes)

    # Extract text
    reader = PdfReader(pdf_path)
    chunks = []
    metadata = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i in range(0, len(text), 800):
            chunk = text[i:i+800].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"page": page_no})

    # Embed all chunks (one by one to reduce API load)
    vectors = []
    for c in chunks:
        vectors.append(embed_text(c))

    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    # Store vector DB
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, f"{user_path}/vectors.index")
    pickle.dump({"chunks": chunks, "metadata": metadata}, open(f"{user_path}/chunks.pkl", "wb"))

    return {"status": "PDF processed successfully", "chunks": len(chunks)}


# ----------------------------
# ASK QUESTION ENDPOINT
# ----------------------------
@app.post("/ask")
def ask(req: AskRequest):
    uid = req.uid
    user_path = f"user_data/{uid}"

    index = faiss.read_index(f"{user_path}/vectors.index")
    data = pickle.load(open(f"{user_path}/chunks.pkl", "rb"))

    chunks = data["chunks"]
    metadata = data["metadata"]

    query_vec = embed_text(req.question).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    scores, idxs = index.search(query_vec, 3)

    # Build context
    context = ""
    for idx in idxs[0]:
        context += f"[Page {metadata[idx]['page']}] {chunks[idx]}\n\n"

    # LLM answer
    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer using ONLY the context."},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {req.question}"}
        ]
    )

    return {"answer": completion.choices[0].message.content}


# ----------------------------
# DELETE USER DATA
# ----------------------------
@app.delete("/delete_user")
def delete_user(uid: str):
    import shutil
    dir_path = f"user_data/{uid}"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    return {"status": "deleted"}


