from fastapi import FastAPI
from pydantic import BaseModel
import os, requests, faiss, numpy as np, pickle
from openai import OpenAI
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware

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

client = OpenAI(api_key=os.environ["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"], base_url="https://api.groq.com/openai/v1")

def embed(text):
    r = client.embeddings.create(
        model="nomic-embed-text",
        input=text
    )
    return np.array(r.data[0].embedding, dtype="float32")

groq = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

class ProcessRequest(BaseModel):
    uid: str
    url: str

class AskRequest(BaseModel):
    uid: str
    question: str


# ------------------------------
# PROCESS PDF
# ------------------------------
@app.post("/process_pdf")
def process_pdf(req: ProcessRequest):
    uid = req.uid
    os.makedirs(f"user_data/{uid}", exist_ok=True)

    pdf_path = f"user_data/{uid}/doc.pdf"
    r = requests.get(req.url)
    open(pdf_path, "wb").write(r.content)

    reader = PdfReader(pdf_path)
    chunks = []
    metadata = []

    # chunk per page
    for pageno, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i in range(0, len(text), 800):
            chunk = text[i:i+800]
            chunks.append(chunk)
            metadata.append({"page": pageno})

    vectors = embedding_model.encode(chunks)
    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(384)
    index.add(vectors)

    faiss.write_index(index, f"user_data/{uid}/vectors.index")
    pickle.dump({"chunks": chunks, "metadata": metadata},
                open(f"user_data/{uid}/chunks.pkl", "wb"))

    return {"status": "PDF embedded successfully"}


# ------------------------------
# ASK QUESTION
# ------------------------------
@app.post("/ask")
def ask(req: AskRequest):
    uid = req.uid
    index = faiss.read_index(f"user_data/{uid}/vectors.index")
    data = pickle.load(open(f"user_data/{uid}/chunks.pkl", "rb"))

    chunks = data["chunks"]
    metadata = data["metadata"]

    query_vec = embedding_model.encode(req.question).reshape(1, -1)
    query_vec = query_vec.astype("float32")
    faiss.normalize_L2(query_vec)

    scores, idxs = index.search(query_vec, 3)

    context = ""
    for idx in idxs[0]:
        context += f"[Page {metadata[idx]['page']}]: {chunks[idx]}\n\n"

    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer ONLY using context."},
            {"role": "user", "content": f"Context:\n{context}\nQuestion:\n{req.question}"}
        ]
    )

    return {"answer": completion.choices[0].message.content}


# ------------------------------
# DELETE USER DATA
# ------------------------------
@app.delete("/delete_user")
def delete_user(uid: str):
    path = f"user_data/{uid}"
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    return {"status": "User data deleted"}



