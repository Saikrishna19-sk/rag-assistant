from fastapi import FastAPI
from pydantic import BaseModel
import json, os
from dotenv import load_dotenv
load_dotenv()
from app.utils.chunker import chunk_text
from app.services.embedder import get_embedding
from app.services.vectorstore import VectorStore
from app.services.llm import generate_answer

app = FastAPI()
store = VectorStore()
chat_history = {}

class ChatRequest(BaseModel):
    sessionId: str
    message: str

@app.on_event("startup")
def startup_event():
    with open("data/docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        chunks = chunk_text(doc["content"], 50)
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            store.add(emb, {"title": doc["title"], "chunk_id": i, "text": chunk})

@app.post("/api/chat")
def chat(req: ChatRequest):
    history = chat_history.get(req.sessionId, [])
    query_emb = get_embedding(req.message)
    results = store.search(query_emb, top_k=3)
    if not results:
        return {"reply": "No relevant context found.", "tokensUsed": 0, "retrievedChunks": 0}
    context = "\n".join([r[1]["text"] for r in results])
    answer = generate_answer(context, history, req.message)
    history.append((req.message, answer))
    chat_history[req.sessionId] = history[-5:]
    return {"reply": answer, "tokensUsed": 0, "retrievedChunks": len(results)}

@app.get("/health")
def health():
    return {"status": "healthy"}
