from fastapi import FastAPI
from pydantic import BaseModel

from app.services.vectorstore import VectorStore
from app.services.rag import RAG
from app.services.llm import generate_answer
from app.index.builder import build_index_async

app = FastAPI()

store = VectorStore()
rag = RAG(store)

chat_history = {}

# -------------------------
# STARTUP EVENT (ASYNC INDEX)
# -------------------------
@app.on_event("startup")
async def startup_event():
    global store, rag

    store = await build_index_async()
    rag = RAG(store)

# -------------------------
# REQUEST MODEL
# -------------------------
class ChatRequest(BaseModel):
    sessionId: str
    message: str

# -------------------------
# CHAT ENDPOINT
# -------------------------
@app.post("/api/chat")
def chat(req: ChatRequest):

    history = chat_history.get(req.sessionId, [])

    results = rag.retrieve(req.message)

    if not results:
        return {
            "reply": "No relevant context found.",
            "tokensUsed": 0,
            "retrievedChunks": 0
        }

    context = "\n".join([r[1]["text"] for r in results])

    answer = generate_answer(context, history, req.message)

    history.append((req.message, answer))
    chat_history[req.sessionId] = history[-5:]

    return {
        "reply": answer,
        "tokensUsed": 0,
        "retrievedChunks": len(results)
    }

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}