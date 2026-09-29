"""FastAPI service.  Run: uvicorn api:app --reload"""
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from graph import ask, build_graph


@asynccontextmanager
async def lifespan(_: FastAPI):
    build_graph()  # compile once at startup
    yield


app = FastAPI(title="Agentic AI eBook RAG Chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, examples=["What is Agentic AI?"])


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return ask(req.query.strip())
    except Exception as e:  # surface upstream (OpenAI/Pinecone) failures cleanly
        raise HTTPException(status_code=502, detail=f"RAG pipeline error: {e}")