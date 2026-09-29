"""LangGraph RAG workflow.

    START -> retrieve --(no relevant chunks)--> refuse -> END
                 |
                 v
             generate <----------------------+
                 |                            | (grounding < threshold
                 v                            |  and attempts left)
               grade --------------------------+
                 |
                 v
             finalize -> END
"""
from __future__ import annotations

from functools import lru_cache
from statistics import mean
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field

import config

NOT_FOUND = "I could not find this information in the Agentic AI eBook."
LOW_GROUNDING = (
    "I could not produce an answer that is sufficiently supported by the Agentic AI eBook."
)


class RAGState(TypedDict, total=False):
    query: str
    chunks: List[dict]        # {"text", "page", "score"}
    retrieval_score: float    # mean similarity of kept chunks
    answer: str
    attempts: int
    feedback: str             # unsupported claims from the grader, fed to regeneration
    grounding_score: float
    refused: bool
    confidence: float


class GroundingGrade(BaseModel):
    score: float = Field(ge=0, le=1, description="Fraction of claims in the answer supported by the context")
    unsupported_claims: List[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _resources():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return {
        "embeddings": OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        "index": pc.Index(config.PINECONE_INDEX),
        "llm": ChatOpenAI(model=config.LLM_MODEL, temperature=0),
    }


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about the Agentic AI eBook using ONLY the numbered context passages below.\n"
     "Rules:\n"
     "- Do not use outside knowledge, even if you know the answer.\n"
     "- If the context does not contain the answer, reply exactly: " + NOT_FOUND + "\n"
     "- Be concise and cite pages like (p. 4) where useful.\n"
     "{feedback}\n\nContext:\n{context}"),
    ("human", "{query}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Given CONTEXT and an ANSWER, list every factual claim in the "
     "answer that is NOT directly supported by the context, and give score = supported claims / "
     "total claims (0 to 1). Ignore citations and phrasing; judge only factual support."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


def _format_context(chunks: List[dict]) -> str:
    return "\n\n".join(f"[{i}] (page {c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1))


# ---------------------------------------------------------------- nodes
def retrieve(state: RAGState) -> RAGState:
    r = _resources()
    vec = r["embeddings"].embed_query(state["query"])
    res = r["index"].query(
        vector=vec, top_k=config.TOP_K, include_metadata=True,
        namespace=config.PINECONE_NAMESPACE,
    )
    chunks = [
        {"text": m.metadata["text"], "page": int(m.metadata.get("page", 0)), "score": float(m.score)}
        for m in res.matches
        if m.score >= config.MIN_RELEVANCE
    ]
    return {
        "chunks": chunks,
        "retrieval_score": mean(c["score"] for c in chunks) if chunks else 0.0,
        "attempts": 0,
        "feedback": "",
    }


def generate(state: RAGState) -> RAGState:
    fb = state.get("feedback", "")
    feedback = (
        "Your previous answer contained unsupported claims. Remove or correct these:\n- "
        + "\n- ".join(fb.split("\n")) if fb else ""
    )
    msg = (GENERATE_PROMPT | _resources()["llm"]).invoke({
        "context": _format_context(state["chunks"]),
        "query": state["query"],
        "feedback": feedback,
    })
    return {"answer": msg.content.strip(), "attempts": state.get("attempts", 0) + 1}


def grade(state: RAGState) -> RAGState:
    if state["answer"].startswith(NOT_FOUND):
        return {"refused": True, "grounding_score": 0.0}
    grader = _resources()["llm"].with_structured_output(GroundingGrade)
    g: GroundingGrade = (GRADE_PROMPT | grader).invoke({
        "context": _format_context(state["chunks"]), "answer": state["answer"],
    })
    return {"grounding_score": g.score, "feedback": "\n".join(g.unsupported_claims), "refused": False}


def refuse(state: RAGState) -> RAGState:
    return {"answer": NOT_FOUND, "chunks": [], "grounding_score": 0.0, "confidence": 0.0, "refused": True}


def finalize(state: RAGState) -> RAGState:
    if state.get("refused"):
        return {"answer": NOT_FOUND, "chunks": [], "confidence": 0.0}
    g = state["grounding_score"]
    if g < config.MIN_GROUNDING:  # still ungrounded after retries -> don't risk a hallucination
        return {"answer": LOW_GROUNDING, "confidence": round(g, 2)}
    # Heuristic: 70% answer grounding (LLM-judged) + 30% retrieval similarity.
    return {"confidence": round(0.7 * g + 0.3 * state["retrieval_score"], 2)}


# -------------------------------------------------------------- routing
def after_retrieve(state: RAGState) -> str:
    return "generate" if state["chunks"] else "refuse"


def after_grade(state: RAGState) -> str:
    if state.get("refused") or state["grounding_score"] >= config.GROUNDING_THRESHOLD:
        return "finalize"
    return "generate" if state["attempts"] < config.MAX_GENERATION_ATTEMPTS else "finalize"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RAGState)
    for name, fn in [("retrieve", retrieve), ("generate", generate), ("grade", grade),
                     ("refuse", refuse), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "finalize": "finalize"})
    g.add_edge("refuse", END)
    g.add_edge("finalize", END)
    return g.compile()


def ask(query: str) -> dict:
    """Run the graph and return the required response payload."""
    out = build_graph().invoke({"query": query})
    return {
        "query": query,
        "final_answer": out["answer"],
        "retrieved_context_chunks": [c["text"] for c in out.get("chunks", [])],
        "confidence_score": out["confidence"],
    }
