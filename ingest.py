"""Ingestion pipeline: PDF -> pages -> chunks -> OpenAI embeddings -> Pinecone.

Usage:
    python ingest.py                       # uses data/Ebook-Agentic-AI.pdf or $PDF_URL
    python ingest.py --url https://.../Ebook-Agentic-AI.pdf
    python ingest.py --pdf path/to/file.pdf
"""
import argparse
import hashlib
import os
import time

import requests
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

import config

BATCH = 100


def download_pdf(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    return dest


def load_pages(pdf_path: str) -> list[Document]:
    """One Document per non-empty page, carrying its 1-based page number."""
    reader = PdfReader(pdf_path)
    source = os.path.basename(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i, "source": source}))
    print(f"Parsed {len(docs)} non-empty pages from {source}")
    return docs


def chunk_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def ensure_index(pc: Pinecone):
    if config.PINECONE_INDEX not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{config.PINECONE_INDEX}'")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
            time.sleep(2)
    return pc.Index(config.PINECONE_INDEX)


def chunk_id(doc: Document) -> str:
    # Deterministic ID -> re-running ingestion is idempotent (upsert, no duplicates).
    h = hashlib.sha1(f"{doc.metadata['page']}|{doc.page_content}".encode()).hexdigest()[:16]
    return f"p{doc.metadata['page']}-{h}"


def ingest(pdf_path: str) -> int:
    chunks = chunk_pages(load_pages(pdf_path))
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    index = ensure_index(Pinecone(api_key=config.PINECONE_API_KEY))

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        vectors = embeddings.embed_documents([c.page_content for c in batch])
        index.upsert(
            vectors=[
                {
                    "id": chunk_id(c),
                    "values": v,
                    "metadata": {
                        "text": c.page_content,
                        "page": c.metadata["page"],
                        "source": c.metadata["source"],
                    },
                }
                for c, v in zip(batch, vectors)
            ],
            namespace=config.PINECONE_NAMESPACE,
        )
        print(f"  upserted {min(start + BATCH, len(chunks))}/{len(chunks)}")
    return len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="local PDF path")
    ap.add_argument("--url", help="URL to download the PDF from")
    args = ap.parse_args()

    pdf_path = args.pdf or config.DEFAULT_PDF_PATH
    url = args.url or config.PDF_URL
    if not os.path.exists(pdf_path):
        if not url:
            raise SystemExit(
                f"PDF not found at {pdf_path}. Pass --url / set PDF_URL, or place the file there."
            )
        download_pdf(url, pdf_path)
    n = ingest(pdf_path)
    print(f"Done. {n} chunks indexed in '{config.PINECONE_INDEX}/{config.PINECONE_NAMESPACE}'.")


if __name__ == "__main__":
    main()
