from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import structlog

from config import ConfigAssistant


@dataclass(frozen=True)
class ManualPassage:
    text: str
    page: int
    source: str
    similarity: float


class VehicleManualManager:
    def __init__(self, opt: ConfigAssistant):
        if opt.rag_chunk_words <= opt.rag_chunk_overlap_words or opt.rag_chunk_overlap_words < 0:
            raise ValueError("RAG chunk size must exceed its non-negative overlap")
        if opt.rag_top_k < 1 or opt.rag_context_max_chars < 1 or opt.rag_timeout <= 0:
            raise ValueError("RAG result count, context budget and timeout must be positive")
        if not 0 <= opt.rag_min_similarity <= 1:
            raise ValueError("RAG minimum similarity must be between zero and one")
        root = Path(__file__).resolve().parents[1]
        self.opt = opt
        self.manual_path = root / opt.rag_manual_directory / opt.rag_manual_filename
        self.index_path = root / opt.rag_index_directory
        self.logger = structlog.get_logger()
        self._collection: Any = None
        self._prepare_lock = asyncio.Lock()

    def _read_chunks(self) -> tuple[str, list[dict[str, Any]]]:
        import pymupdf

        fingerprint = hashlib.sha256(self.manual_path.read_bytes()).hexdigest()
        chunks: list[dict[str, Any]] = []
        step = self.opt.rag_chunk_words - self.opt.rag_chunk_overlap_words
        with pymupdf.open(self.manual_path) as document:
            for page_number, page in enumerate(document, start=1):
                words = page.get_text(sort=True).split()
                for start in range(0, len(words), step):
                    chunks.append({
                        "text": " ".join(words[start:start + self.opt.rag_chunk_words]),
                        "page": page_number,
                        "source": self.manual_path.name,
                    })
                    if start + self.opt.rag_chunk_words >= len(words):
                        break
        if not chunks:
            raise ValueError("Manual has no extractable text; run OCR before indexing it")
        return fingerprint, chunks

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        timeout = aiohttp.ClientTimeout(total=self.opt.rag_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"http://{self.opt.ollama_host}:{self.opt.ollama_port}/api/embed",
                json={"model": self.opt.rag_embedding_model, "input": texts, "truncate": False},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Ollama embedding request failed (HTTP {response.status}); "
                        f"check the server and run ollama pull {self.opt.rag_embedding_model}"
                    )
                payload = await response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("Ollama returned an unexpected number of embeddings")
        for embedding in embeddings:
            if (
                not isinstance(embedding, list)
                or not embedding
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding)
                or not any(value != 0 for value in embedding)
            ):
                raise ValueError("Ollama returned an invalid embedding")
        return embeddings

    def _open_collection(self, name: str) -> Any:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(self.index_path), settings=Settings(anonymized_telemetry=False)
        )
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    async def prepare(self) -> None:
        async with self._prepare_lock:
            if self._collection is not None:
                return
            fingerprint, chunks = await asyncio.to_thread(self._read_chunks)
            identity = (
                f"v1:{fingerprint}:{self.manual_path.name}:{self.opt.rag_embedding_model}:"
                f"{self.opt.rag_chunk_words}:{self.opt.rag_chunk_overlap_words}"
            )
            name = "manual_" + hashlib.sha256(identity.encode()).hexdigest()[:40]
            collection = await asyncio.to_thread(self._open_collection, name)
            if await asyncio.to_thread(collection.count) != len(chunks):
                for start in range(0, len(chunks), 16):
                    batch = chunks[start:start + 16]
                    embeddings = await self._embed([chunk["text"] for chunk in batch])
                    await asyncio.to_thread(
                        collection.upsert,
                        ids=[str(start + offset) for offset in range(len(batch))],
                        documents=[chunk["text"] for chunk in batch],
                        metadatas=[{"page": chunk["page"], "source": chunk["source"]} for chunk in batch],
                        embeddings=embeddings,
                    )
            self._collection = collection
            self.logger.info("Vehicle manual index ready", source=self.manual_path.name, chunks=len(chunks))

    async def retrieve(self, query: str) -> list[ManualPassage]:
        if not query.strip():
            return []
        await self.prepare()
        embedding = await self._embed([query])
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=embedding,
            n_results=min(self.opt.rag_top_k, await asyncio.to_thread(self._collection.count)),
            include=["documents", "metadatas", "distances"],
        )
        return [
            ManualPassage(text=text, page=metadata["page"], source=metadata["source"], similarity=1 - distance)
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
            if 1 - distance >= self.opt.rag_min_similarity
        ]

    async def context_for(self, question: str, history: list[dict[str, str]]) -> str:
        previous_questions = [item["content"] for item in history if item["role"] == "user"][-1:]
        query = "\n".join([*previous_questions, question])[-6000:]
        try:
            passages = await self.retrieve(query)
        except Exception as error:
            self.logger.warning("Vehicle manual retrieval unavailable", error=str(error))
            return "Manual retrieval is unavailable. No verified manual passages are available."
        blocks: list[str] = []
        remaining = self.opt.rag_context_max_chars
        for passage in passages:
            block = f"Source: {passage.source}; PDF page {passage.page}\n{passage.text}"
            separator_size = 2 if blocks else 0
            if len(block) + separator_size > remaining:
                continue
            blocks.append(block)
            remaining -= len(block) + separator_size
        context = "\n\n".join(blocks)
        self.logger.info("Vehicle manual context retrieved", passages=len(blocks), characters=len(context))
        return context or "No sufficiently relevant manual passages were found."