import torch
import numpy as np
import pandas as pd
import networkx as nx
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Optional IBM Docling import (graceful fallback if not installed yet)
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class VeritasGraphEngine:
    def __init__(self, confidence_threshold: float = 0.75):
        print("[1/3] Loading SentenceTransformer Embeddings (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="veritas_temporal_memory")
        self.graph = nx.DiGraph()
        
        print("[2/3] Loading PyTorch DeBERTa-v3 NLI Contradiction Model...")
        model_name = "cross-encoder/nli-deberta-v3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nli_model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.audit_log = []
        print("[3/3] VERITAS-Graph Engine Ready!\n")

    def parse_pdf_with_docling(self, file_path: str) -> str:
        """Uses IBM Research's Docling library to extract structured markdown from PDFs/DOCX."""
        if not DOCLING_AVAILABLE or not Path(file_path).exists():
            return ""
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()

    def check_contradiction(self, old_fact: str, new_fact: str) -> dict:
        inputs = self.tokenizer([old_fact], [new_fact], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        # cross-encoder/nli-deberta-v3-small label mapping: 0=Contradiction, 1=Entailment, 2=Neutral
        contradiction_prob = float(probs[0])
        return {
            "is_contradiction": contradiction_prob > 0.50,
            "confidence": contradiction_prob,
            "probs": {
                "contradiction": round(float(probs[0]), 4),
                "entailment": round(float(probs[1]), 4),
                "neutral": round(float(probs[2]), 4)
            }
        }

    def ingest_fact(self, fact_id: str, text: str, timestamp: str, source_doc: str, topic: str):
        active_nodes = [
            n for n, attr in self.graph.nodes(data=True)
            if attr.get("topic") == topic and attr.get("status") == "ACTIVE"
        ]
        
        for old_id in active_nodes:
            old_text = self.graph.nodes[old_id]["text"]
            nli_result = self.check_contradiction(old_text, text)
            
            if nli_result["is_contradiction"]:
                if nli_result["confidence"] < self.confidence_threshold:
                    self.audit_log.append({
                        "action": "GUARDRAIL_BLOCKED_ROGUE_OVERWRITE",
                        "deprecated_id": old_id,
                        "new_id": fact_id,
                        "confidence": round(nli_result["confidence"], 3),
                        "reason": "Uncertainty threshold triggered — queued for CrewAI / Human review."
                    })
                else:
                    self.graph.nodes[old_id]["status"] = "DEPRECATED"
                    self.graph.nodes[old_id]["valid_to"] = timestamp
                    self.graph.add_edge(old_id, fact_id, relation="SUPERSEDED_BY", confidence=nli_result["confidence"])
                    self.audit_log.append({
                        "action": "VERITAS_AUTO_REPAIRED",
                        "deprecated_id": old_id,
                        "new_id": fact_id,
                        "confidence": round(nli_result["confidence"], 3),
                        "reason": f"Superseded by {source_doc} on {timestamp}"
                    })

        self.graph.add_node(
            fact_id, text=text, topic=topic, source=source_doc,
            valid_from=timestamp, valid_to="PRESENT", status="ACTIVE"
        )
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            ids=[fact_id], embeddings=[embedding], documents=[text],
            metadatas=[{"topic": topic, "timestamp": timestamp, "source": source_doc}]
        )

    def retrieve_baseline_rag(self, query: str, top_k: int = 2) -> list:
        """Simulates ordinary RAG (ignores temporal validity and returns stale facts)."""
        q_emb = self.embedder.encode(query).tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return [{"id": fid, **self.graph.nodes[fid]} for fid in res["ids"][0]]

    def retrieve_veritas_verified(self, query: str, top_k: int = 2) -> list:
        """VERITAS-Graph Retrieval: Filters out DEPRECATED memories using Temporal Graph state."""
        q_emb = self.embedder.encode(query).tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return [
            {"id": fid, **self.graph.nodes[fid]}
            for fid in res["ids"][0]
            if self.graph.nodes[fid]["status"] == "ACTIVE"
        ]


if __name__ == "__main__":
    engine = VeritasGraphEngine(confidence_threshold=0.75)

    # 1. Ingest January Policy (v1)
    engine.ingest_fact(
        fact_id="POL_JAN_01",
        text="Employees are allowed to work remotely 4 days per week without manager approval.",
        timestamp="2026-01-10",
        source_doc="IBM_HR_Policy_v1.pdf",
        topic="remote_work"
    )

    # 2. Ingest March Policy Update (v2 - Contradicts January Policy)
    engine.ingest_fact(
        fact_id="POL_MAR_02",
        text="Remote work is strictly prohibited. All employees must work from the office 5 days a week.",
        timestamp="2026-03-15",
        source_doc="IBM_HR_Policy_v2.pdf",
        topic="remote_work"
    )

    print("=== 1. VERITAS-GRAPH SELF-AUDIT LOG ===")
    print(pd.DataFrame(engine.audit_log).to_string(index=False))

    query = "What is the remote work policy for employees?"
    print(f"\n=== 2. ORDINARY RAG RETRIEVAL (Returns Stale Memory!) ===")
    for item in engine.retrieve_baseline_rag(query):
        print(f" -> [{item['status']}] ({item['source']}): {item['text']}")

    print(f"\n=== 3. VERITAS-GRAPH VERIFIED RETRIEVAL (Self-Healed!) ===")
    for item in engine.retrieve_veritas_verified(query):
        print(f" -> [{item['status']}] ({item['source']} | Valid from {item['valid_from']}): {item['text']}")
