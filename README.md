# VERITAS-Graph: Self-Auditing Bi-Temporal RAG & Multi-Agent Memory Governance

**VERITAS-Graph** is an enterprise AI memory and provenance engine that detects semantic contradictions across evolving documents, repairs outdated knowledge graph nodes, and enforces deterministic guardrails against rogue agent mutations.

## Key Capabilities
1. **Unstructured Document Ingestion (IBM Docling):** Parses enterprise PDFs/DOCX into structured markdown and atomic fact nodes.
2. **Bi-Temporal Knowledge Graph (NetworkX + ChromaDB):** Tracks [valid_from, valid_to] lifecycles and SUPERSEDED_BY provenance edges.
3. **PyTorch NLI Contradiction Engine (DeBERTa-v3):** Classifies incoming facts against active memories into Contradiction, Entailment, or Neutral.
4. **Multi-Agent Memory Repair (CrewAI + IBM Granite-3.0 2B SLM):** Orchestrates a Temporal Provenance Auditor and Rogue-Agent Safety Controller locally via Ollama.

## Controlled Ablation Study Results
Evaluated on an 8-pair enterprise policy evolution benchmark (benchmark_ablation.py):

| Architecture | Conflict F1 | Precision / Recall | Stale Memory Rate | Avg Ingest+Audit Latency |
| :--- | :---: | :---: | :---: | :---: |
| **1. Baseline RAG (ChromaDB Vector Only)** | 0.00 | 0.00 / 0.00 | 100.0% | 14.2 ms |
| **2. VERITAS-Graph (Temporal KG + DeBERTa NLI)** | **0.91** | **0.83 / 1.00** | **0.0%** | **273.9 ms** |

## Quickstart
    pip install -r requirements.txt
    ollama pull granite3-dense:2b
    python veritas_core.py
    python benchmark_ablation.py
    python veritas_agents.py
