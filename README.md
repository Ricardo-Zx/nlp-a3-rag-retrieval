# Improving RAG Retrieval Quality via Hybrid Sparse-Dense Retrieval

NLP Assignment 3 — Group Project, University of Technology Sydney

## Overview

This project investigates retrieval quality in RAG (Retrieval-Augmented Generation) pipelines. Poor retrieval is a key bottleneck in RAG systems - even a strong language model cannot recover from a wrong document. We propose **Hybrid RRF** (Reciprocal Rank Fusion of BM25 + Dense retrieval) as a practical solution, and validate it against baselines on the SQuAD v1.1 benchmark.

## Experiments

| Block | Description | Key Finding |
|-------|-------------|-------------|
| **Block 1** | BM25 vs Dense vs Hybrid RRF (main experiment) | Hybrid best on all metrics |
| **Block 2** | Chunking strategy (100 / 200 / 400 words) | Larger chunks slightly better for SQuAD |
| **Block 3** | Embedding model comparison (MiniLM / mpnet / bge-small) | bge-small best Recall@1 and MRR; mpnet-base highest Recall@10 but much slower |
| **Block 4** | Per-question error analysis | BM25 vocabulary mismatch accounts for most failures |

## Main Results

Evaluated on full SQuAD v1.1 validation set (10,570 questions, 2,067 documents):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|--------|----------|----------|-----------|-----|
| BM25 | 0.6026 | 0.7764 | 0.8289 | 0.6787 |
| Dense | 0.6251 | 0.8640 | 0.9172 | 0.7270 |
| **Hybrid RRF** | **0.6700** | **0.8749** | **0.9305** | **0.7574** |

## Setup

```bash
pip install rank_bm25 sentence-transformers faiss-cpu datasets matplotlib seaborn
```

Or open directly in Google Colab - all dependencies are installed in the first cell.

## Dataset

[SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) - loaded automatically via HuggingFace `datasets`.

## Team Members

- Yuchang Zhang (25678259)
- [Name] ([Student ID])
- [Name] ([Student ID])
- [Name] ([Student ID])
