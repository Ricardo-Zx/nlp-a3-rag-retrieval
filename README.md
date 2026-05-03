# NLP Assignment 3: Hybrid Sparse-Dense Retrieval for RAG

Group project repository for UTS NLP Assessment 3.

## Team Members

- Yuchang Zhang (25678259)
- [Name] ([Student ID])
- [Name] ([Student ID])
- [Name] ([Student ID])

## Project Scope

This project evaluates retrieval-stage improvements for Retrieval-Augmented Generation (RAG)
pipelines by comparing three retrievers:

- BM25
- Dense Retrieval
- Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

The main experiment uses the SQuAD v1.1 validation set and includes:

- Main retrieval comparison
- Chunking strategy comparison
- Embedding model comparison
- Error analysis

## Repository Contents

- `rag_retrieval_comparison.ipynb`: main notebook containing code, results, and figures

## Dependencies

The notebook installs the following packages in Colab:

- `rank_bm25`
- `sentence-transformers`
- `faiss-cpu`
- `datasets`
- `matplotlib`
- `seaborn`

## Usage

Open `rag_retrieval_comparison.ipynb` in Google Colab or Jupyter and run the cells in order.
