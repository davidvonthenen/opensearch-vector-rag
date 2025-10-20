PYTHON ?= python
INDEX ?= bbc
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION ?= Tell me about the connection between Ernie Wise and Vodafone.

.PHONY: ingest serve query env

ingest:
	$(PYTHON) -m src.ingest_bbc --data-dir $(DATA_DIR) --index-name $(INDEX)

serve:
	$(PYTHON) -m src.server

query:
	$(PYTHON) -m src.client --question "$(QUESTION)"

env:
@echo "OPENSEARCH_HOST=$${OPENSEARCH_HOST:-127.0.0.1}"
@echo "OPENSEARCH_PORT=$${OPENSEARCH_PORT:-9200}"
@echo "OPENSEARCH_INDEX=$${OPENSEARCH_INDEX:-$(INDEX)}"
@echo "EMBEDDING_MODEL_NAME=$${EMBEDDING_MODEL_NAME:-thenlper/gte-small}"
@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-./models/llama.gguf}"
@echo "LLAMA_CTX=$${LLAMA_CTX:-8192}"
@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(os.cpu_count() or 1)')}"
@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:--1}"
@echo "RAG_TOP_K=$${RAG_TOP_K:-5}"
@echo "RAG_NUM_CANDIDATES=$${RAG_NUM_CANDIDATES:-50}"
@echo "SERVER_HOST=$${SERVER_HOST:-$(HOST)}"
@echo "SERVER_PORT=$${SERVER_PORT:-$(PORT)}"
