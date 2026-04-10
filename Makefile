PYTHON ?= python
INDEX ?= bbc-vector-chunks
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION1 ?= How much did OpenAI purchase Windsurf for?
QUESTION2 ?= How much did Google purchase Windsurf for?

.PHONY: ingest agent server query client env

ingest:
	$(PYTHON) -m src.1_ingest.ingest

agent:
	$(PYTHON) -m src.2_rag_agent.server

server: agent

query:
	$(PYTHON) -m src.3_rag_client.client --question "$(QUESTION1)"
	$(PYTHON) -m src.3_rag_client.client --question "$(QUESTION2)"

client: query

env:
	@echo "OPENSEARCH_HOST=$${OPENSEARCH_HOST:-127.0.0.1}"
	@echo "OPENSEARCH_PORT=$${OPENSEARCH_PORT:-9201}"
	@echo "OPENSEARCH_INDEX=$${OPENSEARCH_INDEX:-$(INDEX)}"
	@echo "EMBEDDING_MODEL_NAME=$${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-./models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf}"
	@echo "LLAMA_CTX=$${LLAMA_CTX:-65536}"
	@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(os.cpu_count() or 1)')}"
	@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:-20}"
	@echo "RAG_TOP_K=$${RAG_TOP_K:-3}"
	@echo "RAG_NUM_CANDIDATES=$${RAG_NUM_CANDIDATES:-50}"
	@echo "SERVER_HOST=$${SERVER_HOST:-$(HOST)}"
	@echo "SERVER_PORT=$${SERVER_PORT:-$(PORT)}"
