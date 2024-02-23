# BentoRAG: Llama-index RAG Implementation with PDF Inputs

### Deploy Service Locally
```bash
python -m venv bento_rag
source bento_rag/bin/activate
pip install -r requirements.txt
bentoml serve
```

### Ingest PDF
`python ingest.py`

### Query
`python query.py`