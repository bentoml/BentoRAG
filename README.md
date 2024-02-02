## setup virtual environment

```bash
python3 -m venv venv && source venv/bin/activate && pip install "bentoml>=1.2.0rc1" llama-index openllm-client
```

## setup Sentence Embeddings and LLM service urls:

First copy `conf_tmpl.py` as `conf.py`, then deploy sentence embeddings and OpenLLM service. Fill their urls in `conf.py`
