from typing import Any, Coroutine, List
from llama_index.core.embeddings.base import Embedding
from llama_index.embeddings.base import BaseEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index import set_global_service_context
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.schema import Document
import os
# import openai
# import bentoml.models
from bentoml.client import Client
from llama_index.llms import OpenLLMAPI
from llama_index.text_splitter import SentenceSplitter
import bentoml




class BentoMLEmbeddings(BaseEmbedding):
    url = "https://sentence-embedding-yatai.bc-staging.bentoml.ai"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def sync_client(self, query: str):
        response = {}
        with bentoml.SyncHTTPClient(self.url) as client:
            if isinstance(query, List): response = client.encode(sentences=query)
            else: response = client.encode(sentences=[query])  
        return response
    
    async def async_client(self, query: str):
        response = {}
        async with bentoml.AsyncHTTPClient(self.url) as client:
            if isinstance(query, List): response = client.encode(sentences=query)
            else: response = client.encode(sentences=[query])  
        return response
    
    def _aget_query_embedding(self, query: str):
        return self.async_client(query)[0]
    
    def _get_query_embedding(self, query: str):
        return self.sync_client(query)[0]
    
    def _get_text_embedding(self, text):
        if isinstance(text, str): return self.sync_client(text)
        else: return self.sync_client(text[0].get_text())
    
    def _get_text_embeddings(self, text):
        if isinstance(text, str) or isinstance(text[0], str): return self.sync_client(text)
        else: return self.sync_client(text[0].get_text())

embed_model = BentoMLEmbeddings()

embeddings = embed_model.get_query_embedding("Hello World")
# print(embeddings)
documents = SimpleDirectoryReader("data").load_data()
# print(documents[0].get_text())
# embeddings = embed_model.get_text_embedding(documents)
# print(embeddings)

# openai.api_key = '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'


text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
remote_llm = OpenLLMAPI(address="https://mixtral-8x7b-gptq-test-3-yatai.bc-staging.bentoml.ai")
embed_context = ServiceContext.from_defaults(embed_model=embed_model)
# Create a ServiceContext with the custom model and all the configurations
service_context = ServiceContext.from_defaults(
    llm=remote_llm,
    embed_model=embed_model,
    text_splitter=text_splitter,
    context_window=8192,
    num_output=4096,
)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("What did Paul do?")
print(response)
