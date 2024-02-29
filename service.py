from __future__ import annotations

import bentoml

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

import os
import numpy as np
from pathlib import Path
from typing import Annotated
import openai


from openai_utils import openai_deco, _make_httpx_client


LLM_MAX_TOKENS = 4096
LLM_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

@openai_deco(served_model=LLM_MODEL_ID)
@bentoml.service(
    traffic={
        "timeout": 600,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(
            model=LLM_MODEL_ID,
            max_model_len=LLM_MAX_TOKENS
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)


@bentoml.service(
    traffic={"timeout": 600, "max_concurrency": 8},
)
class OCRService:
        
    @bentoml.api
    def ingest_pdf(self, pdf: Annotated[Path, bentoml.validators.ContentType("application/pdf")]) -> str:
        from pdf2image import convert_from_path

        pages = convert_from_path(pdf)
        extracted_text = ''
        for page in pages:
            preprocessed_image = self.deskew(np.array(page))
            text = self.extract_text_from_image(preprocessed_image)
            extracted_text += text
    
        return extracted_text

    def deskew(self, image):
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def extract_text_from_image(self, image):
        import pytesseract
        text = pytesseract.image_to_string(image)
        return text


import typing as t

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

@bentoml.service(
    traffic={"timeout": 600},
    resources={"memory": "2Gi"},
)
class SentenceTransformers:

    def __init__(self) -> None:

        import torch
        from sentence_transformers import SentenceTransformer, models
        
        # Load model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # define layers
        first_layer = SentenceTransformer(MODEL_ID)
        pooling_model = models.Pooling(first_layer.get_sentence_embedding_dimension())
        self.model = SentenceTransformer(modules=[first_layer, pooling_model])
        print("Model loaded, ", "device:", self.device)

    @bentoml.api(batchable=True)
    def encode(
        self,
        sentences: t.List[str],
    ) -> np.ndarray:
        print("encoding sentences:", len(sentences))
        # Tokenize sentences
        sentence_embeddings= self.model.encode(sentences)
        return sentence_embeddings


class BentoMLEmbeddings(BaseEmbedding):
    _model: bentoml.Service = PrivateAttr()

    def __init__(self, embed_model: bentoml.Service, **kwargs) -> None:
        self._model = embed_model
        super().__init__(**kwargs)
        
    def sync_client(self, query: str):

        response = {}
        if isinstance(query, list):
            response = self._model.encode(sentences=query)
        else:
            response = self._model.encode(sentences=[query])  
        return response
    
    async def async_client(self, query: str):
        response = {}
        if isinstance(query, list):
            response = await self._model.encode(sentences=query)
        else:
            response = await self._model.encode(sentences=[query])  
        return response
    
    async def _aget_query_embedding(self, query: str):
        res = await self.async_client(query)
        return res[0]
    
    def _get_query_embedding(self, query: str):
        return self.sync_client(query)[0]
    
    def _get_text_embedding(self, text):
        if isinstance(text, str):
            return self.sync_client(text).tolist()
        else:
            return self.sync_client(text[0].get_text()).tolist()

    def _get_text_embeddings(self, text):
        if isinstance(text, str) or isinstance(text[0], str):
            # print(self.sync_client(text))
            return self.sync_client(text).tolist()
        else:
            return self.sync_client(text[0].get_text())




@bentoml.service(
    traffic={"timeout": 600},
)
class RAGService:
    ocr_service = bentoml.depends(OCRService)
    embedding_service = bentoml.depends(SentenceTransformers)
    vllm_service = bentoml.depends(VLLM)

    def __init__(self):
        from llama_index.core import Settings

        self.api_key = ""
        if not os.path.exists("./RAGData"):
            os.mkdir("RAGData")

        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        self.index = None
        self.query_engine = None
        self.embed_model = BentoMLEmbeddings(self.embedding_service)

        from bentoml._internal.container import BentoMLContainer
        self.vllm_url = BentoMLContainer.remote_runner_mapping.get()["VLLM_OpenAI"]

        # Configure Llama-index Global Settings
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.text_splitter


    @bentoml.api
    def ingest(self, pdf: Annotated[Path, bentoml.validators.ContentType("application/pdf")], pdf_name: str) -> str:
        from llama_index.core import Settings
        Settings.embed_model = self.embed_model

        extracted_text = self.ocr_service.ingest_pdf(pdf)
        destination = f'RAGData/{pdf_name}.txt'
        with open(destination, "w") as txt_file:    
            txt_file.write(extracted_text)
        
        documents = SimpleDirectoryReader("RAGData").load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.index.storage_context.persist(persist_dir="./storage")
        return f"Successfully Loaded Document: {destination}"
    
    @bentoml.api
    def ingest_text(self, texts, file_name):
        print(texts)
        with open("./RAGData/" + file_name, "w") as txt_file:
            txt_file.write(texts)

        documents = SimpleDirectoryReader("RAGData").load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.index.storage_context.persist(persist_dir="./storage")
        return f"Successfully Loaded Document: {destination}"

    @bentoml.api
    def query(self, query: str) -> str:
        from transformers import AutoTokenizer
        from llama_index.core import Settings

        Settings.num_output = 256
        Settings.context_window = LLM_MAX_TOKENS
        Settings.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_ID
        )

        httpx_client, base_url = _make_httpx_client(self.vllm_url, VLLM)
        llm = OpenAILike(
            api_base= base_url + "/v1/",
            api_key="no-need",
            is_chat_model=True,
            http_client=httpx_client,
            temperature=0.2,
            model=LLM_MODEL_ID,
        )
        Settings.llm = llm

        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = load_index_from_storage(storage_context)
        self.query_engine = self.index.as_query_engine()
        response = self.query_engine.query(query)
        return str(response)
    
    @bentoml.api
    async def encode(
        self,
        sentences: t.List[str],
    ) -> np.ndarray:
        print("encoding sentences:", len(sentences))
        # Tokenize sentences
        sentence_embeddings = await self.embedding_model.encode(sentences)
        return sentence_embeddings
    

