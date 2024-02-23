from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openllm import OpenLLMAPI
from llama_index.core.node_parser import SentenceSplitter
import bentoml
from llama_index.core import Settings

import os
from embeddings import BentoMLEmbeddings

OPENLLM_URL = os.environ.get('OPENLLM_ENDPOINT')

if __name__ == "__main__":
    embed_model = BentoMLEmbeddings()
    documents = SimpleDirectoryReader("data").load_data()

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    remote_llm = OpenLLMAPI(OPENLLM_URL)
    # Create a ServiceContext with the custom model and all the configurations
    # service_context = ServiceContext.from_defaults(
    #     llm=remote_llm,
    #     embed_model=embed_model,
    #     text_splitter=text_splitter,
    #     context_window=8192,
    #     num_output=4096,
    # )
    Settings.embed_model = embed_model
    Settings.node_parser = text_splitter
    Settings.num_output = 4096
    Settings.context_window = 8192
    Settings.llm = remote_llm
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    user_input = input("Please ask a question about Paul: ")
    while user_input:
        response = query_engine.query(user_input)
        print(response)
        user_input = input("Please ask a question about Paul: ")

# import os.path
# from llama_index.core import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     StorageContext,
#     load_index_from_storage,
# )

# # check if storage already exists
# PERSIST_DIR = "./storage"
# if not os.path.exists(PERSIST_DIR):
#     # load the documents and create the index
#     documents = SimpleDirectoryReader("data").load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     # store it for later
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
# else:
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#     index = load_index_from_storage(storage_context)

# # Either way we can now query the index
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)