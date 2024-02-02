from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenLLMAPI
from llama_index.text_splitter import SentenceSplitter
import bentoml

import conf
from embeddings import BentoMLEmbeddings


if __name__ == "__main__":
    embed_model = BentoMLEmbeddings()
    documents = SimpleDirectoryReader("data").load_data()

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    remote_llm = OpenLLMAPI(address=conf.OPENLLM_URL)
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

    user_input = input("Please ask a question about Paul: ")
    while user_input:
        response = query_engine.query(user_input)
        print(response)
        user_input = input("Please ask a question about Paul: ")
