
# Building a Simple RAG Application with Llama-Index and BentoML
LlamaIndex provides a comprehensive framework for managing and retrieving private and domain-specific data. It acts as a bridge between the extensive knowledge of LLMs and the unique, contextual data needs of specific applications.

BentoML is a framework for building reliable, scalable, and cost-efficient AI applications. It comes with everything you need for model serving, application packaging, and production deployment.

We have already finished the RAG application in this repo. However, we will walk you through step by step on how to create a RAG application with Llama-Index and BentoML. This tutorial consists of three parts
1. Quickstart on the usage of this repository
2. Customize LlamaIndex Embedding Model with BentoML and Sentence Transformers
3. Customize LlamaIndex LLM Model with BentoML and Mistral 7B

## Quickstart

### Clone this repository using git
`git clone https://github.com/bentoml/BentoRAG.git`

### setup virtual environment

```bash
python3 -m venv venv && source venv/bin/activate && pip install "bentoml>=1.2.0rc1" llama-index openllm-client
```

### setup Sentence Embeddings and LLM service urls:

First copy `conf_tmpl.py` as `conf.py`, then deploy sentence embeddings and OpenLLM service. Fill their urls in `conf.py`

### Start asking questions about Paul Graham!

```bash
python 01_starter.py
```

Ask your questions!

## Customize LlamaIndex Embedding Model with BentoML and Sentence Transformers

Now we've seen the functionality of this RAG application, let's start building it from scratch.

By default, Llama-Index uses the OpenAI gpt-3.5-turbo endpoint as the embedding model to index documents and queries. However, we can also achieve this by replacing OpenAI with BentoML, offering more flexibility over embedding models. In this example, we will be using a pre-built SBert Sentence Transformers BentoML service to achieve this goal. The steps are as followed:

1. Make a new directory you would like to realize this tutorial `mkdir <dir_name>`
2. cd in to the directory, then clone the Sentence Transformers BentoML service repository `git clone <https://github.com/bentoml/BentoSentenceTransformers.git`>
3. Copy the data directory to your newly created directory. This folder contains a text file that we want to index later for the RAG application.
4. Deploy the service following the steps
- `cd BentoSentenceTransformers`
- Create a python virutal environment `python -m venv Llama-BentoML`
- Activate the environment `source Llama-BentoML/bin/activate`
- Install required packages `pip install -r requirement.txt`
- Start the service by one simple command `bentoml serve`

Now the Sentence Embedding service is up and running, you can try to test the service by sendding request via curl. There is a detailed guide README file under BentoSentenceTransformers directory.

### Replacing OpenAI endpoints by BentoML

Make a new python file inside your directory you created for this tutorial. Copy paste the following code into the file.

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenLLMAPI
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings.base import BaseEmbedding
import bentoml

import conf

import bentoml

class BentoMLEmbeddings(BaseEmbedding):
    url = "http://localhost:3000"
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
```

The above code builds a BentoML embedding model class from the Llama-Index BaseEmbedding abstract class. It embeds documents and queries by posting requests to the up and running BentoML service you just initialized. To test, simply run the following 3-liner by openning up a new terminal:

```python
embed_model = BentoMLEmbeddings()
embeddings = embed_model.get_query_embedding("Hello World")
print(embeddings)

```

You can also embed a txt file. Make a directory called `data`, then create a new text file with random text you would prefer. Then run the following code:

```python
documents = SimpleDirectoryReader("data").load_data()
embeddings = embed_model.get_text_embedding(documents)
print(embeddings)

```

Now you should see the embeddings printed on your terminal.

## Deploying BentoML service onto BentoCloud

The current BentoML service is running on local host. We can deploy the service onto the BentoCloud platform with one simple command. However, make sure you are logged into BentoCloud first. The steps are as followed:

1. Create a BentoCloud account via https://default.cloud-staging.bentoml.com/ .
2. Once you are logged in, you should see the following tabs on the left hand side. Navigate to My API Tokens. 
    
    ![截屏2024-01-31 14.12.06.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ca09ac1-c034-487e-836d-e6939c220392/975fe1db-864f-4d54-a766-c93611533f12/%E6%88%AA%E5%B1%8F2024-01-31_14.12.06.png)
    
3. Create a API Token following the instructions [here](https://docs.bentoml.com/en/1.2/bentocloud/how-tos/manage-access-token.html#create-an-api-token).
4. After you created the API Token, you should see a log in command auto-generated. Copy and paste the command into your terminal. You are now logged into BentoCloud on your machine.
5. Run `bentoml deploy .` to deploy the service. This process can take up to 10+ mins.
6. Once you have finished deploying, you should see your service listed on BentoCloud under the Overview tab. 
    
    ![截屏2024-01-31 14.17.24.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ca09ac1-c034-487e-836d-e6939c220392/c68383d7-d7eb-40f8-a1b0-f7962897d82b/%E6%88%AA%E5%B1%8F2024-01-31_14.17.24.png)
    
7. Clicked into your service, you can see all the availble options. To simply make a request to the service, copy the url of your service, and replace the local host url you originally had inside the python script. You should be able to see the same output after you run the code.
    
    ![截屏2024-01-31 14.20.54.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ca09ac1-c034-487e-836d-e6939c220392/23e894a0-8946-472d-bfc1-992e78d5c6fb/%E6%88%AA%E5%B1%8F2024-01-31_14.20.54.png)
    
## Customize LlamaIndex LLM Model with BentoML and Mistral 7B**

Now we have replaced the default embedding model, let’s move on to LLM. Since Llama-Index has built-in support for OpenLLM (a LLM service platform built by BentoML), fewer steps are needed comparing to replacing the embedding model. 

### Deploying Mistral 7B Service onto BentoCloud

First, let’s deploy a LLM service. 

1. On [BentoCloud](https://default.cloud-staging.bentoml.com/) (which you should already have a account), hover your mouse to Mistral 7B, click deploy.
    
    ![截屏2024-02-02 14.46.19.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ca09ac1-c034-487e-836d-e6939c220392/76869fb1-4a86-4c0b-bd5e-aeef58601530/%E6%88%AA%E5%B1%8F2024-02-02_14.46.19.png)
    
2. Inside the deployment page, all the hardware specifications are already pre-generated. Change the “Deployment Name” under the “Deployment Config” tab to `mistralai--mistral-7b-instruct-service-pzq3`
    
    ![截屏2024-02-02 14.48.53.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ca09ac1-c034-487e-836d-e6939c220392/863079e4-bfb6-466e-9a76-3317cf692753/%E6%88%AA%E5%B1%8F2024-02-02_14.48.53.png)
    
3. Click on the “Submit” button on the bottom right of the page. The deployment can take a few minutes. Congratulations! now you have deployed the LLM service onto BentoCloud using Mistral 7B.

### Using Mistral 7B as the LLM service for Llama-Index

Now let’s use the service to finalize our RAG application. The following 2 liner will create a tokenizer and a LLM service recognized by Llama-Index

```python
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
remote_llm = OpenLLMAPI(address=conf.OPENLLM_URL)
```

Together with the `embed_model` we have created in the previous section, we can now fill in the service context for our RAG application

```python
service_context = ServiceContext.from_defaults(
        llm=remote_llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        context_window=8192,
        num_output=4096,
    )
```

Pass the service_context to Llama-Index so the application will use our newly created `embed_model` and `remote_llm` to create a index. Now send a query just like how you would normally do using the default Llama-Index syntax.

```python
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    user_input = input("Please ask a question about Paul: ")
    while user_input:
        response = query_engine.query(user_input)
        print(response)
        user_input = input("Please ask a question about Paul: ")
```

Run the python code and ask you questions. You should see a answer generated on the terminal. Congratulations! You just finished building a complete RAG application by customizing embedding and LLM models using BentoML!
