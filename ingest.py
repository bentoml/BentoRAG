import bentoml
from tqdm import tqdm
import concurrent.futures

import os
from pathlib import Path


# url = "https://rag-service-wbow-e3c1c7db.mt-guc1.bentoml.ai"
url = input("Please input your bento deployment url (http://0.0.0.0:3000 for local deployment): ")

def ingest(client, file) -> str:
    file_name = file.split('.')[0]
    file = "./example_pdfs/" + file
    return client.ingest(pdf=file, pdf_name=file_name)

def helper(iter):
    return ingest(iter[0], iter[1])

results = []
with bentoml.SyncHTTPClient(url, timeout=3000) as client:
    pool_iter = []
    for file in os.listdir("./example_pdfs"):
        if file.endswith(".pdf"):
            pool_iter.append([client, file])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(helper, pool_iter), total=len(pool_iter), desc="Ingesting pdfs using BentoOCR service"))
        
print(results)
    






    