import bentoml

# url = "https://rag-service-wbow-e3c1c7db.mt-guc1.bentoml.ai"
url = input("Please input your bento deployment url (http://0.0.0.0:3000 for local deployment): ")
user_input = input("Please ask a question about Sherlock: ")
api_key = input("Please paste your openai key: ")

while user_input:
    with bentoml.SyncHTTPClient(url, timeout=300) as client:
        response = client.query(user_input, api_key)
        print(response)
        user_input = input("Please ask a question about Sherlock: ")
