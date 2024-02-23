import bentoml

url = "http://0.0.0.0:3000"
user_input = input("Please ask a question about Paul: ")
while user_input:
    with bentoml.SyncHTTPClient(url, timeout=300) as client:
        response = client.query(user_input)
        print(response)
        user_input = input("Please ask a question about Sherlock: ")