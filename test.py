import requests
import json

url = "http://127.0.0.1:8000/translate-pdf/"

# File path to the PDF
file_path = "/home/darshil-thakkar-929/Desktop/Projects/LegalDoc-Translate-Query-Assistant/Documents/test.pdf"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        # Make POST request to the API
        response = requests.post(url, files=files)
        response.raise_for_status()

        print("Response:")
        print(json.dumps(response.json(), indent=4))
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")


url = "http://127.0.0.1:8000/query/"

# User query
user_query = "What were the key legal arguments in the case?"

# Request payload
payload = {"user_query": user_query}

try:
    # Make POST request to the API
    response = requests.post(url, json=payload)
    response.raise_for_status()

    print("Response:")
    print(json.dumps(response.json(), indent=4))
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
