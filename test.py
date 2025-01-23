import requests
import json

# Define API endpoint
url = "http://127.0.0.1:8000/process-pdf/"

# File path to the PDF
file_path = "/home/darshil-thakkar-929/Desktop/Projects/LegalDoc-Translate-Query-Assistant/Documents/test.pdf"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        # Make POST request to the API
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Pretty-print the response JSON
        print("Response:")
        print(json.dumps(response.json(), indent=4))
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
