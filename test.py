import requests

url = "http://127.0.0.1:8000/process-pdf/"
file_path = "/home/darshil-thakkar-929/Desktop/Projects/LegalDoc-Translate-Query-Assistant/Documents/test.pdf"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
