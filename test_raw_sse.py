import requests

url = "http://127.0.0.1:8000/api/chat"
headers = {"Content-Type": "application/json"}
data = {"message": "hi", "success_criteria": "just answer"}

try:
    with requests.post(url, json=data, stream=True) as r:
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                print("RAW CHUNK:", repr(chunk))
except Exception as e:
    print("Error:", e)
