import urllib.request
import json
import uuid

url = 'http://127.0.0.1:8000/api/chat'
data = {
    'message': 'search for information on async playwright',
    'success_criteria': 'just give a summary',
    'thread_id': str(uuid.uuid4())
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'))
req.add_header('Content-Type', 'application/json')

try:
    with urllib.request.urlopen(req) as response:
        for line in response:
            line_str = line.decode('utf-8').strip()
            if line_str:
                print(line_str)
except Exception as e:
    print(f"Error: {e}")
