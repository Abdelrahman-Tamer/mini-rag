import sys, traceback
sys.path.insert(0, '/mnt/c/Users/Mohamed Ahmed/Desktop/mini-rag/src')
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
r = client.post("/api/v1/nlp/chat", json={"text": "hello", "limit": 3})
print(f"Status: {r.status_code}")
print(f"Body: {r.text[:500]}")
