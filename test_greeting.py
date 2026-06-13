import requests, json

BASE = "http://localhost:5000/api/v1"

def chat(text, limit=3, session_id=None, city=None, doc_type=None):
    body = {"text": text, "limit": limit}
    if session_id: body["session_id"] = session_id
    if city: body["city"] = city
    if doc_type: body["doc_type"] = doc_type
    r = requests.post(f"{BASE}/nlp/chat", json=body, timeout=60)
    return r.json()

# Test greeting
print("=== Greeting test ===")
res = chat("HI")
print(f"Query: HI")
print(f"Signal: {res.get('signal')}")
print(f"Answer: {res.get('answer', '')[:200]}")
print(f"Sources: {len(res.get('sources', []))}")
print(f"Related: {res.get('related_questions')}")
print(f"Session ID: {res.get('session_id')}")
print()

# Test normal query still works
print("=== Normal query test ===")
res2 = chat("restaurants in cairo")
print(f"Signal: {res2.get('signal')}")
print(f"Answer: {res2.get('answer', '')[:200]}")
print(f"Sources: {len(res2.get('sources', []))}")
print()

# Test Arabic greeting
print("=== Arabic greeting test ===")
res3 = chat("مرحبا")
print(f"Signal: {res3.get('signal')}")
print(f"Answer: {res3.get('answer', '')[:200]}")
print(f"Sources: {len(res3.get('sources', []))}")
print()

print("DONE")
