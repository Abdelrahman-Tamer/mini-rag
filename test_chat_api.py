import requests, json

BASE = "http://localhost:5000/api/v1"

def chat(text, limit=3, session_id=None, city=None, doc_type=None):
    body = {"text": text, "limit": limit}
    if session_id: body["session_id"] = session_id
    if city: body["city"] = city
    if doc_type: body["doc_type"] = doc_type
    r = requests.post(f"{BASE}/nlp/chat", json=body, timeout=60)
    print(f"  DEBUG: status={r.status_code} len={len(r.text)} text={repr(r.text[:500])}", flush=True)
    if not r.text:
        return {"error": "empty response"}
    return r.json()

# Test 1: New chat (no session)
print("=== 1. New chat (no session) ===")
res = chat("What is RAG?")
print(f"Signal: {res.get('signal')}")
print(f"Message: {res.get('message')}")
print(f"Session ID: {res.get('session_id')}")
print(f"Answer (first 200): {res.get('answer', '')[:200]}")
print(f"Sources: {len(res.get('sources', []))} items")
if res.get('sources'):
    s = res['sources'][0]
    print(f"  - document_name: {s.get('document_name')}")
    print(f"  - city: {s.get('city')}")
    print(f"  - score: {s.get('score')}")
print(f"Related questions: {res.get('related_questions')}")
print()

# Test 2: Follow-up with session
print("=== 2. Follow-up with session ===")
sid = res.get('session_id')
res2 = chat("What about Italian options?", session_id=sid)
print(f"Signal: {res2.get('signal')}")
print(f"Message: {res2.get('message')}")
print(f"Answer (first 200): {res2.get('answer', '')[:200]}")
print(f"Related questions: {res2.get('related_questions')}")
print()

# Test 3: No results
print("=== 3. No results ===")
res3 = chat("xyzzy_nonexistent_12345")
print(f"Signal: {res3.get('signal')}")
print(f"Message: {res3.get('message')}")
print(f"Session ID: {res3.get('session_id')}")
print()

# Test 4: City filter
print("=== 4. City filter ===")
res4 = chat("What is there to do?", limit=3, city="cairo")
print(f"Signal: {res4.get('signal')}")
print(f"Sources: {len(res4.get('sources', []))} items")
for s in res4.get('sources', []):
    print(f"  - {s.get('document_name')} (city={s.get('city')}, score={s.get('score')})")
print()

print("=== ALL TESTS DONE ===")
