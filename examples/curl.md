# Tidemark with curl

```bash
URL=http://127.0.0.1:18200
AUTH="Authorization: Token change-me"

# Store messages (returns once they are durable and searchable)
curl -s $URL/add -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "request_id": "req-1", "user_id": "alice", "session_id": "s1",
  "messages": [{"role": "user", "timestamp": 1683554160000,
                "content": "Alice: I adopted a cat named Pixel yesterday!"}]}'

# Search
curl -s $URL/search -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "query": "When did I adopt my cat?", "top_k": 5}'

# List, get, correct, history, delete (record IDs contain "#": URL-encode them as %23)
curl -s "$URL/memories?user_id=alice" -H "$AUTH"
ID='...'   # an "id" from the list, URL-encoded
curl -s "$URL/memories/$ID?user_id=alice" -H "$AUTH"
curl -s -X PATCH "$URL/memories/$ID" -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "content": "Alice: I adopted a cat named Pixel on 7 May 2023."}'
curl -s "$URL/memories/$ID/history?user_id=alice" -H "$AUTH"
curl -s -X DELETE "$URL/memories/$ID?user_id=alice" -H "$AUTH"

# Delete everything stored for a user
curl -s -X DELETE "$URL/users/alice" -H "$AUTH"
```
