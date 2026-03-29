# Chat API Specification

This document describes the API contract used by the chatbot client in `src/chat/api/chatApiClient.js`.

## Endpoint

- **Method:** `POST`
- **URL:** configured via runtime `apiUrl`
- **Default URL:** `https://salemai-assistant.web.app/api/chat`
- **Headers:**
  - `Content-Type: application/json`
- **Timeout:** `12000ms`

## Request Body

```json
{
  "message": "User latest message",
  "history": [
    { "role": "assistant", "text": "Welcome message..." },
    { "role": "user", "text": "Hi" },
    { "role": "assistant", "text": "Hello! How can I help?" }
  ]
}
```

### Request Fields

- `message` (`string`, required): the latest user message.
- `history` (`array`, required): chronological chat history.
  - item shape: `{ "role": "user|assistant", "text": "..." }`

## Accepted Response Shape

The client accepts only this JSON response format:

```json
{
  "reply": "Your response text here"
}
```

If `reply` is missing, empty, or not a string, the client treats it as an error and shows the service-unavailable fallback message.

## cURL Example

```bash
curl -X POST "https://salemai-assistant.web.app/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What sectors does Salem Balhamer operate in?",
    "history": [
      { "role": "assistant", "text": "Hello. Ask about our sectors, latest news, or general enquiries — we are glad to help." },
      { "role": "user", "text": "Hi" },
      { "role": "assistant", "text": "Hello! How can I help?" }
    ]
  }'
```

## Error Handling Expectations

- Return a non-2xx HTTP status for server-side failures.
- Return JSON for success responses.
- Ensure response includes one valid non-empty `reply` string.

