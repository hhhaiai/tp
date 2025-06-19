# tp
free api

## build test use code

``` bash
docker build --no-cache --compress -t tpp .
docker run -p 7860:7860 -m 2g -e DEBUG=false tpp




docker run -d --restart always  --name tp  -p 7860:7860 ghcr.io/hhhaiai/tp:latest


curl http://127.0.0.1:7860/v1/models


curl -X POST http://localhost:7860/api/v1/chat/completions \
  -H 'Accept: application/json' \
  -H 'Authorization: Bearer sk-lx' \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7,
    "max_tokens": 150,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }'

```

