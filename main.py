from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
import asyncio
import json

app = FastAPI()

GROQ_API_KEY = "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"

async def stream_response(chat_completion):
    for chunk in chat_completion:
        yield chunk.choices[0].delta.content or ""
        await asyncio.sleep(0.05)  # Simulate streaming delay

@app.post("/query")
async def process_query(query: dict):
    text = query.get('query')
    image = query.get('image')

    if not text:
        raise HTTPException(status_code=400, detail="Query text is required")

    client = Groq(api_key=GROQ_API_KEY)

    if image:
        # Image + text query
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
            stream=True
        )
    else:
        # Text-only query
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="llama-3.1-70b-versatile",
            stream=True
        )

    return StreamingResponse(stream_response(chat_completion), media_type="text/plain")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
