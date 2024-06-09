from fastapi import FastAPI
import httpx
import time

app = FastAPI()

OLLAMA_HOST = "ollama"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

@app.post("/ask")
async def ask_question(question: str):
    payload = {
        "model": "mistral",
        "prompt": question,
        "stream": False
    }

    timeout = httpx.Timeout(600.0, connect=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            start = time.time()
            response = await client.post(OLLAMA_BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            end = time.time() - start
            print("elapsed time: ", end)
            return {"response": data}
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail="Failed to get response from Ollama")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"An error occurred while requesting Ollama: {exc}")
