from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import ollama

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_with_model(request: PromptRequest):
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': request.prompt}
        ])
        return {"response": response['message']['content']}
    except ollama.ResponseError as e:
        raise HTTPException(status_code=500, detail=str(e))
