from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model import load_model_and_tokenizer, extract_data


llm_store = {}

class ModelRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_store["model"], llm_store["tokenizer"] = load_model_and_tokenizer()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/extract")
async def extract_info(model_request: ModelRequest):
    model = llm_store["model"]
    tokenizer = llm_store["tokenizer"]
    
    result = extract_data(model, tokenizer, text =  model_request.text)
    print(result)
    return {"traction": result}
