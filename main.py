from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from nu_extract import load_model_and_tokenizer, predict_nuextract


llm_store = {}

class NuextractRequest(BaseModel):
    nu_schema: dict
    text: str
    example: list[dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_store["model"], llm_store["tokenizer"] = load_model_and_tokenizer()
    yield
    llm_store.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/extract")
async def extract_info(nuextract_request: NuextractRequest):
    model = llm_store["model"]
    tokenizer = llm_store["tokenizer"]
    
    result = predict_nuextract(model, tokenizer, text =  nuextract_request.text, nu_schema = nuextract_request.nu_schema, example = nuextract_request.example)
    print(result)
    return {"traction": result}
