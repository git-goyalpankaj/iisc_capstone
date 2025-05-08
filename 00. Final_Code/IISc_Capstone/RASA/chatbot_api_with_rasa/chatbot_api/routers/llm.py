from fastapi import APIRouter
from models.schemas import TextInput, LLMResponse
from services.llm_service import query_llm

router = APIRouter()

@router.post("/", response_model=LLMResponse)
def llm_route(input: TextInput):
    response = query_llm(input.text)
    return {"response": response}