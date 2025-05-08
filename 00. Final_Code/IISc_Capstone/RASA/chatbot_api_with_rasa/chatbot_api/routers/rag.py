# rag_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.rag_service import run_rag_pipeline

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    sources: List[str]

@router.post("/rag_query", response_model=QueryResponse)
def rag_query(req: QueryRequest):
    try:
        result = run_rag_pipeline(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
