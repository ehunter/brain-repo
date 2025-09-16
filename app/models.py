from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class LoadResponse(BaseModel):
    message: str
    rows: int
    columns: List[str]
    text_columns: List[str]
    metadata_columns: List[str]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=20)


class Match(BaseModel):
    score: float
    row_index: int
    text: str
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    matches: List[Match]


class ErrorResponse(BaseModel):
    detail: str
