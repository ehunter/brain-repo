from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    natural_language: bool = Field(False, description="Use natural language response instead of structured output")


class Match(BaseModel):
    score: float
    row_index: int
    text: str
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    matches: List[Match]
    follow_up_suggestions: Optional[List[str]] = None
    data_insights: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    detail: str


# New models for structured queries

class ProductModel(BaseModel):
    id: int
    name: str
    description: str
    category: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class StructuredQueryRequest(BaseModel):
    category: Optional[str] = Field(None, description="Filter by category (partial match)")
    name_contains: Optional[str] = Field(None, description="Filter by name containing text")
    description_contains: Optional[str] = Field(None, description="Filter by description containing text")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")
    explain: bool = Field(False, description="Include natural language explanation of results")


class StructuredQueryResponse(BaseModel):
    products: List[ProductModel]
    total_count: int
    query_summary: str
    explanation: Optional[str] = None


class DatabaseStatsResponse(BaseModel):
    total_products: int
    categories: List[str]
    database_path: str


# Brain research data models

class BrainSpecimenModel(BaseModel):
    id: int
    subject_id: str
    repository: Optional[str] = None
    race: Optional[str] = None
    subject_sex: Optional[str] = None
    subject_age: Optional[int] = None
    ethnicity: Optional[str] = None
    neuropathology_diagnosis: Optional[str] = None
    clinical_brain_diagnosis: Optional[str] = None
    manner_of_death: Optional[str] = None
    brain_region: Optional[str] = None
    pmi_hours: Optional[float] = None
    rin: Optional[float] = None
    preparation: Optional[str] = None
    created_at: Optional[str] = None


class BrainQueryRequest(BaseModel):
    # Identifiers
    subject_id: Optional[str] = Field(None, description="Filter by subject ID (partial match)")

    # Demographics
    race: Optional[str] = Field(None, description="Filter by race (partial match)")
    subject_sex: Optional[str] = Field(None, description="Filter by sex (Male/Female)")
    age_min: Optional[int] = Field(None, ge=0, le=150, description="Minimum age")
    age_max: Optional[int] = Field(None, ge=0, le=150, description="Maximum age")
    ethnicity: Optional[str] = Field(None, description="Filter by ethnicity")

    # Clinical
    manner_of_death: Optional[str] = Field(None, description="Filter by manner of death")
    diagnosis_contains: Optional[str] = Field(None, description="Search in diagnosis fields")
    repository: Optional[str] = Field(None, description="Filter by repository")

    # Tissue quality
    pmi_max: Optional[float] = Field(None, ge=0, description="Maximum post-mortem interval (hours)")
    rin_min: Optional[float] = Field(None, ge=0, le=10, description="Minimum RNA integrity number")

    # Brain regions
    brain_region_contains: Optional[str] = Field(None, description="Search in brain regions")
    hemisphere: Optional[str] = Field(None, description="Filter by hemisphere")

    # Pagination and explanation
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")
    explain: bool = Field(False, description="Include natural language explanation of results")


class BrainQueryResponse(BaseModel):
    specimens: List[BrainSpecimenModel]
    total_count: int
    query_summary: str
    explanation: Optional[str] = None
    follow_up_suggestions: Optional[List[str]] = None
    data_insights: Optional[List[str]] = None


class BrainStatsResponse(BaseModel):
    total_specimens: int
    races: List[str]
    sexes: List[str]
    repositories: List[str]
    average_age: Optional[float] = None
    database_path: str
