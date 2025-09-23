from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    AskRequest, AskResponse, ErrorResponse, LoadResponse, Match,
    StructuredQueryRequest, StructuredQueryResponse, DatabaseStatsResponse,
    ProductModel, BrainQueryRequest, BrainQueryResponse, BrainStatsResponse,
    BrainSpecimenModel
)
from app.services.brain_bank import BrainBank, BrainBankError
from app.services.llm_service import LLMService, LLMServiceError
from app.services.database import DatabaseService, QueryFilter
from app.services.brain_database import BrainDatabaseService, BrainQueryFilter

app = FastAPI(title="Brain Bank CSV Chat Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

brain_bank = BrainBank()
llm_service = LLMService()
db_service = DatabaseService()
brain_db_service = BrainDatabaseService()


@app.on_event("startup")
async def startup_event():
    """Initialize databases with CSV data on startup."""
    # Load product data
    csv_path = Path(__file__).resolve().parent.parent / "sample_data.csv"
    if csv_path.exists():
        try:
            db_service.load_csv_data(csv_path)
            print(f"✅ Product database initialized from {csv_path}")
        except Exception as e:
            print(f"❌ Failed to initialize product database: {e}")

    # Load brain research data
    brain_csv_path = Path(__file__).resolve().parent.parent / "sample_repo_data.csv"
    if brain_csv_path.exists():
        try:
            brain_db_service.load_csv_data(brain_csv_path)
            print(f"✅ Brain database initialized from {brain_csv_path}")
        except Exception as e:
            print(f"❌ Failed to initialize brain database: {e}")
    else:
        print(f"❌ Brain CSV file not found: {brain_csv_path}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown."""
    db_service.close()
    brain_db_service.close()


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(str(index_path))


@app.post("/load_csv", response_model=LoadResponse, responses={400: {"model": ErrorResponse}})
async def load_csv(
    file: UploadFile = File(...),
    text_columns: Optional[str] = Form(None),
    metadata_columns: Optional[str] = Form(None),
) -> LoadResponse:
    file_bytes = await file.read()
    text_cols = _parse_columns(text_columns)
    metadata_cols = _parse_columns(metadata_columns)

    try:
        status = brain_bank.load_csv(
            file_bytes,
            text_columns=text_cols,
            metadata_columns=metadata_cols,
        )
    except BrainBankError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return LoadResponse(
        message="CSV loaded successfully.",
        rows=status.rows,
        columns=status.columns,
        text_columns=status.text_columns,
        metadata_columns=status.metadata_columns,
    )


@app.get("/status", response_model=LoadResponse, responses={404: {"model": ErrorResponse}})
async def status() -> LoadResponse:
    try:
        status = brain_bank.status()
    except BrainBankError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return LoadResponse(
        message="CSV currently loaded.",
        rows=status.rows,
        columns=status.columns,
        text_columns=status.text_columns,
        metadata_columns=status.metadata_columns,
    )


@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}})
async def ask(request: AskRequest) -> AskResponse:
    try:
        # For natural language responses, retrieve more matches for better accuracy
        # The LLM service will intelligently chunk based on token limits
        # For structured responses, use the user-specified top_k
        if request.natural_language:
            status = brain_bank.status()
            # Retrieve a substantial portion but let LLM service handle chunking
            top_k = min(status.rows, 1000)  # Cap at 1000 to avoid excessive retrieval
        else:
            top_k = request.top_k

        result = brain_bank.ask(request.question, top_k=top_k)
    except BrainBankError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    converted_matches = [convert_numpy_types(match) for match in result["matches"]]
    matches = [Match(**match) for match in converted_matches]

    # Generate natural language response if requested and available
    if request.natural_language and llm_service.is_available:
        try:
            natural_answer = llm_service.generate_natural_response(
                request.question, result["matches"]
            )
            return AskResponse(answer=natural_answer, matches=matches)
        except LLMServiceError as e:
            # Check for specific quota/billing issues
            error_msg = str(e).lower()
            print(f"DEBUG: LLMServiceError caught: {e}")  # Debug logging
            if "quota" in error_msg or "insufficient_quota" in error_msg or "rate limit" in error_msg:
                structured_answer = result["answer"]
                quota_notice = "⚠️ Natural language responses are temporarily unavailable due to OpenAI API quota limits. Please check your OpenAI billing settings. Here are the structured results:\n\n"
                return AskResponse(answer=quota_notice + structured_answer, matches=matches)
            else:
                # Fall back to structured response for other LLM errors
                structured_answer = result["answer"]
                error_notice = f"⚠️ Natural language responses are temporarily unavailable (OpenAI API error: {str(e)}). Here are the structured results:\n\n"
                return AskResponse(answer=error_notice + structured_answer, matches=matches)
    elif request.natural_language and not llm_service.is_available:
        # Inform user that natural language is not available
        structured_answer = result["answer"]
        unavailable_notice = "Natural language responses are not available (OpenAI API key not configured). Here are the structured results:\n\n"
        return AskResponse(answer=unavailable_notice + structured_answer, matches=matches)

    return AskResponse(answer=result["answer"], matches=matches)


def _parse_columns(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    columns = [col.strip() for col in raw_value.split(",") if col.strip()]
    return columns or None


# New structured query endpoints

@app.get("/products", response_model=StructuredQueryResponse, responses={400: {"model": ErrorResponse}})
async def query_products(
    category: Optional[str] = None,
    name_contains: Optional[str] = None,
    description_contains: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    explain: bool = False,
) -> StructuredQueryResponse:
    """Fast structured query endpoint for products with optional LLM explanation."""
    try:
        # Create filter object
        filters = QueryFilter(
            category=category,
            name_contains=name_contains,
            description_contains=description_contains,
            limit=limit,
            offset=offset
        )

        # Execute SQL query
        result = db_service.query_products(filters)

        # Convert to Pydantic models
        products = [ProductModel(**product) for product in result.products]

        response = StructuredQueryResponse(
            products=products,
            total_count=result.total_count,
            query_summary=result.query_summary,
            explanation=None
        )

        # Optional LLM explanation
        if explain and llm_service.is_available and result.products:
            try:
                # Format products for LLM
                products_text = "\n".join([
                    f"- {p['name']}: {p['description']} (Category: {p['category']})"
                    for p in result.products
                ])

                prompt = f"""Based on the following query results, provide a brief, natural language summary:

Query: {result.query_summary}

Results:
{products_text}

Provide a 1-2 sentence summary explaining what was found and any notable patterns."""

                explanation = llm_service._client.chat.completions.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7,
                )
                response.explanation = explanation.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM explanation failed: {e}")
                # Don't fail the whole request if explanation fails

        return response

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/products/query", response_model=StructuredQueryResponse, responses={400: {"model": ErrorResponse}})
async def query_products_post(request: StructuredQueryRequest) -> StructuredQueryResponse:
    """POST version of structured query for complex filters."""
    try:
        # Create filter object
        filters = QueryFilter(
            category=request.category,
            name_contains=request.name_contains,
            description_contains=request.description_contains,
            limit=request.limit,
            offset=request.offset
        )

        # Execute SQL query
        result = db_service.query_products(filters)

        # Convert to Pydantic models
        products = [ProductModel(**product) for product in result.products]

        response = StructuredQueryResponse(
            products=products,
            total_count=result.total_count,
            query_summary=result.query_summary,
            explanation=None
        )

        # Optional LLM explanation
        if request.explain and llm_service.is_available and result.products:
            try:
                # Format products for LLM
                products_text = "\n".join([
                    f"- {p['name']}: {p['description']} (Category: {p['category']})"
                    for p in result.products
                ])

                prompt = f"""Based on the following query results, provide a brief, natural language summary:

Query: {result.query_summary}

Results:
{products_text}

Provide a 1-2 sentence summary explaining what was found and any notable patterns."""

                explanation = llm_service._client.chat.completions.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7,
                )
                response.explanation = explanation.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM explanation failed: {e}")

        return response

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/database/stats", response_model=DatabaseStatsResponse)
async def get_database_stats() -> DatabaseStatsResponse:
    """Get database statistics and metadata."""
    try:
        stats = db_service.get_stats()
        return DatabaseStatsResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Brain research endpoints

@app.get("/brain/specimens", response_model=BrainQueryResponse, responses={400: {"model": ErrorResponse}})
async def query_brain_specimens(
    race: Optional[str] = None,
    subject_sex: Optional[str] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    manner_of_death: Optional[str] = None,
    diagnosis_contains: Optional[str] = None,
    repository: Optional[str] = None,
    pmi_max: Optional[float] = None,
    rin_min: Optional[float] = None,
    brain_region_contains: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    explain: bool = False,
) -> BrainQueryResponse:
    """Fast structured query endpoint for brain research specimens with optional LLM explanation."""
    try:
        # Create filter object
        filters = BrainQueryFilter(
            race=race,
            subject_sex=subject_sex,
            age_min=age_min,
            age_max=age_max,
            manner_of_death=manner_of_death,
            diagnosis_contains=diagnosis_contains,
            repository=repository,
            pmi_max=pmi_max,
            rin_min=rin_min,
            brain_region_contains=brain_region_contains,
            limit=limit,
            offset=offset
        )

        # Execute SQL query
        result = brain_db_service.query_specimens(filters)

        # Convert to Pydantic models
        specimens = [BrainSpecimenModel(**specimen) for specimen in result.specimens]

        response = BrainQueryResponse(
            specimens=specimens,
            total_count=result.total_count,
            query_summary=result.query_summary,
            explanation=None
        )

        # Optional LLM explanation
        if explain and llm_service.is_available and result.specimens:
            try:
                # Format specimens for LLM (limit to first 10 for brevity)
                sample_specimens = result.specimens[:10]
                specimens_text = "\n".join([
                    f"- Subject {s['subject_id']}: {s['subject_sex']}, {s['subject_age']} years old, {s['race']}, {s['manner_of_death']}"
                    for s in sample_specimens if s.get('subject_id')
                ])

                prompt = f"""Based on the following brain research query results, provide a brief, natural language summary:

Query: {result.query_summary}

Sample Results:
{specimens_text}

Provide a 1-2 sentence summary of the research cohort and any notable patterns in demographics or clinical characteristics."""

                explanation = llm_service._client.chat.completions.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7,
                )
                response.explanation = explanation.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM explanation failed: {e}")
                # Don't fail the whole request if explanation fails

        return response

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/brain/specimens/query", response_model=BrainQueryResponse, responses={400: {"model": ErrorResponse}})
async def query_brain_specimens_post(request: BrainQueryRequest) -> BrainQueryResponse:
    """POST version of structured query for complex brain research filters."""
    try:
        # Create filter object
        filters = BrainQueryFilter(
            race=request.race,
            subject_sex=request.subject_sex,
            age_min=request.age_min,
            age_max=request.age_max,
            ethnicity=request.ethnicity,
            manner_of_death=request.manner_of_death,
            diagnosis_contains=request.diagnosis_contains,
            repository=request.repository,
            pmi_max=request.pmi_max,
            rin_min=request.rin_min,
            brain_region_contains=request.brain_region_contains,
            hemisphere=request.hemisphere,
            limit=request.limit,
            offset=request.offset
        )

        # Execute SQL query
        result = brain_db_service.query_specimens(filters)

        # Convert to Pydantic models
        specimens = [BrainSpecimenModel(**specimen) for specimen in result.specimens]

        response = BrainQueryResponse(
            specimens=specimens,
            total_count=result.total_count,
            query_summary=result.query_summary,
            explanation=None
        )

        # Optional LLM explanation
        if request.explain and llm_service.is_available and result.specimens:
            try:
                # Format specimens for LLM (limit to first 10 for brevity)
                sample_specimens = result.specimens[:10]
                specimens_text = "\n".join([
                    f"- Subject {s['subject_id']}: {s['subject_sex']}, {s['subject_age']} years old, {s['race']}, {s['manner_of_death']}"
                    for s in sample_specimens if s.get('subject_id')
                ])

                prompt = f"""Based on the following brain research query results, provide a brief, natural language summary:

Query: {result.query_summary}

Sample Results:
{specimens_text}

Provide a 1-2 sentence summary of the research cohort and any notable patterns in demographics or clinical characteristics."""

                explanation = llm_service._client.chat.completions.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7,
                )
                response.explanation = explanation.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM explanation failed: {e}")

        return response

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/brain/stats", response_model=BrainStatsResponse)
async def get_brain_stats() -> BrainStatsResponse:
    """Get brain research database statistics and metadata."""
    try:
        stats = brain_db_service.get_stats()
        return BrainStatsResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
