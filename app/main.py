from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import AskRequest, AskResponse, ErrorResponse, LoadResponse, Match
from app.services.brain_bank import BrainBank, BrainBankError

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
        result = brain_bank.ask(request.question, top_k=request.top_k)
    except BrainBankError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    matches = [Match(**match) for match in result["matches"]]
    return AskResponse(answer=result["answer"], matches=matches)


def _parse_columns(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    columns = [col.strip() for col in raw_value.split(",") if col.strip()]
    return columns or None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
