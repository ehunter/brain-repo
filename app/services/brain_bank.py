from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class BrainBankError(Exception):
    """Raised when the Brain Bank cannot fulfill an operation."""


@dataclass
class BrainBankStatus:
    rows: int
    columns: List[str]
    text_columns: List[str]
    metadata_columns: List[str]


class BrainBank:
    """In-memory store that pairs CSV rows with a FAISS index for retrieval."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._text_column_name = "_brain_bank_text"
        self._text_columns: List[str] = []
        self._metadata_columns: List[str] = []

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._df is not None and self._vectorizer is not None

    def load_csv(
        self,
        file_bytes: bytes,
        *,
        text_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ) -> BrainBankStatus:
        if not file_bytes:
            raise BrainBankError("The provided CSV file is empty.")

        df = self._read_csv(file_bytes)
        if df.empty:
            raise BrainBankError("The CSV file does not contain any rows.")

        selected_text_cols = self._resolve_text_columns(df, text_columns)
        resolved_metadata = self._resolve_metadata_columns(df, metadata_columns, selected_text_cols)

        combined_text = (
            df[selected_text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace("\s+", " ", regex=True)
            .str.strip()
        )
        if combined_text.eq("").all():
            raise BrainBankError("The selected columns do not contain textual information to index.")

        df = df.copy()
        df[self._text_column_name] = combined_text.replace("", "(empty row)")

        vectorizer = TfidfVectorizer(stop_words="english")
        doc_matrix = vectorizer.fit_transform(df[self._text_column_name])
        dense_vectors = doc_matrix.toarray().astype("float32")
        if dense_vectors.size == 0:
            raise BrainBankError("Unable to build embeddings from the provided CSV.")
        faiss.normalize_L2(dense_vectors)

        index = faiss.IndexFlatIP(dense_vectors.shape[1])
        index.add(dense_vectors)

        self._df = df
        self._vectorizer = vectorizer
        self._index = index
        self._text_columns = selected_text_cols
        self._metadata_columns = resolved_metadata

        return BrainBankStatus(
            rows=int(df.shape[0]),
            columns=list(df.columns),
            text_columns=selected_text_cols,
            metadata_columns=resolved_metadata,
        )

    def ask(self, question: str, *, top_k: int = 3) -> Dict[str, Any]:
        if not self.is_ready:
            raise BrainBankError("Load a CSV file before asking questions.")
        if not question.strip():
            raise BrainBankError("The question cannot be empty.")

        query_vec = self._vectorizer.transform([question])  # type: ignore[union-attr]
        dense_query = query_vec.toarray().astype("float32")
        if dense_query.size == 0:
            raise BrainBankError("Unable to embed the question. Try rephrasing it.")
        faiss.normalize_L2(dense_query)

        assert self._index is not None
        assert self._df is not None

        top_k = max(1, min(top_k, len(self._df)))
        scores, indices = self._index.search(dense_query, top_k)
        scores_list = scores[0].tolist()
        idx_list = indices[0].tolist()

        matches = []
        for score, row_idx in zip(scores_list, idx_list):
            if row_idx < 0 or row_idx >= len(self._df):
                continue
            row = self._df.iloc[row_idx]
            metadata = {col: row[col] for col in self._metadata_columns if col in row}
            matches.append(
                {
                    "score": float(score),
                    "row_index": int(row_idx),
                    "text": row[self._text_column_name],
                    "metadata": metadata,
                }
            )

        answer = self._format_answer(matches)
        return {"answer": answer, "matches": matches}

    def status(self) -> BrainBankStatus:
        if not self.is_ready:
            raise BrainBankError("No CSV has been loaded yet.")
        assert self._df is not None
        return BrainBankStatus(
            rows=int(self._df.shape[0]),
            columns=list(self._df.columns),
            text_columns=list(self._text_columns),
            metadata_columns=list(self._metadata_columns),
        )

    def _read_csv(self, file_bytes: bytes) -> pd.DataFrame:
        buffer = io.BytesIO(file_bytes)
        try:
            return pd.read_csv(buffer)
        except Exception:
            buffer.seek(0)
            try:
                return pd.read_csv(buffer, sep=None, engine="python")
            except Exception as exc:  # pragma: no cover - defensive
                raise BrainBankError(f"Unable to read the CSV file: {exc}") from exc

    def _resolve_text_columns(
        self, df: pd.DataFrame, requested: Optional[List[str]]
    ) -> List[str]:
        if requested:
            missing = [col for col in requested if col not in df.columns]
            if missing:
                raise BrainBankError(f"Text column(s) not found: {', '.join(missing)}")
            return requested

        candidate_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" or str(df[col].dtype).startswith("string")
        ]
        if not candidate_cols:
            # Fallback to every column cast to string.
            candidate_cols = list(df.columns)
        return candidate_cols

    def _resolve_metadata_columns(
        self,
        df: pd.DataFrame,
        requested: Optional[List[str]],
        text_columns: List[str],
    ) -> List[str]:
        if requested:
            missing = [col for col in requested if col not in df.columns]
            if missing:
                raise BrainBankError(f"Metadata column(s) not found: {', '.join(missing)}")
            return requested

        return [col for col in df.columns if col not in text_columns]

    def _format_answer(self, matches: List[Dict[str, Any]]) -> str:
        if not matches:
            return "No relevant rows were found in the CSV."

        lines = ["Top matches from the CSV:"]
        for match in matches:
            snippet = match["text"]
            metadata = match.get("metadata") or {}
            details = ", ".join(f"{key}: {value}" for key, value in metadata.items())
            if details:
                lines.append(f"• {snippet} ({details})")
            else:
                lines.append(f"• {snippet}")
        return "\n".join(lines)


__all__ = ["BrainBank", "BrainBankError", "BrainBankStatus"]
