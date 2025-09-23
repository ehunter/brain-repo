from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMServiceError(Exception):
    """Raised when the LLM service cannot fulfill an operation."""


class LLMService:
    """Service for generating natural language responses using OpenAI's API."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._client = None
        else:
            try:
                # Initialize client with manual configuration to bypass compatibility issues
                import httpx
                self._client = openai.OpenAI(
                    api_key=api_key,
                    http_client=httpx.Client(
                        timeout=60.0,
                        follow_redirects=True
                    )
                )
            except Exception as e:
                print(f"Failed to initialize OpenAI client with custom http_client: {e}")
                # Fallback to basic initialization
                try:
                    self._client = openai.OpenAI(api_key=api_key)
                except Exception as e2:
                    print(f"Basic initialization failed: {e2}")
                    self._client = None

    @property
    def is_available(self) -> bool:
        """Check if the LLM service is available (API key is set)."""
        # Try to initialize if we don't have a client but API key is now available
        if self._client is None and os.environ.get("OPENAI_API_KEY"):
            api_key = os.environ.get("OPENAI_API_KEY")
            try:
                import httpx
                self._client = openai.OpenAI(
                    api_key=api_key,
                    http_client=httpx.Client(
                        timeout=60.0,
                        follow_redirects=True
                    )
                )
            except Exception:
                pass
        return self._client is not None

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4

    def _chunk_matches_by_tokens(self, matches: List[Dict[str, Any]], question: str, max_context_tokens: int = 150000) -> List[Dict[str, Any]]:
        """Select matches that fit within token limits, prioritizing by relevance score."""
        base_prompt_tokens = self._estimate_tokens(f"""You are a helpful assistant answering questions based on data from a CSV file.

Question: {question}

Dataset information:

Please provide a natural, conversational answer to the question based on the dataset information above. Analyze all the data provided to give the most accurate and comprehensive answer possible. Keep your response concise and helpful.""")

        available_tokens = max_context_tokens - base_prompt_tokens - 300  # Reserve for response
        selected_matches = []
        current_tokens = 0

        # Sort matches by score (highest relevance first)
        sorted_matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)

        for match in sorted_matches:
            text = match["text"]
            metadata = match.get("metadata", {})

            # Estimate tokens for this match
            if metadata:
                metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                match_text = f"{text} ({metadata_str})"
            else:
                match_text = text

            match_tokens = self._estimate_tokens(match_text)

            if current_tokens + match_tokens <= available_tokens:
                selected_matches.append(match)
                current_tokens += match_tokens
            else:
                # If we can't fit this match, stop adding more
                break

        return selected_matches

    def generate_natural_response(
        self,
        question: str,
        matches: List[Dict[str, Any]],
        *,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 300,
    ) -> str:
        """Generate a natural language response based on the question and retrieved matches."""
        if not self.is_available:
            raise LLMServiceError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable to enable natural language responses."
            )

        if not matches:
            return "I couldn't find any relevant information in the CSV to answer your question."

        # Determine model limits and chunk accordingly
        if model == "gpt-3.5-turbo":
            max_context_tokens = 150000  # Conservative limit for gpt-3.5-turbo
        elif model == "gpt-4":
            max_context_tokens = 120000  # Conservative limit for gpt-4
        else:
            max_context_tokens = 150000  # Default conservative limit

        # Intelligently select matches that fit within token limits
        selected_matches = self._chunk_matches_by_tokens(matches, question, max_context_tokens)

        # If we had to reduce matches, try GPT-4 with original data
        if len(selected_matches) < len(matches) and model == "gpt-3.5-turbo":
            try:
                return self.generate_natural_response(question, matches, model="gpt-4", max_tokens=max_tokens)
            except Exception:
                # Fall back to chunked data with gpt-3.5-turbo
                pass

        # Build context from selected matches
        context_parts = []
        for i, match in enumerate(selected_matches, 1):
            text = match["text"]
            metadata = match.get("metadata", {})

            if metadata:
                metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                context_parts.append(f"{i}. {text} ({metadata_str})")
            else:
                context_parts.append(f"{i}. {text}")

        context = "\n".join(context_parts)

        # Add info about data reduction if applicable
        data_note = ""
        if len(selected_matches) < len(matches):
            data_note = f" (Analyzed {len(selected_matches)} most relevant entries out of {len(matches)} total matches)"

        prompt = f"""You are a helpful assistant answering questions based on data from a CSV file.

Question: {question}

Dataset information{data_note}:
{context}

Please provide a natural, conversational answer to the question based on the dataset information above. Analyze all the data provided to give the most accurate and comprehensive answer possible. Keep your response concise and helpful."""

        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMServiceError("Empty response from OpenAI API")

            return content.strip()

        except Exception as exc:
            raise LLMServiceError(f"Failed to generate natural language response: {exc}") from exc


__all__ = ["LLMService", "LLMServiceError"]