from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import openai
from dotenv import load_dotenv

from .brain_database import BrainDatabaseService, BrainQueryFilter

# Load environment variables from .env file
load_dotenv()


@dataclass
class QueryAnalysis:
    """Analysis of a natural language query for brain research data."""
    subject_ids: List[str]
    demographics: Dict[str, Any]
    clinical_terms: List[str]
    brain_regions: List[str]
    filters: BrainQueryFilter
    intent: str  # "find_subject", "compare_demographics", "search_diagnosis", etc.
    confidence: float  # 0.0 to 1.0


@dataclass
class EnhancedResponse:
    """Enhanced response with natural language and follow-up suggestions."""
    answer: str
    follow_up_suggestions: List[str]
    structured_filters: Optional[BrainQueryFilter]
    data_insights: List[str]


class EnhancedLLMServiceError(Exception):
    """Raised when the enhanced LLM service cannot fulfill an operation."""


class EnhancedLLMService:
    """Enhanced service for brain research queries with intelligent parsing and suggestions."""

    def __init__(self, brain_db_service: BrainDatabaseService) -> None:
        self.brain_db = brain_db_service
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._client = None
        else:
            try:
                import httpx
                self._client = openai.OpenAI(
                    api_key=api_key,
                    http_client=httpx.Client(
                        timeout=60.0,
                        follow_redirects=True
                    )
                )
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self._client = None

    @property
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self._client is not None

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a natural language query to extract search parameters."""
        query_lower = query.lower()

        # Extract subject IDs (patterns like NIH1000, BR123, 179130, etc.)
        subject_id_patterns = [
            r'\b(nih\d+)\b',
            r'\b(br\d+)\b',
            r'\b(\d{6})\b',  # 6-digit numbers like 179130
            r'\b(\d{4,5})\b',  # 4-5 digit numbers
            r'\b([a-z]+\d+)\b',
            r'\bsubject\s+(?:id\s+)?["\']?([a-z0-9]+)["\']?\b',
            r'\bfind\s+subject\s+([a-z0-9]+)\b',
            r'\bsubject\s+([a-z0-9]+)\b'
        ]

        subject_ids = []
        for pattern in subject_id_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            subject_ids.extend(matches)

        # Remove duplicates while preserving order
        subject_ids = list(dict.fromkeys(subject_ids))

        # Extract demographics
        demographics = {}

        # Age patterns
        age_match = re.search(r'age[s]?\s*(?:of|is|are|between)?\s*(\d+)(?:\s*(?:to|and|-)\s*(\d+))?', query_lower)
        if age_match:
            age_min = int(age_match.group(1))
            age_max = int(age_match.group(2)) if age_match.group(2) else None
            demographics['age_min'] = age_min
            if age_max:
                demographics['age_max'] = age_max

        # Sex/Gender
        if re.search(r'\b(male|males|men)\b', query_lower):
            demographics['subject_sex'] = 'Male'
        elif re.search(r'\b(female|females|women)\b', query_lower):
            demographics['subject_sex'] = 'Female'

        # Race/Ethnicity
        race_patterns = {
            'white': r'\b(white|caucasian)\b',
            'black': r'\b(black|african.?american)\b',
            'hispanic': r'\b(hispanic|latino|latina)\b',
            'asian': r'\b(asian)\b'
        }

        for race, pattern in race_patterns.items():
            if re.search(pattern, query_lower):
                demographics['race'] = race.title()
                break

        # Clinical terms
        clinical_terms = []
        clinical_patterns = [
            r'\b(alzheimer|alzheimers|dementia)\b',
            r'\b(depression|depressive)\b',
            r'\b(suicide|suicidal)\b',
            r'\b(alcohol|alcoholism|substance)\b',
            r'\b(parkinson|parkinsons)\b',
            r'\b(autism|autistic)\b',
            r'\b(bipolar)\b',
            r'\b(schizophrenia|schizophrenic)\b'
        ]

        for pattern in clinical_patterns:
            matches = re.findall(pattern, query_lower)
            clinical_terms.extend(matches)

        # Brain regions
        brain_regions = []
        region_patterns = [
            r'\b(frontal|prefrontal|cortex)\b',
            r'\b(hippocampus|hippocampal)\b',
            r'\b(amygdala)\b',
            r'\b(cerebellum|cerebellar)\b',
            r'\b(temporal|parietal|occipital)\b',
            r'\b(brainstem|brain.?stem)\b'
        ]

        for pattern in region_patterns:
            matches = re.findall(pattern, query_lower)
            brain_regions.extend(matches)

        # Determine intent
        intent = "general_search"
        confidence = 0.5

        if subject_ids:
            intent = "find_subject"
            confidence = 0.9
        elif demographics and len(demographics) >= 2:
            intent = "demographic_search"
            confidence = 0.8
        elif clinical_terms:
            intent = "clinical_search"
            confidence = 0.7
        elif brain_regions:
            intent = "region_search"
            confidence = 0.7

        # Create filter object
        filters = BrainQueryFilter(
            subject_id=subject_ids[0] if subject_ids else None,
            age_min=demographics.get('age_min'),
            age_max=demographics.get('age_max'),
            subject_sex=demographics.get('subject_sex'),
            race=demographics.get('race'),
            diagnosis_contains=clinical_terms[0] if clinical_terms else None,
            brain_region_contains=brain_regions[0] if brain_regions else None
        )

        return QueryAnalysis(
            subject_ids=subject_ids,
            demographics=demographics,
            clinical_terms=clinical_terms,
            brain_regions=brain_regions,
            filters=filters,
            intent=intent,
            confidence=confidence
        )

    def generate_follow_up_suggestions(self,
                                     query_analysis: QueryAnalysis,
                                     results: Dict[str, Any]) -> List[str]:
        """Generate intelligent follow-up suggestions based on query and results."""
        suggestions = []
        specimens = results.get('specimens', [])

        if query_analysis.intent == "find_subject" and specimens:
            # Suggest exploring related subjects or demographics
            specimen = specimens[0]
            if specimen.get('race'):
                suggestions.append(f"Show other {specimen['race']} subjects")
            if specimen.get('subject_sex') and specimen.get('subject_age'):
                suggestions.append(f"Find {specimen['subject_sex'].lower()} subjects aged {specimen['subject_age']-5} to {specimen['subject_age']+5}")
            if specimen.get('repository'):
                suggestions.append(f"Show other specimens from {specimen['repository']}")

        elif query_analysis.intent == "demographic_search" and specimens:
            # Suggest exploring clinical aspects
            if specimens:
                suggestions.append("Show diagnostic information for these subjects")
                suggestions.append("Filter by brain regions available")
                suggestions.append("Show tissue quality metrics (PMI, RIN)")

        elif query_analysis.intent == "clinical_search" and specimens:
            # Suggest demographic breakdowns
            suggestions.append("Break down by demographics (age, sex, race)")
            suggestions.append("Show brain regions analyzed")
            suggestions.append("Compare with control subjects")

        # Generic suggestions based on data availability
        if len(specimens) > 10:
            suggestions.append("Narrow down results with more specific criteria")
            suggestions.append("Sort by age or other demographics")
        elif len(specimens) < 5 and len(specimens) > 0:
            suggestions.append("Broaden search criteria to find more specimens")
            suggestions.append("Find subjects with similar characteristics")
        elif len(specimens) == 0:
            suggestions.append("Try broader search terms")
            suggestions.append("Search by repository or general demographics")

        # Remove duplicates and limit to 4 suggestions
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:4]

    def generate_data_insights(self, specimens: List[Dict[str, Any]]) -> List[str]:
        """Generate insights about the data returned."""
        if not specimens:
            return ["No specimens found matching the criteria."]

        insights = []

        # Age distribution
        ages = [s.get('subject_age') for s in specimens if s.get('subject_age')]
        if ages:
            avg_age = sum(ages) / len(ages)
            min_age, max_age = min(ages), max(ages)
            insights.append(f"Age range: {min_age}-{max_age} years (average: {avg_age:.1f})")

        # Sex distribution
        sexes = [s.get('subject_sex') for s in specimens if s.get('subject_sex')]
        if sexes:
            male_count = sexes.count('Male')
            female_count = sexes.count('Female')
            total = len(sexes)
            insights.append(f"Sex distribution: {male_count}/{total} male ({male_count/total*100:.0f}%), {female_count}/{total} female ({female_count/total*100:.0f}%)")

        # Race distribution
        races = [s.get('race') for s in specimens if s.get('race')]
        if races:
            unique_races = list(set(races))
            if len(unique_races) <= 3:
                race_counts = {race: races.count(race) for race in unique_races}
                race_summary = ", ".join([f"{count} {race}" for race, count in race_counts.items()])
                insights.append(f"Racial composition: {race_summary}")

        # Repository distribution
        repositories = [s.get('repository') for s in specimens if s.get('repository')]
        if repositories:
            unique_repos = list(set(repositories))
            if len(unique_repos) <= 3:
                repo_counts = {repo: repositories.count(repo) for repo in unique_repos}
                repo_summary = ", ".join([f"{count} from {repo}" for repo, count in repo_counts.items()])
                insights.append(f"Repository sources: {repo_summary}")

        return insights[:3]  # Limit to 3 insights

    def generate_enhanced_response(self,
                                 query: str,
                                 query_analysis: QueryAnalysis,
                                 database_results: Dict[str, Any]) -> EnhancedResponse:
        """Generate an enhanced natural language response with insights and suggestions."""

        if not self.is_available:
            # Fallback response without LLM
            specimens = database_results.get('specimens', [])
            total_count = database_results.get('total_count', 0)

            if total_count == 0:
                answer = "No specimens found matching your search criteria."
            elif query_analysis.intent == "find_subject" and query_analysis.subject_ids:
                answer = f"Found {total_count} specimen(s) for subject ID '{query_analysis.subject_ids[0]}'."
                if specimens:
                    specimen = specimens[0]
                    details = []
                    if specimen.get('subject_age'): details.append(f"Age: {specimen['subject_age']}")
                    if specimen.get('subject_sex'): details.append(f"Sex: {specimen['subject_sex']}")
                    if specimen.get('race'): details.append(f"Race: {specimen['race']}")
                    if details:
                        answer += f" Subject details: {', '.join(details)}."
            else:
                answer = f"Found {total_count} specimens matching your criteria."

            follow_ups = self.generate_follow_up_suggestions(query_analysis, database_results)
            insights = self.generate_data_insights(database_results.get('specimens', []))

            return EnhancedResponse(
                answer=answer,
                follow_up_suggestions=follow_ups,
                structured_filters=query_analysis.filters,
                data_insights=insights
            )

        # Generate LLM-powered response
        specimens = database_results.get('specimens', [])
        insights = self.generate_data_insights(specimens)
        follow_ups = self.generate_follow_up_suggestions(query_analysis, database_results)

        # Prepare context for LLM
        context_parts = []
        for i, specimen in enumerate(specimens[:10], 1):  # Limit to first 10 for context
            details = []
            if specimen.get('subject_id'): details.append(f"ID: {specimen['subject_id']}")
            if specimen.get('subject_age'): details.append(f"Age: {specimen['subject_age']}")
            if specimen.get('subject_sex'): details.append(f"Sex: {specimen['subject_sex']}")
            if specimen.get('race'): details.append(f"Race: {specimen['race']}")
            if specimen.get('repository'): details.append(f"Repository: {specimen['repository']}")

            context_parts.append(f"{i}. {', '.join(details)}")

        context = "\n".join(context_parts)
        query_summary = database_results.get('query_summary', '')

        prompt = f"""You are a brain research data assistant. Answer the user's question based on the search results.

User Question: {query}

Search Results Summary: {query_summary}

Specimen Details:
{context}

Data Insights:
{chr(10).join(f"• {insight}" for insight in insights)}

Provide a natural, informative response that:
1. Directly answers the user's question
2. Highlights key findings from the data
3. Uses appropriate scientific terminology
4. Is concise but comprehensive
5. Mentions specific subject IDs when relevant

Keep the response under 200 words and focus on the most relevant information."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                raise EnhancedLLMServiceError("Empty response from OpenAI API")

            return EnhancedResponse(
                answer=content.strip(),
                follow_up_suggestions=follow_ups,
                structured_filters=query_analysis.filters,
                data_insights=insights
            )

        except Exception as exc:
            # Fallback to structured response
            total_count = database_results.get('total_count', 0)
            fallback_answer = f"Found {total_count} specimens. " + " ".join(insights) if insights else f"Found {total_count} specimens matching your criteria."

            return EnhancedResponse(
                answer=fallback_answer,
                follow_up_suggestions=follow_ups,
                structured_filters=query_analysis.filters,
                data_insights=insights
            )


__all__ = ["EnhancedLLMService", "EnhancedLLMServiceError", "QueryAnalysis", "EnhancedResponse"]