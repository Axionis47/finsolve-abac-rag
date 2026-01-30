"""
Metadata Extractor for RAG Indexing.

Extracts temporal (year, quarter) and topic metadata from documents
to enable better filtering and retrieval.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Optional


def extract_temporal_metadata(text: str, filename: str) -> Dict[str, Optional[str | List[str]]]:
    """
    Extract year and quarter from content and filename.

    Args:
        text: Document text content
        filename: Document filename (can include path)

    Returns:
        Dict with keys:
        - primary_quarter: Most likely quarter (Q1-Q4 or None)
        - primary_year: Most likely year (e.g., "2024" or None)
        - mentioned_quarters: List of all quarters mentioned
        - mentioned_years: List of all years mentioned
    """
    # Get just the filename without path
    basename = os.path.basename(filename)

    # From filename (e.g., marketing_report_q1_2024.md, market_report_q4_2024.md)
    # Match patterns like: q1_2024, q1-2024, q12024, Q1_2024
    quarter_match = re.search(r'q([1-4])[-_]?(20\d{2})', basename, re.I)
    year_match = re.search(r'(20\d{2})', basename)

    # From content - find all quarter and year references
    content_quarters = re.findall(r'\b(Q[1-4])\b', text, re.I)
    content_years = re.findall(r'\b(202[0-9])\b', text)

    # Determine primary values
    primary_quarter: Optional[str] = None
    primary_year: Optional[str] = None

    if quarter_match:
        primary_quarter = f"Q{quarter_match.group(1)}"
        primary_year = quarter_match.group(2)
    elif year_match:
        primary_year = year_match.group(1)

    # If no filename match, use most frequent content match
    if not primary_quarter and content_quarters:
        quarter_counts = Counter(q.upper() for q in content_quarters)
        primary_quarter = quarter_counts.most_common(1)[0][0]

    if not primary_year and content_years:
        year_counts = Counter(content_years)
        primary_year = year_counts.most_common(1)[0][0]

    return {
        "primary_quarter": primary_quarter,
        "primary_year": primary_year,
        "mentioned_quarters": sorted(set(q.upper() for q in content_quarters)),
        "mentioned_years": sorted(set(content_years)),
    }


def extract_topics(text: str) -> List[str]:
    """
    Extract key topics/keywords from text for enhanced retrieval.

    Args:
        text: Document text content

    Returns:
        List of topic tags found in the text
    """
    # Topic patterns: (regex_pattern, topic_tag)
    topic_patterns = [
        # Financial topics
        (r'\brevenue\b', 'revenue'),
        (r'\bexpenses?\b', 'expenses'),
        (r'\bprofit\b', 'profit'),
        (r'\bbudget\b', 'budget'),
        (r'\bforecast\b', 'forecast'),
        (r'\binvestment\b', 'investment'),
        # HR topics
        (r'\bsalary|salaries\b', 'salary'),
        (r'\bcompensation\b', 'compensation'),
        (r'\bvacation|leave|pto|time.?off\b', 'leave_policy'),
        (r'\bbenefits?\b', 'benefits'),
        (r'\bperformance\b', 'performance'),
        (r'\bhiring|recruitment\b', 'hiring'),
        (r'\bemployee\b', 'employee'),
        # Engineering topics
        (r'\barchitecture\b', 'architecture'),
        (r'\bsecurity\b', 'security'),
        (r'\binfrastructure\b', 'infrastructure'),
        (r'\bscalability\b', 'scalability'),
        (r'\bapi\b', 'api'),
        (r'\bdatabase\b', 'database'),
        # Marketing topics
        (r'\bmarketing\b', 'marketing'),
        (r'\bcampaign\b', 'campaign'),
        (r'\bbrand\b', 'brand'),
        (r'\bcustomer\b', 'customer'),
        (r'\bgrowth\b', 'growth'),
    ]

    topics = []
    text_lower = text.lower()

    for pattern, topic in topic_patterns:
        if re.search(pattern, text_lower):
            topics.append(topic)

    return sorted(set(topics))


def extract_sensitivity_indicators(text: str) -> Dict[str, bool]:
    """
    Detect indicators of sensitive content in text.

    Args:
        text: Document text content

    Returns:
        Dict with boolean flags for different sensitivity types
    """
    text_lower = text.lower()

    return {
        "has_pii": bool(re.search(
            r'\b(ssn|social.?security|date.?of.?birth|dob|passport)\b',
            text_lower
        )),
        "has_financial_pii": bool(re.search(
            r'\b(salary|compensation|bonus|payroll|bank.?account)\b',
            text_lower
        )),
        "has_confidential_markers": bool(re.search(
            r'\b(confidential|restricted|private|internal.?only|do.?not.?share)\b',
            text_lower
        )),
        "has_performance_data": bool(re.search(
            r'\b(performance.?review|rating|evaluation|termination|disciplinary)\b',
            text_lower
        )),
    }


def enrich_chunk_metadata(
    chunk: Dict,
    full_text: str,
    source_path: str
) -> Dict:
    """
    Enrich a chunk with extracted metadata.

    Args:
        chunk: The chunk dictionary to enrich
        full_text: Full document text (for context)
        source_path: Path to source document

    Returns:
        Enriched chunk dictionary
    """
    chunk_text = chunk.get("text", "")

    # Extract temporal from full document (for primary) and chunk (for mentions)
    doc_temporal = extract_temporal_metadata(full_text, source_path)
    chunk_temporal = extract_temporal_metadata(chunk_text, source_path)

    # Use document-level primary values, but chunk-level mentions
    chunk["year"] = doc_temporal["primary_year"]
    chunk["quarter"] = doc_temporal["primary_quarter"]
    chunk["temporal_refs"] = chunk_temporal["mentioned_quarters"]

    # Extract topics from chunk
    chunk["topics"] = extract_topics(chunk_text)

    # Check sensitivity indicators
    sensitivity = extract_sensitivity_indicators(chunk_text)
    chunk["has_sensitive_content"] = any(sensitivity.values())
    chunk["sensitivity_flags"] = sensitivity

    return chunk
