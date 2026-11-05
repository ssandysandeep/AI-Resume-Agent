"""Shared utilities for the agent module."""
import os
import logging
import asyncio
from typing import Dict, Any, Tuple, List, Set
from dotenv import load_dotenv
import openai

# Load environment variables once
load_dotenv()

# Configure logging once
logging.basicConfig(level=logging.INFO)

# Shared logger
logger = logging.getLogger(__name__)

# REMOVED: SOFT_SKILLS_SET - LLM extracts and categorizes soft skills directly from job descriptions
# No static list needed - LLM identifies soft skills based on context and training

# Shared OpenAI clients
_openai_client = None
_async_openai_client = None

def get_openai_client():
    """Get or create synchronous OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def get_async_openai_client():
    """Get or create async OpenAI client."""
    global _async_openai_client
    if _async_openai_client is None:
        from openai import AsyncOpenAI
        _async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _async_openai_client

def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize text for matching: replace & with 'and', handle variations, normalize spaces.
    
    Args:
        text: Text to normalize
        lowercase: If True, convert to lowercase (default: True)
    """
    if not text:
        return ""
    normalized = text.lower().replace("&", " and ").replace("/", " ").replace("-", " ") if lowercase else text.replace("&", " and ").replace("/", " ").replace("-", " ")
    # Replace regex with simple string method for whitespace normalization
    return " ".join(normalized.split()).strip()

def normalize_whitespace(text: str) -> str:
    """Simple whitespace normalization for text scanning."""
    return " ".join((text or "").split()).strip()

def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword for matching (preserves case)."""
    return normalize_text(keyword, lowercase=False)

def strip_excess_chars(text: str) -> str:
    """Replace multiple slashes or pipes with a single comma or space."""
    if not text:
        return text
    # Replace regex with simple string methods
    while "||" in text or "//" in text or "|/" in text or "/|" in text:
        text = text.replace("||", ",").replace("//", ",").replace("|/", ",").replace("/|", ",")
    while "  " in text:  # Normalize spaces
        text = text.replace("  ", " ")
    return text.strip()

def dedupe_items(items: List[str]) -> List[str]:
    """
    Unified deduplication function for skills/keywords.
    CRITICAL: Only removes EXACT duplicates (case-insensitive) - does NOT remove substring matches.
    User requirement: Do not remove skills - only remove exact duplicates.
    
    Args:
        items: List of strings to deduplicate
        
    Returns:
        Deduplicated list of strings (only exact duplicates removed)
    """
    if not items:
        return []
    
    items_clean = [s for s in items if s and s.strip()]
    if not items_clean:
        return []
    
    # ONLY exact/case-insensitive dedupe: keep first occurrence
    # DO NOT do substring deduplication - it removes valid skills
    seen = set()
    kept = []
    for s in items_clean:
        s_stripped = s.strip()
        norm = " ".join(s_stripped.split()).lower()
        if norm not in seen:
            seen.add(norm)
            kept.append(s_stripped)
    
    return kept


def dedupe_items_set(items: Set[str]) -> Set[str]:
    """
    Unified deduplication function that returns a Set.
    Same logic as dedupe_items but returns Set[str].
    """
    return set(dedupe_items(list(items)))


def is_non_technical_entry(entry: str) -> bool:
    """
    Check if entry is education, legal status, or vague term that should NOT be in Skills section.
    
    This function is used to filter out non-technical entries from the Technical Skills section.
    It detects:
    - Education entries (degrees, certifications, etc.)
    - Legal status (work authorization, security clearance, etc.)
    - Vague/generic terms (related field, best practices, etc.)
    
    Args:
        entry: String to check (skill or keyword)
        
    Returns:
        True if entry should be excluded from Skills section, False otherwise
    """
    if not entry:
        return False
    
    entry_lower = str(entry).lower().strip()
    
    # Education patterns
    education_patterns = [
        'bachelor', 'master', 'phd', 'degree', 'diploma', 'certification', 'certified',
        'related field', 'field of study', 'major in', 'minor in', 'education',
        'computer science degree', 'engineering degree', 'bachelor\'s', 'master\'s'
    ]
    if any(pattern in entry_lower for pattern in education_patterns):
        return True
    
    # Legal status / authorization patterns
    legal_patterns = [
        'authorized to work', 'eligible to obtain', 'security clearance', 'clearance',
        'work authorization', 'work permit', 'visa', 'h1b', 'green card', 'citizenship',
        'eligible for', 'authorized for', 'u.s. government', 'government security',
        'u.s.', 'us government'
    ]
    if any(pattern in entry_lower for pattern in legal_patterns):
        return True
    
    # Vague/generic terms
    vague_patterns = [
        'related field', 'best practices', 'general knowledge', 'familiarity with',
        'understanding of', 'knowledge of', 'ability to', 'capable of'
    ]
    if any(pattern in entry_lower for pattern in vague_patterns):
        return True
    
    return False

def sanitize_error_message(error_msg: str) -> str:
    """
    Sanitize error messages by removing HTML content and extracting useful information.
    Returns a user-friendly error message.
    """
    if not error_msg:
        return "Unknown error"
    
    # Check if error message contains HTML
    if "<!DOCTYPE html>" in error_msg or "<html" in error_msg.lower():
        # Extract useful information from HTML error pages using string methods instead of regex
        if "<h2" in error_msg and "</h2>" in error_msg:
            # Simple extraction without regex
            start = error_msg.find("<h2")
            end = error_msg.find("</h2>", start)
            if start != -1 and end != -1:
                h2_content = error_msg[start:end+5]
                # Extract text between tags
                tag_start = h2_content.find(">")
                if tag_start != -1:
                    text = h2_content[tag_start+1:].replace("</h2>", "").strip()
                    if text:
                        return f"Network/DNS error: {text}"
        
        # Check for Cloudflare DNS errors
        if "Origin DNS error" in error_msg or "error 1016" in error_msg.lower():
            return "Network connection error: Unable to connect to OpenAI API (DNS resolution failed). Please check your internet connection and try again."
        
        # Check for other Cloudflare errors using string methods
        if "<span" in error_msg and "</span>" in error_msg:
            # Try to find error codes in spans
            parts = error_msg.split("<span")
            for part in parts[1:]:  # Skip first part
                if "</span>" in part:
                    span_content = part.split("</span>")[0]
                    if span_content and span_content.strip().isdigit() and len(span_content.strip()) == 4:
                        code = span_content.strip()
                        return f"Network error ({code}): Unable to connect to OpenAI API. Please check your internet connection and try again."
        
        # Generic HTML error
        return "Network connection error: Unable to connect to OpenAI API. Please check your internet connection and try again."
    
    # Check for common connection error patterns
    error_lower = error_msg.lower()
    if "dns" in error_lower or "name resolution" in error_lower or "cannot resolve" in error_lower:
        return "Network connection error: DNS resolution failed. Please check your internet connection and try again."
    
    if "timed out" in error_lower or ("timeout" in error_lower and "request" in error_lower):
        return "Request timed out: The API call took too long to complete. Please try again."
    
    if "connection" in error_lower and ("refused" in error_lower or "timeout" in error_lower or "failed" in error_lower):
        return "Network connection error: Unable to connect to OpenAI API. Please check your internet connection and try again."
    
    # Return original message if it doesn't contain HTML
    return error_msg


async def handle_openai_error(error: Exception, attempt: int, max_retries: int, operation: str = "operation") -> bool:
    """
    Handle OpenAI API errors with retry logic.
    Returns True if should retry, False if should raise.
    """
    error_msg = str(error)
    sanitized_msg = sanitize_error_message(error_msg)
    
    # Check for timeout errors specifically
    if isinstance(error, openai.APITimeoutError) or "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
        if attempt >= max_retries:
            logger.error(f"Request timeout after {max_retries + 1} attempts for {operation}. The API call took too long.")
            return False
        # Wait before retrying timeout errors
        wait_time = min((2 ** attempt) * 1, 3)  # Max 3 seconds
        logger.warning(f"Request timeout on attempt {attempt + 1} for {operation}. Waiting {wait_time}s before retry...")
        await asyncio.sleep(wait_time)
        return True
    
    # Check for connection/DNS errors (don't retry these aggressively)
    if isinstance(error, openai.APIConnectionError) or \
       "DNS" in error_msg or "connection" in error_msg.lower() or "<!DOCTYPE html>" in error_msg:
        if attempt >= max_retries:
            logger.error(f"Network/Connection error after {max_retries + 1} attempts for {operation}: {sanitized_msg}")
            return False
        # Wait longer for network errors (exponential backoff)
        wait_time = min((2 ** attempt) * 1, 5)  # Max 5 seconds
        logger.warning(f"Network/Connection error on attempt {attempt + 1} for {operation}: {sanitized_msg}. Waiting {wait_time}s before retry...")
        await asyncio.sleep(wait_time)
        return True
    
    if isinstance(error, openai.RateLimitError):
        wait_time = (2 ** attempt) * 2
        if attempt >= max_retries:
            logger.error(f"OpenAI Rate Limit Error (429): Too many requests after {max_retries + 1} attempts. Error: {sanitized_msg}")
            return False
        logger.warning(f"Rate limit error (429) on attempt {attempt + 1} for {operation}: {sanitized_msg}. Waiting {wait_time}s before retry...")
        await asyncio.sleep(wait_time)
        return True
    elif isinstance(error, openai.APIError):
        if hasattr(error, 'status_code') and error.status_code == 429:
            wait_time = (2 ** attempt) * 2
            if attempt >= max_retries:
                logger.error(f"OpenAI API Error 429 (Too Many Requests) after {max_retries + 1} attempts for {operation}: {sanitized_msg}")
                return False
            logger.warning(f"Rate limit error (429) on attempt {attempt + 1} for {operation}: {sanitized_msg}. Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
            return True
        else:
            if attempt >= max_retries:
                logger.error(f"OpenAI API Error after {max_retries + 1} attempts for {operation}: {sanitized_msg}")
                return False
            logger.warning(f"OpenAI API Error on attempt {attempt + 1} for {operation}: {sanitized_msg}. Retrying...")
            await asyncio.sleep(0.2)  # Reduced wait time
            return True
    else:
        if attempt >= max_retries:
            logger.error(f"Error in {operation} after {max_retries + 1} attempts: {sanitized_msg}")
            return False
        logger.warning(f"Error on attempt {attempt + 1} for {operation}: {sanitized_msg}. Retrying...")
        await asyncio.sleep(0.2)  # Reduced wait time
        return True

