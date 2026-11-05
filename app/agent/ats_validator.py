"""
ATS text-layer preflight: deterministic keyword coverage + deterministic injection + frequency control.

Hard constraints:
- ATS is treated as literal string-matching (verbatim coverage)
- Generate resume with LLM exactly once
- After generation: NO LLM rewrite/regeneration; only deterministic insertion using a phrase library
- Inject ONLY into safe sections: Skills (Technical Skills), Experience bullets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Set, Tuple
import asyncio

from app.agent.utils import logger, normalize_whitespace, get_async_openai_client, is_non_technical_entry
# Removed: is_noise_keyword - filtering now done via LLM in extractor.py
from app.schemas.resume import ResumeContent, SkillCategory, Experience
from pydantic import BaseModel, Field

KeywordType = Literal[
    "hard_skill",
    "education",
    "framework",
    "tool",
    "artifact",
    "authorization",
    "soft_skill",
    "action",
    "responsibility",
    "reporting_audit",
    "other",
]


@dataclass(frozen=True)
class KeywordLimits:
    min_count: int
    max_count: int


DEFAULT_LIMITS: Dict[KeywordType, KeywordLimits] = {
    # User requirement: avoid repetition; default max_count=1 for all keyword types.
    "hard_skill": KeywordLimits(1, 1),
    "education": KeywordLimits(1, 1),
    "framework": KeywordLimits(1, 1),
    "tool": KeywordLimits(1, 1),
    "artifact": KeywordLimits(1, 1),
    "authorization": KeywordLimits(1, 1),
    "soft_skill": KeywordLimits(1, 1),
    "action": KeywordLimits(1, 1),
    "responsibility": KeywordLimits(1, 1),
    "reporting_audit": KeywordLimits(1, 1),
    "other": KeywordLimits(1, 1),
}


# LLM classification functions (moved from llm_classifier.py)
class SkillFilterResult(BaseModel):
    """Result for skill filtering."""
    skill: str = Field(description="The skill string")
    is_descriptive: bool = Field(description="True if descriptive phrase (remove), False if pure technical skill (keep)")

class BatchSkillFilterResult(BaseModel):
    """Batch skill filter results."""
    results: List[SkillFilterResult] = Field(description="List of filtering results")

class KeywordWithTechnicalCheck(BaseModel):
    """Keyword classification with technical check."""
    keyword: str = Field(description="The keyword")
    classification: KeywordType = Field(description="The classification type")
    is_technical_skill: bool = Field(description="True if keyword looks like a technical skill/tool suitable for Skills section")

class BatchTechnicalCheckResult(BaseModel):
    """Batch technical check results."""
    results: List[KeywordWithTechnicalCheck] = Field(description="List of classification and technical check results")


async def filter_descriptive_phrases_llm(skills: List[str]) -> Dict[str, bool]:
    """
    Batch filter skills to identify descriptive phrases using LLM.
    
    Args:
        skills: List of skill strings to check
        
    Returns:
        Dictionary mapping skill -> True if descriptive (should remove), False if pure technical (keep)
    """
    if not skills:
        return {}
    
    results = {}
    batch_size = 60  # Increased batch size to reduce API calls
    
    for i in range(0, len(skills), batch_size):
        batch = skills[i:i+batch_size]
        
        try:
            client = get_async_openai_client()
            
            system_prompt = """You are a skill validator for resume generation. Classify each skill as descriptive phrase (remove) or pure technical skill (keep).

DESCRIPTIVE (remove): Experience-phrased items ("experience with...", "proficiency in...", "familiarity with...", "ability to use..."), starts with verbs, action words, placeholder text, non-technical prefixes, activity descriptions, verb-led phrases. These should be routed to Experience/Summary sections, NOT Hard Skills section.

PURE TECHNICAL (keep): Standalone noun-based tool names (Python, Java, AWS, Docker), programming languages, frameworks (React, Django), databases (PostgreSQL, MySQL), platforms (Kubernetes), technologies (CI/CD, REST API). Hard Skills section must contain ONLY standalone noun-based skills.

Return True if descriptive (remove from Hard Skills), False if pure technical (keep in Hard Skills)."""

            user_prompt = f"""Classify each skill as descriptive (remove) or pure technical (keep):

{chr(10).join([f'{j+1}. {skill}' for j, skill in enumerate(batch)])}

Return classification for all {len(batch)} skills."""

            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",  # OPTIMIZATION: Use faster model for filtering
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=BatchSkillFilterResult,
                temperature=0.0,
                timeout=8.0,  # OPTIMIZATION: Reduced from 10.0 to 8.0
                max_tokens=2000  # OPTIMIZATION: Reduced from 3000 to 2000 for faster processing
            )
            
            parsed = response.choices[0].message.parsed
            for result in parsed.results:
                results[result.skill] = result.is_descriptive
                
        except Exception as e:
            logger.warning(f"LLM batch descriptive phrase check failed, defaulting to False for all: {e}")
            # Default to keeping skills if LLM fails
            for skill in batch:
                if skill not in results:
                    results[skill] = False
    
    return results


async def classify_keywords_with_technical_check_llm(keywords: List[str]) -> Dict[str, Tuple[KeywordType, bool]]:
    """
    Classify keywords and determine if they look like technical skills.
    This combines classification and technical skill detection in one LLM call.
    OPTIMIZED: Processes batches in parallel for faster execution.
    
    Args:
        keywords: List of keywords to classify
        
    Returns:
        Dictionary mapping keyword -> (classification_type, is_technical_skill)
    """
    if not keywords:
        return {}
    
    batch_size = 80  # Increased batch size to reduce API calls
    batches = [keywords[i:i+batch_size] for i in range(0, len(keywords), batch_size)]
    
    async def process_batch(batch: List[str], batch_num: int) -> Dict[str, Tuple[KeywordType, bool]]:
        """Process a single batch of keywords."""
        batch_results = {}
        try:
            client = get_async_openai_client()
            
            system_prompt = """You are a keyword classifier for ATS-optimized resume generation. Classify each keyword into ONE category and determine if it's a technical skill.

Categories:
- hard_skill: Programming languages, technical tools, technologies, frameworks, databases (Python, AWS, Docker, React, PostgreSQL) - MUST be standalone noun-based skills, NOT experience-phrased items
- tool: Cloud platforms, development tools, software applications (AWS, Docker, Tableau, Power BI)
- framework: Compliance/governance frameworks, methodologies (NIST, FedRAMP, ISO, SOC, Agile, Scrum)
- artifact: Compliance documents/artifacts (POA&M, SSP, SAR, ATO, Continuous Monitoring)
- education: Educational fields/degrees (Computer Science, Mathematics, Statistics, Engineering)
- authorization: Visa/clearance/security terms (H1B, TS/SCI clearance, security clearance)
- soft_skill: Interpersonal abilities, behavioral traits, leadership qualities, personality traits (problem-solving, communication, collaboration, time management, visionary, strategic thinker, innovative leader, adaptable, organized, committed)
- action: Action verbs/phrases (develop, implement, design, troubleshoot, optimize)
- responsibility: Responsibility phrases (Design & Build, Develop & Deploy)
- reporting_audit: Audit/assessment/reporting terms (audit, assessment, compliance, documentation)
- other: Noise, garbage, URLs, non-technical phrases, experience-phrased items (e.g., "experience with X", "proficiency in X") that should be routed to Experience/Summary sections, anything else

CRITICAL ROUTING RULE: Experience-phrased items (e.g., "experience with Python", "proficiency in data analysis", "familiarity with AWS", "ability to use Docker") should be classified as "other" or "action" and routed to Experience/Summary sections, NOT Hard Skills. Hard Skills section must contain ONLY standalone noun-based skills.

Technical skill flag: True if pure technical term (e.g., "Python", "AWS", "Docker") - standalone noun-based skill, False if descriptive (e.g., "experience with Python", "ability to use AWS", "proficiency in data analysis") - experience-phrased items.

Return classification and technical skill flag for each keyword."""

            user_prompt = f"""Classify the following keywords and determine if each is a technical skill:

{chr(10).join([f'{j+1}. {kw}' for j, kw in enumerate(batch)])}

Return classification and technical skill flag for all {len(batch)} keywords."""

            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",  # OPTIMIZATION: Use faster model for classification
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=BatchTechnicalCheckResult,
                temperature=0.0,
                timeout=8.0,  # OPTIMIZATION: Reduced from 10.0 to 8.0 for faster failure detection
                max_tokens=2000  # OPTIMIZATION: Reduced from 3000 for faster processing
            )
            
            parsed = response.choices[0].message.parsed
            for result in parsed.results:
                batch_results[result.keyword] = (result.classification, result.is_technical_skill)
                
        except Exception as e:
            logger.warning(f"LLM classification batch {batch_num} failed, using defaults: {e}")
            # Default to 'other' and False if LLM fails
            for kw in batch:
                if kw not in batch_results:
                    batch_results[kw] = ("other", False)
        
        return batch_results
    
    # Process all batches in parallel
    if len(batches) == 1:
        # Single batch - no need for parallelization
        return await process_batch(batches[0], 0)
    else:
        # Multiple batches - process in parallel
        batch_tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results_list = await asyncio.gather(*batch_tasks)
        
        # Merge all results
        results = {}
        for batch_results in batch_results_list:
            results.update(batch_results)
        
        return results




class PhraseLibrary:
    """
    Deterministic ATS-safe insertion templates. These are deliberately simple to prevent semantic drift.
    """

    @staticmethod
    def experience_bullet_for(keyword: str, kw_type: KeywordType) -> str:
        """Generate experience bullet point for a keyword based on its type."""
        if kw_type == "soft_skill":
            # Deterministic, ATS-safe: include the keyword verbatim with minimal extra text.
            return f"Demonstrated {keyword}."
        # All other types: verbatim keyword
        return f"{keyword}."


# Removed: _normalize_for_scan - now using normalize_whitespace from utils.py


def build_plain_text(content: ResumeContent) -> str:
    """Optimized: Use list comprehension and single join for better performance."""
    parts: List[str] = []
    if content.profile_summary:
        parts.append(content.profile_summary)
    if content.technical_skills:
        for cat in content.technical_skills:
            if cat.category:
                parts.append(cat.category)
            if cat.skills:
                parts.extend(cat.skills)
    if content.experience:
        for exp in content.experience:
            if exp.role:
                parts.append(exp.role)
            if exp.company:
                parts.append(exp.company)
            if exp.highlights:
                parts.extend(exp.highlights)
    if content.projects:
        for proj in content.projects:
            if proj.name:
                parts.append(proj.name)
            if proj.description:
                parts.extend(proj.description)
    return normalize_whitespace(" ".join(parts)) if parts else ""


def build_skills_text(content: ResumeContent) -> str:
    """Optimized: Use list comprehension and single join for better performance."""
    if not content.technical_skills:
        return ""
    parts: List[str] = []
    for cat in content.technical_skills:
        if cat.category:
            parts.append(cat.category)
        if cat.skills:
            parts.extend(cat.skills)
    return normalize_whitespace(" ".join(parts)) if parts else ""


def build_experience_text(content: ResumeContent) -> str:
    """Optimized: Use list comprehension and single join for better performance."""
    if not content.experience:
        return ""
    parts: List[str] = []
    for exp in content.experience:
        if exp.role:
            parts.append(exp.role)
        if exp.company:
            parts.append(exp.company)
        if exp.highlights:
            parts.extend(exp.highlights)
    return normalize_whitespace(" ".join(parts)) if parts else ""


def count_occurrences(text: str, keyword: str) -> int:
    """
    Optimized literal-ish keyword count. For ATS we want verbatim substrings,
    but we also want word-boundary correctness for single tokens.
    
    Performance optimizations:
    - Early returns for empty inputs
    - Cached keyword normalization
    - Optimized word matching using string methods
    """
    if not keyword or not text:
        return 0
    
    # Normalize keyword once
    needle = keyword.strip()
    if not needle:
        return 0
    
    # Normalize whitespace in keyword to match build_plain_text normalization.
    if " " in needle:
        needle = " ".join(needle.split())
    
    # Fast path: If keyword contains non-word chars, use simple substring count
    # Check first char before iterating (common case optimization)
    if not needle[0].isalnum() or not needle[-1].isalnum():
        return text.count(needle)
    
    # Check if any non-word char exists (optimized: short-circuit on first match)
    has_non_word_char = False
    for c in needle:
        if not c.isalnum() and c != '_':
            has_non_word_char = True
            break
    
    if has_non_word_char:
        return text.count(needle)
    
    # Word-only keywords: use optimized word boundary matching
    # For single words, use faster string method
    if ' ' not in needle:
        # Single word: count occurrences with word boundaries
        # Use split and count (faster than regex for simple cases)
        words = text.split()
        return sum(1 for w in words if w == needle)
    
    # Multi-word: use sliding window (optimized)
    needle_words = needle.split()
    needle_len = len(needle_words)
    hay_words = text.split()
    hay_len = len(hay_words)
    
    if needle_len > hay_len:
        return 0
    
    # Optimized sliding window: compare slices
    count = 0
    for i in range(hay_len - needle_len + 1):
        if hay_words[i:i+needle_len] == needle_words:
            count += 1
    
    return count


# Removed: classify_keyword() - string-based classification
# Now using LLM-based classification with batch processing upfront
# Classification results are cached and passed to functions that need them


def _capitalize_skill_title_case(skill: str) -> str:
    """
    Capitalize skill using Title Case (aligned with existing education capitalization logic):
    - Capitalize all principal words (e.g., "Robotic Programming", "Path Planning")
    - Acronyms must remain fully capitalized (e.g., "WAAM", "KUKA", "AWS", "API", "SQL")
    - Preserve existing proper capitalization if already correct
    - Do NOT lowercase skills under any circumstance
    
    This function is used for injected keywords that may come from job description verbatim
    and need to be formatted to match LLM-generated skills (which are already capitalized per prompt).
    """
    if not skill or not skill.strip():
        return skill
    
    skill_original = skill.strip()
    skill_lower = skill_original.lower()
    
    # If already properly capitalized (Title Case or all caps), preserve it
    if skill_original != skill_lower:
        # Check if it's all uppercase (acronym) - keep it
        if skill_original.isupper():
            return skill_original
        # Check if it's already in Title Case (first letter of each word is uppercase)
        words = skill_original.split()
        if words and all(word[0].isupper() if word else True for word in words):
            # Already properly capitalized - preserve it
            return skill_original
    
    # Common acronyms that should remain fully capitalized (aligned with education logic style)
    common_acronyms = {
        'aws', 'api', 'sql', 'etl', 's3', 'sla', 'sso', 'saml', 'oauth', 'jwt',
        'rest', 'graphql', 'http', 'https', 'tcp', 'udp', 'dns', 'dhcp', 'vpn',
        'iam', 'rds', 'ec2', 'lambda', 'dynamodb', 'redshift',
        'kubernetes', 'docker', 'jenkins', 'git', 'ci', 'cd', 'devops',
        'ml', 'ai', 'nlp', 'cv', 'dl', 'cnn', 'rnn', 'lstm', 'gpu', 'cpu',
        'json', 'xml', 'yaml', 'csv', 'tsv', 'pdf', 'html', 'css', 'js',
        'ui', 'ux', 'sdk', 'ide', 'cli', 'gui', 'tui'
    }
    
    # Split into words and apply capitalization (similar to education capitalization)
    words = skill_original.split()
    capitalized_words = []
    
    for word in words:
        word_lower = word.lower()
        # Remove punctuation for checking
        word_clean = ''.join(c for c in word if c.isalnum()).lower()
        
        # Check if word is an acronym (in list, or all caps short word)
        if word_clean in common_acronyms or (word.isupper() and 2 <= len(word_clean) <= 5):
            capitalized_words.append(word.upper())
        elif '/' in word:
            # Handle slash-separated (e.g., "AI/ML")
            subparts = word.split('/')
            capitalized_subparts = []
            for subpart in subparts:
                subpart_clean = ''.join(c for c in subpart if c.isalnum()).lower()
                if subpart_clean in common_acronyms or (subpart.isupper() and 2 <= len(subpart_clean) <= 5):
                    capitalized_subparts.append(subpart.upper())
                else:
                    capitalized_subparts.append(subpart.capitalize())
            capitalized_words.append('/'.join(capitalized_subparts))
        else:
            # Regular word - capitalize first letter (like education capitalization)
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)


def _ensure_skill_category(content: ResumeContent, category_name: str) -> SkillCategory:
    for cat in content.technical_skills:
        if cat.category.strip().lower() == category_name.strip().lower():
            return cat
    new_cat = SkillCategory(category=category_name, skills=[])
    content.technical_skills.append(new_cat)
    return new_cat


async def inject_missing_keywords(
    content: ResumeContent,
    missing: Set[str],
    classification_cache: Dict[str, KeywordType],
    technical_check_cache: Dict[str, bool],
    required_technical_skills: Set[str] = None,
) -> Dict[str, List[str]]:
    """
    Inject missing keywords into safe sections using LLM-based classification.
    - PURE technical skills => Technical Skills section
    - Everything else => Experience bullets
    """
    injected: Dict[str, List[str]] = {"skills": [], "experience": []}

    # Ensure lists exist
    if content.technical_skills is None:
        content.technical_skills = []
    if content.experience is None:
        content.experience = []

    skills_cat = _ensure_skill_category(content, "Technical Skills")

    # Ensure at least 1 experience entry for safe injection
    if not content.experience:
        content.experience.append(Experience(company="", role="", duration="", highlights=[]))

    exp0 = content.experience[0]
    if exp0.highlights is None:
        exp0.highlights = []

    # Build set of all existing skills across ALL categories (for duplicate checking)
    # Optimized: Single pass with list comprehension for better performance
    all_existing_skills = set()
    if content.technical_skills:
        for cat in content.technical_skills:
            if cat.skills:
                for skill in cat.skills:
                    if skill:
                        skill_lower = str(skill).lower().strip()
                        all_existing_skills.add(skill_lower)
                        # Also add main skill name (before parentheses) for flexible matching
                        skill_main = skill_lower.split('(')[0].strip()
                        if skill_main:
                            all_existing_skills.add(skill_main)
    
    # Pre-compute technical skill types set for faster lookup
    # CRITICAL: Exclude "education" and "authorization" - these should NOT be in Skills section
    technical_types = {"framework", "artifact", "tool", "hard_skill"}
    
    # Inject deterministically (stable order)
    pending_experience_keywords: List[str] = []
    skipped_technical = []  # Track skills that should be technical but aren't being injected
    for kw in sorted(missing):
        if not kw:
            continue
        
        kw_type = classification_cache.get(kw, "other")
        is_technical = technical_check_cache.get(kw, False)
        
        # CRITICAL: Exclude education, authorization, and vague terms from Skills section
        # Even if classified as technical, filter out non-technical entries
        if is_non_technical_entry(kw):
            pending_experience_keywords.append(kw)
            continue
        
        # ALL technical skills go to Technical Skills section (excluding education/authorization)
        # Also include skills that are in required_technical_skills (from job description hard skills)
        is_required_technical = required_technical_skills and kw in required_technical_skills
        should_go_to_skills = (kw_type in technical_types or 
                              (is_technical and kw_type not in ("education", "authorization")) or
                              is_required_technical)
        
        if should_go_to_skills:
            kw_lower = kw.lower().strip()
            kw_main = kw_lower.split('(')[0].strip()
            
            # Check for exact matches only (less aggressive duplicate detection)
            # Only skip if exact match exists (case-insensitive) or if it's a sub-skill of an existing skill
            already_exists = False
            
            # Check exact match (case-insensitive)
            if kw_lower in all_existing_skills or kw in skills_cat.skills:
                already_exists = True
            # Check if main part (before parentheses) matches exactly
            elif kw_main in all_existing_skills and len(kw_main) > 3:
                # Only consider it a duplicate if the main part is substantial (more than 3 chars)
                # and matches exactly (not substring)
                already_exists = True
            # Only check substring match for very short skills (2-3 chars) to avoid false positives
            elif len(kw_main) <= 3:
                # For very short skills, check if exact match exists
                already_exists = kw_main in all_existing_skills
            # For longer skills, be more lenient - only skip if it's clearly a duplicate
            # (e.g., "AWS" vs "AWS S3" should both be included)
            
            if not already_exists:
                # Capitalize keyword using Title Case before adding
                kw_capitalized = _capitalize_skill_title_case(kw)
                skills_cat.skills.append(kw_capitalized)
                injected["skills"].append(kw_capitalized)
                # Update tracking set (use original for matching, but store capitalized)
                all_existing_skills.add(kw_lower)
                if kw_main:
                    all_existing_skills.add(kw_main)
                all_existing_skills.add(kw_capitalized.lower())
            else:
                # Log why skill was skipped (for debugging)
                logger.debug(f"Skipping skill '{kw}' - already exists in skills section")
        else:
            # Skill not going to skills section - check if it should be
            if is_required_technical or (kw_type in technical_types and kw_type not in ("education", "authorization")):
                skipped_technical.append(f"{kw} (type={kw_type}, technical={is_technical}, required={is_required_technical})")
            pending_experience_keywords.append(kw)
    
    # Log any technical skills that weren't injected into skills section
    if skipped_technical:
        logger.warning(f"WARNING: {len(skipped_technical)} technical skills were not injected into Skills section (routed to Experience instead): {', '.join(skipped_technical[:10])}{'...' if len(skipped_technical) > 10 else ''}")

    # Batch all non-technical keywords into experience bullets
    if pending_experience_keywords:
        soft_skills = []
        actions = []
        other_kws = []
        
        for kw in pending_experience_keywords:
            kw_type = classification_cache.get(kw, "other")
            if kw_type == "soft_skill":
                soft_skills.append(kw)
            elif kw_type in ("action", "responsibility", "reporting_audit"):
                actions.append(kw)
            else:
                other_kws.append(kw)
        
        # Add soft skills
        for soft_skill in soft_skills:
            bullet = PhraseLibrary.experience_bullet_for(soft_skill, "soft_skill")
            if bullet not in exp0.highlights:
                exp0.highlights.append(bullet)
            injected["experience"].append(soft_skill)
        
        if len(soft_skills) > 3:
            combined_bullet = f"Demonstrated {', '.join(str(s) for s in soft_skills[:8] if s)}{'...' if len(soft_skills) > 8 else ''} in professional environments."
            if combined_bullet not in exp0.highlights:
                exp0.highlights.append(combined_bullet)
        
        # Add action/responsibility keywords
        for kw in actions:
            kw_type = classification_cache.get(kw, "other")
            bullet = PhraseLibrary.experience_bullet_for(kw, kw_type)
            if bullet not in exp0.highlights:
                exp0.highlights.append(bullet)
                injected["experience"].append(kw)
        
        # Add remaining keywords
        if other_kws:
            for kw in other_kws:
                bullet = f"{kw}."
                if bullet not in exp0.highlights:
                    exp0.highlights.append(bullet)
                injected["experience"].append(kw)

    # SKILL REMOVAL DISABLED - Do not remove any skills
    # User requirement: Do not remove any skills after extraction
    # Skills are already filtered in extractor.py, so no need to sanitize again
    # removed_from_skills = []  # No skills removed

    return injected


def enforce_frequency_limits(text: str, required_keywords: Set[str], classification_cache: Dict[str, KeywordType]) -> Dict[str, List[str]]:
    """
    Deterministic frequency telemetry. We DO NOT rewrite content here; we only report violations.
    Pruning to address stuffing is handled by the one-page/pruning step.
    """
    under: List[str] = []
    over: List[str] = []
    for kw in required_keywords:
        kw_type = classification_cache.get(kw, "other")
        lim = DEFAULT_LIMITS.get(kw_type, KeywordLimits(1, 1))
        c = count_occurrences(text, kw)
        if c < lim.min_count:
            under.append(f"{kw} ({c}<{lim.min_count})")
        if c > lim.max_count:
            over.append(f"{kw} ({c}>{lim.max_count})")
    return {"under": sorted(under), "over": sorted(over)}


def compute_coverage(text: str, required_keywords: Set[str]) -> Tuple[float, Set[str], Set[str]]:
    present: Set[str] = set()
    missing: Set[str] = set()
    for kw in required_keywords:
        if count_occurrences(text, kw) > 0:
            present.add(kw)
        else:
            missing.add(kw)
    total = len(required_keywords)
    cov = (len(present) / total) if total else 1.0
    return cov, present, missing


class ATSPreflight:
    """
    Runs deterministic ATS validation and deterministic repair (injection only).
    """

    def __init__(self, min_coverage: float = 0.80):
        # Lower default threshold (0.80 instead of 0.95) to be more lenient
        # Validation reports issues but never blocks resume generation
        self.min_coverage = min_coverage

    async def run(self, content: ResumeContent, required_keywords: Iterable[str]) -> Tuple[ResumeContent, Dict[str, object]]:
        required_set = {k.strip() for k in required_keywords if k and k.strip()}
        
        # Batch classify all keywords upfront using LLM
        logger.info(f"Batch classifying {len(required_set)} keywords using LLM...")
        keywords_list = list(required_set)
        classification_results = await classify_keywords_with_technical_check_llm(keywords_list)
        
        # Build classification and technical check caches
        classification_cache: Dict[str, KeywordType] = {}
        technical_check_cache: Dict[str, bool] = {}
        for kw, (kw_type, is_technical) in classification_results.items():
            classification_cache[kw] = kw_type
            technical_check_cache[kw] = is_technical

        text0 = build_plain_text(content)
        skills0 = build_skills_text(content)
        exp0 = build_experience_text(content)
        cov0, present0, missing0 = compute_coverage(text0, required_set)

        # Placement enforcement using LLM classification
        placement_missing: Set[str] = set()
        for kw in required_set:
            kw_type = classification_cache.get(kw, "other")
            is_technical = technical_check_cache.get(kw, False)
            
            # All technical skills go to Skills section
            if kw_type in ("hard_skill", "tool", "education", "framework", "artifact", "authorization") or is_technical:
                if count_occurrences(skills0, kw) == 0:
                    placement_missing.add(kw)
            elif kw_type in ("soft_skill", "action", "responsibility", "reporting_audit"):
                if count_occurrences(exp0, kw) == 0:
                    placement_missing.add(kw)
            elif kw_type == "other":
                # Use technical check from LLM
                if is_technical:
                    if count_occurrences(skills0, kw) == 0:
                        placement_missing.add(kw)
                else:
                    if count_occurrences(exp0, kw) == 0:
                        placement_missing.add(kw)

        missing_for_injection = set(missing0) | placement_missing

        logger.info(
            f"ATS preflight: initial coverage={cov0:.2%} (required={len(required_set)}, "
            f"missing_anywhere={len(missing0)}, missing_wrong_section={len(placement_missing)})"
        )
        if missing_for_injection:
            sample = sorted(missing_for_injection)[:25]
            logger.info(f"ATS missing/placement-missing (sample): {', '.join(str(k) for k in sample if k)}{'...' if len(missing_for_injection) > 25 else ''}")

        injected = {"skills": [], "experience": []}
        # Build required_technical_skills for protection (used in both injection and sanitization)
        required_technical_skills = {
            kw for kw in required_set 
            if classification_cache.get(kw, "other") in ("hard_skill", "tool", "education", "framework", "artifact", "authorization") 
            or technical_check_cache.get(kw, False)
        }
        
        if missing_for_injection:
            injected = await inject_missing_keywords(content, missing_for_injection, classification_cache, technical_check_cache, required_technical_skills)
            inj_total = len(injected.get("skills", [])) + len(injected.get("experience", []))
            logger.info(
                f"ATS injected: total={inj_total} "
                f"(skills={len(injected.get('skills', []))}, experience={len(injected.get('experience', []))})"
            )
            if injected.get("skills"):
                logger.info(f"Injected skills: {', '.join(str(s) for s in injected.get('skills', [])[:20])}{'...' if len(injected.get('skills', [])) > 20 else ''}")
            if injected.get("experience"):
                logger.info(f"Injected experience keywords (sample): {', '.join(str(k) for k in injected.get('experience', [])[:20])}{'...' if len(injected.get('experience', [])) > 20 else ''}")

        # SKILL SANITIZATION DISABLED - Do not remove any skills
        # User requirement: Do not remove any skills after extraction
        # Skills are already filtered in extractor.py, so no need to sanitize again
        total_skills = sum(len(cat.skills) for cat in content.technical_skills or [])
        categories_count = len([cat for cat in content.technical_skills or [] if cat.skills and len(cat.skills) > 0])
        logger.info(f"ATS Preflight: Skill sanitization DISABLED - preserving all {total_skills} skills in {categories_count} categories (no skills will be removed)")
        
        # No skill removal - keep all skills as-is
        # removed_skill_actions is empty, so the code below that processes removed skills will not execute

        text1 = build_plain_text(content)
        skills1 = build_skills_text(content)
        exp1 = build_experience_text(content)
        cov1, present1, missing1 = compute_coverage(text1, required_set)

        # Placement violations after injection
        placement_violations: List[str] = []
        for kw in required_set:
            kw_lower = kw.lower() if kw else ""
            if not kw or not kw.strip() or len(kw.strip()) < 3:
                continue
            # Skip obvious noise (simple string checks for very basic patterns)
            noise_indicators = ['compensation', 'benefits', 'insurance', 'equal opportunity', 'privacy policy', 
                              'applicants', 'consideration', 'achievements', 'awards', 'referral', 'full-time', 
                              'part-time', 'contract', 'internship', '@', 'http', 'www.', 'salary', 'hourly']
            if any(indicator in kw_lower for indicator in noise_indicators):
                continue
                
            kw_type = classification_cache.get(kw, "other")
            is_technical = technical_check_cache.get(kw, False)
            
            # All technical skills MUST be in Skills section
            if kw_type in ("hard_skill", "tool", "education", "framework", "artifact", "authorization") or is_technical:
                if count_occurrences(skills1, kw) == 0:
                    # Allow flexibility for action-containing technical phrases
                    action_words_in_kw = ['administer', 'administration', 'tasks', 'migration', 'migrations',
                                         'operation', 'operations', 'procedure', 'procedures', 'tuning', 'optimization']
                    if not any(word in kw_lower for word in action_words_in_kw):
                        placement_violations.append(f"{kw} -> missing in Skills/Qualifications")
            
            elif kw_type in ("soft_skill", "action", "responsibility", "reporting_audit"):
                if count_occurrences(exp1, kw) == 0:
                    summary_text = (content.profile_summary or "").lower()
                    if kw.lower() not in summary_text:
                        placement_violations.append(f"{kw} -> missing in Experience/Summary")
            
            elif kw_type == "other":
                if is_technical:
                    if count_occurrences(skills1, kw) == 0 and count_occurrences(exp1, kw) == 0:
                        placement_violations.append(f"{kw} -> missing in Skills/Experience")
                else:
                    if count_occurrences(exp1, kw) == 0:
                        placement_violations.append(f"{kw} -> missing in Experience")

        # ALL REMOVAL LOGIC DISABLED - Do not remove any skills or categories
        # User requirement: Do not remove any skills after extraction
        # The LLM generates skills correctly - trust it completely
        # DO NOT deduplicate, DO NOT remove categories, DO NOT modify skills in any way
        total_skills = sum(len(cat.skills) for cat in content.technical_skills or [])
        categories_count = len([cat for cat in content.technical_skills or [] if cat.skills and len(cat.skills) > 0])
        logger.info(f"ATS Preflight: ALL removal/filtering/deduplication logic DISABLED - preserving all {categories_count} categories with {total_skills} total skills exactly as generated by LLM")

        freq_report = enforce_frequency_limits(text1, required_set, classification_cache)

        report: Dict[str, object] = {
            "required_count": len(required_set),
            "coverage_before": cov0,
            "coverage_after": cov1,
            "missing_before": sorted(missing0),
            "missing_after": sorted(missing1),
            "injected": injected,
            "frequency": freq_report,
            "placement_violations": placement_violations,
        }

        # VALIDATION: Always pass - validation should report issues but never block resume generation
        # The user requirement is to ALWAYS get a resume output, even if some keywords are missing
        # We log warnings for quality tracking, but don't block generation
        
        # Check for issues (for reporting/warnings only)
        critical_placement_violations = [v for v in placement_violations if "Skills/Qualifications" in v]
        has_coverage_issue = cov1 < self.min_coverage
        has_placement_issues = len(critical_placement_violations) > 0
        
        # Always return ok=True to ensure resume generation proceeds
        report["ok"] = True
        
        # Log warnings for quality tracking (but don't fail)
        if has_coverage_issue:
            logger.warning(
                f"ATS coverage below threshold ({cov1:.2%} < {self.min_coverage:.2%}): "
                f"{len(missing1)} keywords missing. Resume will be generated anyway."
            )
        if has_placement_issues:
            logger.warning(
                f"ATS placement violations: {len(critical_placement_violations)} critical violations. "
                f"Resume will be generated anyway."
            )
        
        return content, report
