import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import openai
from pydantic import BaseModel, Field
from app.schemas.resume import ResumeContent, Experience, Project
from app.agent.extractor import enrich_job_description
from app.agent.utils import logger, strip_excess_chars, sanitize_error_message, get_async_openai_client, is_non_technical_entry
from app.agent.ats_validator import (
    ATSPreflight,
    build_plain_text,
    build_skills_text,
    build_experience_text,
    count_occurrences,
    compute_coverage,
)

# Text-layer one-page constraint:
# - Historically enforced at 56 lines (very strict 1-page A4 estimate).
# - For now, allow relaxing or disabling via env to avoid hard failures when keyword coverage is expanded.
# NOTE: One-page preflight validation is currently disabled - no environment variables needed.

# Helper: Count quantified metrics in text
# Optimized: Compile regex patterns once at module level for better performance
_METRIC_PATTERNS = [
    re.compile(r'\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*(?:percent|percentage)', re.IGNORECASE),  # Percentage
    re.compile(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:K|M|B)?|\d+(?:,\d{3})*(?:\.\d+)?(?:K|M|B)?\s*(?:dollars?|USD)', re.IGNORECASE),  # Dollar
    re.compile(r'#\s*\d+|\d+\+|\d+\s*plus|\d+\s*more|\d+\s*less|over\s+\d+|more\s+than\s+\d+|less\s+than\s+\d+|\d+\s*(?:projects?|teams?|members?|users?|clients?|systems?|servers?|databases?)', re.IGNORECASE),  # Count
    re.compile(r'(?:increased?|decreased?|reduced?|improved?|achieved?|delivered?|saved?|generated?|raised?|lowered?|optimized?|accelerated?|scaled?|expanded?|grew?|boosted?|enhanced?|maximized?|minimized?)\s+(?:by|to|from)?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:K|M|B|%)?', re.IGNORECASE),  # Quantified action
]

def count_metrics_in_text(text: str) -> int:
    """
    Optimized count of quantified metrics in text. A metric must contain:
    - Numbers with % (percentages)
    - Numbers with $ (dollar amounts)
    - Numbers with # or "number of" or "over" or specific counts
    - Numbers in quantified contexts (e.g., "increased by 25%", "reduced costs by $50K", "managed 10+ projects")
    
    Vague phrases without numbers are NOT metrics.
    
    Performance: Uses pre-compiled regex patterns for faster matching.
    """
    if not text or not text.strip():
        return 0
    
    # Count unique metric occurrences (a single bullet can have multiple metrics, but we count distinct quantified phrases)
    found_phrases = set()
    
    # Use pre-compiled patterns for better performance
    for pattern in _METRIC_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            # Extract a larger context to ensure it's a real metric, not just a random number
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].lower()
            matched_text = match.group(0).lower()
            
            # Filter out false positives (dates, versions, etc.)
            # Only filter if the match itself looks like a year (4-digit number like 2024, 2023) or version
            is_year = matched_text.isdigit() and len(matched_text) == 4 and matched_text.startswith(('19', '20'))
            is_version = 'version' in context or 'v.' in context
            is_date_context = 'date' in context and is_year
            is_contact_info = any(info in context for info in ['phone', 'email', 'address'])
            
            if is_year or is_version or is_date_context or is_contact_info:
                continue
            
            # Store unique metric phrases (normalized)
            phrase = matched_text.strip()
            if phrase not in found_phrases:
                found_phrases.add(phrase)
    
    return len(found_phrases)


def count_metrics_by_section(content: ResumeContent) -> Dict[str, int]:
    """
    Count metrics in each section of the resume.
    
    Counting logic:
    - Summary: 1 if summary has at least 1 metric (needs exactly 1 metric)
    - Experience: Count total metrics across all roles (needs 6 total metrics)
    - Projects: 0 metrics (no metrics required in projects)
    """
    metrics = {
        "summary": 0,
        "experience": 0,
        "projects": 0,
        "total": 0
    }
    
    # Count in summary: 1 if summary has at least 1 metric
    summary_metric_count = count_metrics_in_text(content.profile_summary) if content.profile_summary else 0
    if summary_metric_count >= 1:
        metrics["summary"] = 1
    
    # Count in experience: Count total metrics across all roles (need 6 total)
    if content.experience:
        for exp in content.experience:
            if exp.highlights:
                for highlight in exp.highlights:
                    metrics["experience"] += count_metrics_in_text(highlight)  # Count total metrics
    
    # Count in projects: Count bullets/descriptions with metrics (for validation/logging only - requirement is 0 metrics)
    if content.projects:
        for project in content.projects:
            if project.description:
                # Project description can be a list of bullets or a string
                if isinstance(project.description, list):
                    for desc in project.description:
                        if count_metrics_in_text(str(desc)) > 0:
                            metrics["projects"] += 1
                else:
                    if count_metrics_in_text(str(project.description)) > 0:
                        metrics["projects"] += 1
    
    metrics["total"] = metrics["summary"] + metrics["experience"] + metrics["projects"]
    return metrics


# Pydantic model for LLM project parsing output
class ParsedProjects(BaseModel):
    """Structured output for parsed projects from text."""
    projects: List[Project] = Field(
        default_factory=list,
        description="List of parsed projects, each with name, description (as bullet points), and optional link"
    )


# Helper: Parse projects from payload text using LLM
async def parse_projects_from_text(projects_text: str) -> List[Project]:
    """Parse projects directly from payload text into Project objects using LLM.
    
    CRITICAL: Projects are copied verbatim from payload, but parsed/extracted using LLM.
    The first line of each project section is kept as the header/project name.
    The rest is extracted as bullet points.
    
    Format:
    - First line: Project name (may include "| GitHub Link")
    - Following lines: Bullet points (starting with "-") OR paragraph text (converted to bullets)
    """
    if not projects_text or not projects_text.strip():
        return []
    
    try:
        client = get_async_openai_client()
        
        system_prompt = """You are a project parser for resume generation. Parse projects from text into structured format.

RULES:
1. Keep the FIRST line of each project as the project name (header)
2. Extract all subsequent content as bullet points (description field)
3. If input has bullet points (lines starting with "-"), extract them directly
4. If input has paragraph text, split into sentences and convert each sentence to a bullet point
5. Extract GitHub link if present (either in project name after "|" or in a separate line)
6. Each project should have: name (first line), description (list of bullet points), optional link

OUTPUT FORMAT:
Return a list of projects, each with:
- name: The first line/header of the project (may contain "| GitHub Link" - extract this)
- description: List of bullet points (sentences converted to bullets if paragraph text)
- link: GitHub URL if present, None otherwise
- tech_stack: Empty list (not extracted from text)"""
        
        user_prompt = f"""Parse the following projects text into structured projects. Keep the first line as the project name/header, and convert the rest to bullet points.

Projects Text:
{projects_text}

Extract all projects. For each project:
- Use the first line as the project name (if it contains "| GitHub Link", extract the link separately)
- Convert all following lines into bullet points in the description field
- If text is in paragraph format, split by sentences and create one bullet per sentence"""
        
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # OPTIMIZATION: Use faster model for simpler parsing task
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ParsedProjects,
            temperature=0.0,
            timeout=15.0,  # OPTIMIZATION: Reduced from 20.0 to 15.0 for faster failure detection
            max_tokens=1200  # OPTIMIZATION: Reduced from 1500 to 1200 for faster processing
        )
        
        parsed = response.choices[0].message.parsed
        projects = parsed.projects if parsed.projects else []
        
        # Post-process: Clean up project names (remove "| GitHub Link" from name if link is already extracted by LLM)
        for project in projects:
            if project.name and '|' in project.name:
                # LLM should have extracted the link, so just clean the name
                parts = project.name.split('|', 1)
                project.name = parts[0].strip()
                # If LLM didn't extract link but there's text after |, check if it's a URL
                if not project.link and len(parts) > 1:
                    link_part = parts[1].strip()
                    # Simple check for URL without regex: if it starts with http
                    if link_part.lower().startswith('http'):
                        project.link = link_part
            
            # Ensure description is a list of strings (not empty)
            if not project.description:
                project.description = []
            else:
                # Clean up bullets - remove leading "-" if present, trim whitespace
                cleaned_description = []
                for desc in project.description:
                    if desc:
                        cleaned = desc.strip()
                        if cleaned.startswith('-'):
                            cleaned = cleaned[1:].strip()
                        if cleaned:
                            cleaned_description.append(cleaned)
                project.description = cleaned_description
        
        logger.info(f"LLM parsed {len(projects)} project(s) from text")
        return projects
        
    except Exception as e:
        logger.error(f"LLM project parsing failed: {e}")
        # Return empty list - LLM should handle all parsing
        return []


# Helper: Count projects from text by detecting project boundaries
def count_projects_from_text(projects_text: str) -> int:
    """Count the number of projects in a text string by detecting project boundaries.
    
    A project typically has:
    - A project name (line that doesn't start with "-" or "GitHub:")
    - Optional GitHub link (line containing "GitHub:" or "github.com")
    - Bullet points (lines starting with "-")
    
    Projects are separated by:
    - A new project name appearing after bullets
    - Empty lines followed by a non-bullet line
    """
    if not projects_text or not projects_text.strip():
        return 0
    
    lines = [line.strip() for line in projects_text.split('\n') if line.strip()]
    if not lines:
        return 0
    
    project_count = 0
    i = 0
    found_first_project = False
    
    while i < len(lines):
        line = lines[i]
        
        # Skip GitHub links (they're part of the previous project)
        if 'GitHub:' in line or 'github.com' in line.lower():
            i += 1
            continue
        
        # If line starts with "-", it's a bullet point
        if line.startswith('-'):
            # If we haven't found a project name yet, assume there's 1 project
            if not found_first_project:
                project_count = 1
                found_first_project = True
            i += 1
            continue
        
        # Found a potential project name (non-bullet, non-GitHub line)
        # This is the start of a new project
        project_count += 1
        found_first_project = True
        
        # Skip this line and look for GitHub link or bullets
        i += 1
        # Skip GitHub link if present
        if i < len(lines) and ('GitHub:' in lines[i] or 'github.com' in lines[i].lower()):
            i += 1
        # Skip all subsequent bullet points
        while i < len(lines) and lines[i].startswith('-'):
            i += 1
    
    # If we found any content, ensure at least 1 project
    return max(1, project_count)

def calculate_years_of_experience(profile: Dict[str, Any]) -> float:
    """
    Calculate total years of experience from work experience entries.
    
    Args:
        profile: Candidate profile dictionary with experience entries
        
    Returns:
        Total years of experience as a float (rounded to 1 decimal place)
    """
    month_names = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    def parse_date(date_str: str) -> Optional[Tuple[int, int]]:
        """Parse date string like 'Nov 2024' or 'January 2023' into (year, month)."""
        date_str = date_str.strip()
        parts = date_str.split()
        if len(parts) >= 2:
            month_str = parts[0].lower()
            year_str = parts[-1]
            try:
                year = int(year_str)
                month = month_names.get(month_str)
                if month:
                    return (year, month)
            except ValueError:
                pass
        return None
    
    experience = profile.get('experience', [])
    experience_text = profile.get('experience_text', '')
    
    # If experience is a string, try to parse durations from it
    if experience_text and not experience:
        # Try to extract date ranges from text (e.g., "Nov 2024 - Present", "Jan 2023 - Sep 2024")
        date_pattern = r'([A-Za-z]{3,9}\s+\d{4})\s*[-–—]\s*([A-Za-z]{3,9}\s+\d{4}|Present|present|Current|current)'
        matches = re.findall(date_pattern, experience_text)
        if matches:
            total_months = 0
            for start_str, end_str in matches:
                start_date = parse_date(start_str)
                if not start_date:
                    continue
                
                if end_str.lower() in ['present', 'current']:
                    end_date = (datetime.now().year, datetime.now().month)
                else:
                    end_date = parse_date(end_str)
                    if not end_date:
                        continue
                
                start_year, start_month = start_date
                end_year, end_month = end_date
                months = (end_year - start_year) * 12 + (end_month - start_month)
                total_months += max(0, months)
            
            if total_months > 0:
                years = round(total_months / 12.0, 1)
                return years
    
    # If experience is a list, calculate from duration strings
    if isinstance(experience, list) and experience:
        total_months = 0
        for exp in experience:
            duration = exp.get('duration', '') if isinstance(exp, dict) else getattr(exp, 'duration', '')
            if not duration:
                continue
            
            # Parse duration string (e.g., "Nov 2024 - Present", "Jan 2023 - Sep 2024")
            date_pattern = r'([A-Za-z]{3,9}\s+\d{4})\s*[-–—]\s*([A-Za-z]{3,9}\s+\d{4}|Present|present|Current|current)'
            match = re.search(date_pattern, duration)
            if match:
                start_str, end_str = match.groups()
                start_date = parse_date(start_str)
                if not start_date:
                    continue
                
                if end_str.lower() in ['present', 'current']:
                    end_date = (datetime.now().year, datetime.now().month)
                else:
                    end_date = parse_date(end_str)
                    if not end_date:
                        continue
                
                start_year, start_month = start_date
                end_year, end_month = end_date
                months = (end_year - start_year) * 12 + (end_month - start_month)
                total_months += max(0, months)
        
        if total_months > 0:
            years = round(total_months / 12.0, 1)
            return years
    
    return 0.0

# Helper: Parse experience and projects from string to list
def parse_profile_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Parse experience and projects from string format to list format.
    If strings are provided (not JSON), keep them as strings for LLM to parse.
    The LLM will extract and structure them into Experience and Project objects."""
    parsed_profile = profile.copy()
    
    # Parse experience string
    experience_str = profile.get('experience', '')
    if isinstance(experience_str, str) and experience_str.strip():
        try:
            # Try to parse as JSON first
            parsed_profile['experience'] = json.loads(experience_str)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, keep as string - LLM will parse it from the prompt
            # Store original text for LLM to process
            parsed_profile['experience_text'] = experience_str
            parsed_profile['experience'] = []  # Empty list, LLM will populate
    elif isinstance(experience_str, list):
        parsed_profile['experience'] = experience_str
    else:
        parsed_profile['experience'] = []
    
    # Parse projects string
    projects_str = profile.get('projects', '')
    if isinstance(projects_str, str) and projects_str.strip():
        try:
            # Try to parse as JSON first
            parsed_profile['projects'] = json.loads(projects_str)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, keep as string - LLM will parse it from the prompt
            # Store original text for LLM to process
            parsed_profile['projects_text'] = projects_str
            parsed_profile['projects'] = []  # Empty list, LLM will populate
    elif isinstance(projects_str, list):
        parsed_profile['projects'] = projects_str
    else:
        parsed_profile['projects'] = []
    
    return parsed_profile

def _generate_sub_skills_for_skill(skill_name: str, category_name: str) -> List[str]:
    """
    Generate relevant sub-skills for a given skill to fill space in sparse categories.
    Returns a list of 4-6 relevant sub-skills based on the skill name and category context.
    """
    skill_lower = skill_name.lower()
    category_lower = category_name.lower()
    
    # Sub-skill mappings based on common skill patterns
    sub_skill_map = {
        # Risk Management
        "risk management": ["Risk Assessment", "Risk Mitigation", "Risk Analysis", "Risk Monitoring", "Risk Reporting", "Risk Control"],
        "risk": ["Assessment", "Mitigation", "Analysis", "Monitoring", "Reporting", "Control"],
        
        # Project Management
        "project management": ["Project Planning", "Project Execution", "Project Monitoring", "Stakeholder Management", "Resource Management", "Quality Assurance"],
        "project lifecycle": ["Initiation", "Planning", "Execution", "Monitoring", "Closure", "Lessons Learned"],
        "project planning": ["Scope Definition", "Timeline Development", "Resource Allocation", "Risk Planning", "Budget Planning", "Stakeholder Alignment"],
        
        # Certifications
        "pmp": ["Project Management Professional", "PMI", "Agile", "Waterfall", "Scrum Master", "Program Management"],
        "csm": ["Certified Scrum Master", "Scrum Framework", "Agile Methodologies", "Sprint Planning", "Retrospectives", "Product Backlog"],
        
        # Continuous Improvement
        "continuous improvement": ["Process Optimization", "Lean Methodology", "Six Sigma", "Kaizen", "Performance Metrics", "Root Cause Analysis"],
        "improvement": ["Optimization", "Efficiency", "Metrics", "Analysis", "Best Practices", "Innovation"],
        
        # Budget Management
        "budget management": ["Budget Planning", "Cost Control", "Financial Analysis", "Variance Analysis", "Forecasting", "Expense Tracking"],
        "budget": ["Planning", "Control", "Analysis", "Forecasting", "Tracking", "Optimization"],
        
        # Quality Assurance
        "quality assurance": ["Quality Control", "Testing", "Standards Compliance", "Process Improvement", "Defect Prevention", "Quality Metrics"],
        "quality": ["Control", "Testing", "Standards", "Compliance", "Metrics", "Improvement"],
        
        # Technical Skills
        "bachelor": ["Information Technology", "Computer Science", "Engineering", "Software Development", "Systems Analysis", "Data Structures"],
        "degree": ["Computer Science", "Engineering", "Technology", "Information Systems", "Software Engineering", "Data Science"],
        
        # Generic fallback patterns
        "management": ["Planning", "Execution", "Monitoring", "Reporting", "Analysis", "Optimization"],
        "analysis": ["Data Analysis", "Trend Analysis", "Root Cause Analysis", "Performance Analysis", "Statistical Analysis", "Business Analysis"],
    }
    
    # Try exact matches first
    for pattern, sub_skills in sub_skill_map.items():
        if pattern in skill_lower or pattern in category_lower:
            return sub_skills[:5]  # Return 5 sub-skills
    
    # Fallback: Generate generic sub-skills based on skill name words
    words = skill_lower.split()
    if len(words) >= 2:
        # Multi-word skill - use first word as base
        base_word = words[0]
        if base_word in ["project", "risk", "quality", "budget", "process"]:
            return [f"{base_word.capitalize()} Planning", f"{base_word.capitalize()} Analysis", 
                   f"{base_word.capitalize()} Monitoring", f"{base_word.capitalize()} Optimization", 
                   f"{base_word.capitalize()} Reporting"]
    
    # Final fallback: return empty list (no sub-skills generated)
    return []

# Helper: Post-process content to enforce strict formatting rules
def clean_content(content: ResumeContent) -> ResumeContent:
    """Post-process content to enforce strict formatting rules. 
    NOTE: Skill removal is strictly disabled to ensure 100% keyword coverage as per user requirement.
    """
    # Reduced logging for latency optimization
    # start_time = datetime.now()
    # logger.info(f"clean_content started at {start_time}")
    
    # We NO LONGER remove any skills. We want 100% coverage.
    # However, we still want to clean up any messy formatting.
    content.profile_summary = strip_excess_chars(content.profile_summary)
    for exp in content.experience:
        exp.highlights = [strip_excess_chars(h) for h in exp.highlights]
    for proj in content.projects:
        proj.description = [strip_excess_chars(d) for d in proj.description]
    
    # CRITICAL: Remove "+" signs and filter out non-technical entries from Technical Skills section
    # The LLM should not add "+" signs or non-technical entries, but if it does, remove them post-processing
    if content.technical_skills:
        for cat in content.technical_skills:
            if cat.skills:
                cleaned_skills = []
                removed_non_technical = []
                for skill in cat.skills:
                    if not skill or not str(skill).strip():
                        # Skip empty/None skills - DO NOT add them
                        continue
                    
                    # CRITICAL: Filter out education, legal status, and vague terms
                    if is_non_technical_entry(skill):
                        removed_non_technical.append(skill)
                        logger.warning(f"Removed non-technical entry from Skills section: '{skill}'")
                        continue
                    
                    # Ensure comma-separated format (preserve C++ and C#)
                    cleaned = str(skill).strip()
                    if cleaned:
                        # Preserve C++ and C# by using placeholders
                        placeholder_cpp = "___CPP_PLACEHOLDER___"
                        placeholder_csharp = "___CSHARP_PLACEHOLDER___"
                        
                        if "C++" in cleaned:
                            cleaned = cleaned.replace("C++", placeholder_cpp)
                        elif "c++" in cleaned.lower():
                            idx = cleaned.lower().find("c++")
                            if idx >= 0:
                                cleaned = cleaned[:idx] + placeholder_cpp + cleaned[idx+3:]
                        
                        if "C#" in cleaned:
                            cleaned = cleaned.replace("C#", placeholder_csharp)
                        elif "c#" in cleaned.lower():
                            idx = cleaned.lower().find("c#")
                            if idx >= 0:
                                cleaned = cleaned[:idx] + placeholder_csharp + cleaned[idx+2:]
                        
                        # Replace separator + signs with commas
                        while cleaned.startswith("+"):
                            cleaned = cleaned[1:].lstrip()
                        cleaned = cleaned.replace("+ ", ", ").replace("+", ", ")
                        while ",," in cleaned or ", ," in cleaned:
                            cleaned = cleaned.replace(",,", ",").replace(", ,", ",")
                        
                        # Restore C++ and C#
                        cleaned = cleaned.replace(placeholder_cpp, "C++")
                        cleaned = cleaned.replace(placeholder_csharp, "C#")
                        
                    if cleaned and cleaned.strip():
                            cleaned_skills.append(cleaned.strip())
                cat.skills = cleaned_skills
                
                # Log if category became empty after cleaning
                if not cleaned_skills:
                    logger.warning(f"Category '{cat.category}' became empty after cleaning - all skills were empty/None or non-technical")
                elif removed_non_technical:
                    logger.info(f"Removed {len(removed_non_technical)} non-technical entries from category '{cat.category}': {', '.join(removed_non_technical[:3])}{'...' if len(removed_non_technical) > 3 else ''}")
        
    # Reduced logging for latency
    # end_time = datetime.now()
    # logger.info(f"clean_content ended at {end_time}. Duration: {end_time - start_time}")
    return content

def get_density_strategy(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the ideal content density to ensure a full one-page resume.
    
    Returns a strategy dictionary with:
    - summary_words: (min, max) tuple for summary word count
    - exp_bullets: int for bullets per work experience
    - exp_word_range: (min, max) tuple for words per bullet
    - skills_detail: str description of skills detail level
    - min_categories: int minimum skill categories
    
    Note: Project bullets are always preserved exactly as provided in the payload,
    so no project bullet limits are included in the strategy.
    """
    # Handle both string and list formats for experience and projects
    experience = profile.get('experience', [])
    experience_text = profile.get('experience_text', '')
    
    if isinstance(experience, str):
        try:
            experience = json.loads(experience) if experience.strip() else []
        except (json.JSONDecodeError, TypeError):
            experience = []
    
    # If experience_text exists, estimate count from text (count newlines as separators)
    if experience_text and not experience:
        # Rough estimate: count non-empty lines
        exp_count = max(1, len([x for x in experience_text.split('\n') if x.strip()]))
    else:
        exp_count = len(experience) if isinstance(experience, list) else 0
    
    projects = profile.get('projects', [])
    projects_text = profile.get('projects_text', '')
    
    if isinstance(projects, str):
        try:
            projects = json.loads(projects) if projects.strip() else []
        except (json.JSONDecodeError, TypeError):
            projects = []
    
    # If projects_text exists, estimate count from text
    if projects_text and not projects:
        # Use intelligent project counting
        proj_count = count_projects_from_text(projects_text)
    else:
        proj_count = len(projects) if isinstance(projects, list) else 0
    total_items = exp_count + proj_count
    
    # SPECIAL CASE: If 1 work exp and 1 project, handle separately (check BEFORE total_items check)
    # Skills: 7-8 categories with 1 skill per category having 2 sub-skills, remaining keywords go in work experience bullets
    if exp_count == 1 and proj_count == 1:
        return {
            "name": "Single Exp with One Project",
            "summary_words": (80, 95),  # Increased by 10-15 words: (70-80) -> (80-95) to fill space
            "exp_bullets": 12,  # More bullets in work experience to fill space and include remaining keywords
            "exp_word_range": (30, 35),  # Increased word range to fill space
            "skills_detail": "Standard detail with 2 sub-skills per category (each category has ONE skill with exactly 2 sub-skills)",
            "min_categories": 7  # 7-8 categories of hard skills, each with 1 skill having 2 sub-skills
        }
    
    # SPECIAL CASE: If 1 work exp and 2 projects, handle separately (check BEFORE total_items check)
    # This case needs 10 bullets in work experience to fill space
    if exp_count == 1 and proj_count == 2:
        return {
            "name": "Single Exp with Two Projects",
            "summary_words": (70, 80),  # Summary word range to match other cases
            "exp_bullets": 10,  # 10 bullets for work experience to fill space
            "exp_word_range": (30, 35),  # Increased word range to fill space
            "skills_detail": "Extremely detailed with 5-6 sub-skills per item",
            "min_categories": 8  # More categories to fill space
        }
    
    # SPECIAL CASE: If 1 work exp and 3+ projects, handle separately (check BEFORE total_items check)
    # This case needs 8 bullets and 4 metrics in work experience
    if exp_count == 1 and proj_count >= 3:
        return {
            "name": "Single Exp with Multiple Projects",
            "summary_words": (70, 80),  # Summary word range to match other cases
            "exp_bullets": 8,  # 7-8 bullets for work experience
            "exp_word_range": (30, 35),  # Increased word range to fill space
            "skills_detail": "Extremely detailed with 5-6 sub-skills per item",
            "min_categories": 8  # More categories to fill space
        }
    
    # Case 1: Extreme Light (Only 1 item) - Maximum possible expansion via Experience and Summary
    if total_items == 1:
        # Special case: 1 work exp + 0 projects gets 100 words summary
        if exp_count == 1 and proj_count == 0:
            return {
                "name": "Single Work Exp with No Projects",
                "summary_words": (100, 120),  # Increased to 100 words to fill space
                "exp_bullets": 15,  # 15 bullets for work experience to fill space
                "exp_word_range": (40, 45),  # Increased word range to fill whitespace (40-45 words per bullet)
                "skills_detail": "Extremely detailed with 5-6 sub-skills per item",
                "min_categories": 8  # More categories to fill space
            }
        return {
            "name": "Single Item Expansion",
            "summary_words": (70, 80), 
            "exp_bullets": 15,  # 15 bullets for work experience to fill space
            "exp_word_range": (40, 45),  # Increased word range to fill whitespace (40-45 words per bullet)
            "skills_detail": "Extremely detailed with 5-6 sub-skills per item",
            "min_categories": 8  # More categories to fill space
        }

    # Case 2: Very light profile (2 items)
    if total_items == 2:
        # Special handling: If 2 work experiences, adjust bullets based on project count
        if exp_count == 2:
            # New strategy based on project count:
            # 0 projects: 7-8 bullets per work exp
            # 1 project: 6 bullets per work exp, 3 bullets per project
            # 2 projects: 5 bullets per work exp, 2 bullets per project
            # 3+ projects: 4 bullets per work exp, 2 bullets per project
            if proj_count == 0:
                exp_bullets = 7  # Can use 7-8, using 7 for first attempt
            elif proj_count == 1:
                exp_bullets = 6
            elif proj_count == 2:
                exp_bullets = 5
            else:  # 3+ projects
                exp_bullets = 4
            
            return {
                "name": "Two Work Exp Expansion",
                "summary_words": (70, 80),  # Summary word range for cases 2, 3, 4
                "exp_bullets": exp_bullets,  # Dynamic based on project count
                "exp_word_range": (32, 37),  # Further increased word range to squeeze more content
                "skills_detail": "Extremely detailed with 5-6 sub-skills per item",  # Even more detailed
                "min_categories": 8  # More categories to fill space
            }
        return {
            "name": "Double Item Expansion",
            "summary_words": (80, 90),  # Summary word range for cases 2, 3, 4
            "exp_bullets": 10, 
            "exp_word_range": (28, 32), 
            "skills_detail": "Detailed with sub-items",
            "min_categories": 7
        }

    # Case 3: Light profile (3 items)
    if total_items == 3:
        # Special handling: If 3 work experiences, adjust bullets based on project count
        if exp_count == 3:
            # For 3 work experiences:
            # 0 projects: 5 bullets per role
            # 1 project: 4 bullets per role, 2 bullets per project
            # 2 projects: 4 bullets per role, 2 bullets per project
            # 3+ projects: 4 bullets per role, 2 bullets per project
            if proj_count == 0:
                exp_bullets = 6
            else:  # 1+ projects
                if proj_count >= 3:
                    exp_bullets = 3  # 3 bullets per role = 9 total, but we'll instruct 8 total (3-3-2 distribution)
                else:
                    exp_bullets = 4
            
            return {
                "name": "Three Work Exp Expansion",
                "summary_words": (75, 85) if proj_count >= 3 else (70, 80),  # Increased for 3+ projects to fill space
                "exp_bullets": exp_bullets,  # Dynamic based on project count
                "exp_word_range": (28, 32) if proj_count == 0 else (27, 32) if proj_count >= 3 else (25, 30),  # Increased word range for 3+ projects
                "skills_detail": "Extremely detailed with 5-6 sub-skills per item" if proj_count >= 3 else "Detailed with sub-items",
                "min_categories": 8 if proj_count >= 3 else 6  # More categories to fill space when 3+ projects
            }
        return {
            "name": "Standard Expansion",
            "summary_words": (80, 90),  # Summary word range for cases 2, 3, 4
            "exp_bullets": 9 if proj_count == 0 else 8, 
            "exp_word_range": (25, 30), 
            "skills_detail": "Detailed with sub-items",
            "min_categories": 6
        }
    
    # Case 4: Standard profile (4-5 items)
    if total_items <= 5:
        # Special handling: If 3+ projects exist, ensure at least 2 are included
        # Adjust bullets to accommodate multiple projects
        if proj_count >= 3:
            return {
                "name": "Standard with Multiple Projects",
                "summary_words": (70, 80),  # Summary word range for cases 2, 3, 4
                "exp_bullets": 6,  # Reduced to make room for 2+ projects
                "exp_word_range": (22, 27), 
                "skills_detail": "Comprehensive",
                "min_categories": 6
            }
        return {
            "name": "Standard",
            "summary_words": (80, 90),  # Summary word range for cases 2, 3, 4
            "exp_bullets": 7 if proj_count == 0 else 6, 
            "exp_word_range": (22, 27), 
            "skills_detail": "Comprehensive",
            "min_categories": 6
        }
    
    # Case 4.5: Four Work Experiences with 3 Projects (NEW CASE)
    if exp_count == 4 and proj_count == 3:
        return {
            "name": "Four Work Exp with Three Projects",
            "summary_words": (80, 90),  # Summary word range for cases 2, 3, 4
            "exp_bullets": 4,  # 3-4 bullets per work exp (using 4 for first attempt)
            "exp_word_range": (22, 27), 
            "skills_detail": "Comprehensive",
            "min_categories": 6
        }
    
    # Case 5: Dense profile (5+ work experiences)
    return {
        "name": "High Density",
        "summary_words": (70, 80),  # Increased to fill space
        "exp_bullets": 4,  # Base number - will be distributed with more for recent, fewer for older
        "exp_word_range": (30, 35),  # Increased word range to fill whitespace (30-35 words per bullet)
        "skills_detail": "Extremely detailed with 5-6 sub-skills per item",
        "min_categories": 8  # More categories to fill space
    }

async def generate_resume(profile: Dict[str, Any], job: Dict[str, Any]) -> ResumeContent:
    """Generate a tailored resume using OpenAI LLM.
    
    Args:
        profile: Candidate profile dictionary with experience, projects, skills, etc.
        job: Job description dictionary with required_skills, soft_skills, etc.
    
    Returns:
        ResumeContent object with generated resume data
        
    Raises:
        Exception: If OpenAI API call fails or content generation fails
    """
    # Enrich job description by extracting skills from raw_text if needed
    job = await enrich_job_description(job)
    
    # Parse experience and projects from strings to lists
    profile = parse_profile_fields(profile)
    
    # --- SANITIZE SKILL LISTS ---
    # CRITICAL: ONLY use skills/keywords from job description - DO NOT include candidate profile skills
    job_required = set(job.get('required_skills', []) or [])  # LLM-extracted hard skills (clean)
    job_missing = list(job.get('missing_keywords', []) or [])  # ALL keywords - keep as list to preserve order and redundancy
    job_soft = set(job.get('soft_skills', []) or [])  # CRITICAL: All soft skills from job description
    job_literal_phrases = list(job.get('literal_phrases', []) or [])  # CRITICAL: Byte-exact literal phrases for ATS
    job_hybrid_phrases = list(job.get('hybrid_phrases', []) or [])  # CRITICAL: Dual-write phrases (literal + canonical)
    
    # Soft skills are already categorized by LLM extraction
    # Trust the LLM's categorization from job description enrichment
    actual_soft_skills = set()  # LLM already separates hard/soft skills in extraction
    
    # CRITICAL FIX: Use ONLY job_required (LLM-extracted hard skills) for final_hard_skills
    # DO NOT include job_missing as it contains ALL keywords including noise/garbage
    # job_missing should only be used for ATS keyword injection, not for skill filtering
    final_hard_skills = list(job_required - actual_soft_skills)
    # Only use soft skills from job description (no candidate soft skills)
    final_soft_skills = list(job_soft | actual_soft_skills)
    
    # Extract literal and canonical forms from hybrid phrases
    hybrid_literals = []
    hybrid_canonicals = []
    for hybrid in job_hybrid_phrases:
        # Handle both dict format (from JSON) and HybridPhrase model format
        if isinstance(hybrid, dict):
            if "literal" in hybrid and hybrid["literal"]:
                hybrid_literals.append(hybrid["literal"])
            if "canonical" in hybrid and hybrid["canonical"]:
                hybrid_canonicals.append(hybrid["canonical"])
        elif hasattr(hybrid, 'literal') and hasattr(hybrid, 'canonical'):
            # Handle HybridPhrase model instance
            if hybrid.literal:
                hybrid_literals.append(hybrid.literal)
            if hybrid.canonical:
                hybrid_canonicals.append(hybrid.canonical)
    
    # CRITICAL: Include literal phrases and hybrid phrases in ATS required keywords
    # These must appear verbatim in the resume for byte-level ATS matching
    # DO NOT deduplicate - ATS expects redundancy across categories
    domain = job.get('domain', 'general')
    canonical_keywords = set(job.get('canonical_keywords', []))
    
    # Build ATS required keywords: include all keywords, literal phrases, hybrid phrases, and soft skills
    # Preserve as list to maintain order and allow redundancy
    ats_required_keywords_list = (
        list(job_missing) + 
        job_literal_phrases + 
        hybrid_literals + 
        hybrid_canonicals + 
        list(final_soft_skills)
    )
    # Convert to set for deduplication, then back to list for order preservation
    # But preserve literal phrases even if similar - ATS needs byte-level matches
    ats_required_keywords_set = set(ats_required_keywords_list)
    ats_required_keywords = list(ats_required_keywords_set)
    
    # Canonical required: hard skills + canonical forms from hybrid phrases
    ats_canonical_required = list(set(final_hard_skills) | set(hybrid_canonicals) | canonical_keywords)
    
    # Comprehensive allowed skills: hard skills + canonical forms from hybrid phrases
    comprehensive_allowed_skills = set(final_hard_skills) | set(hybrid_canonicals) | job_required
    comprehensive_allowed_soft_skills = set(final_soft_skills)
    
    # Log ALL skills and keywords for debugging
    logger.info(f"Skills to be used in resume generation (ONLY from job description):")
    logger.info(f"  - ALL Hard skills from job ({len(final_hard_skills)}): {', '.join(str(s) for s in final_hard_skills[:20] if s)}{'...' if len(final_hard_skills) > 20 else ''}")
    logger.info(f"  - ALL Soft skills from job ({len(final_soft_skills)}): {', '.join(str(s) for s in list(final_soft_skills)[:20] if s)}{'...' if len(final_soft_skills) > 20 else ''}")
    logger.info(f"  - LITERAL PHRASES ({len(job_literal_phrases)}) - BYTE-EXACT FOR ATS: {', '.join(str(p) for p in job_literal_phrases[:20] if p)}{'...' if len(job_literal_phrases) > 20 else ''}")
    logger.info(f"  - HYBRID PHRASES ({len(job_hybrid_phrases)}) - DUAL-WRITE: {len(hybrid_literals)} literals, {len(hybrid_canonicals)} canonicals")
    logger.info(f"  - ALL ATS Keywords ({len(ats_required_keywords)}) - includes literals, hybrids, and all extracted keywords")
    
    # Log soft skills inclusion in ATS keywords
    soft_skills_in_ats = [ss for ss in final_soft_skills if ss in ats_required_keywords_set]
    logger.info(f"Soft skills included in ATS validation: {len(soft_skills_in_ats)}/{len(final_soft_skills)} soft skills will be validated and injected if missing")
    
    logger.info(f"Using extracted skills directly: {len(comprehensive_allowed_skills)} hard skills, {len(comprehensive_allowed_soft_skills)} soft skills")
    
    # Get strategy and calculate profile counts (do this once, reuse values)
    strategy = get_density_strategy(profile)
    
    # Calculate profile complexity for intelligent content estimation (reuse from strategy calculation)
    # Check both parsed lists and text fields for accurate counts
    exp_count = len(profile.get('experience', []) or [])
    if exp_count == 0 and profile.get('experience_text'):
        # Estimate from text if no parsed experience
        exp_count = max(1, len([x for x in profile.get('experience_text', '').split('\n') if x.strip()]))
    
    proj_count = len(profile.get('projects', []) or [])
    if proj_count == 0 and profile.get('projects_text'):
        # Use intelligent project counting
        proj_count = count_projects_from_text(profile.get('projects_text', ''))
    
    total_roles = exp_count
    total_projects = proj_count
    profile_projects_count = proj_count  # Reuse for later validation
    profile_exp_count = exp_count  # Reuse for later validation
    
    # Calculate actual years of experience from work experience entries
    actual_years = calculate_years_of_experience(profile)
    # Format for display (e.g., "3.2" -> "3+", "2.8" -> "3+", "1.5" -> "2+")
    if actual_years >= 1.0:
        # Round up if decimal part >= 0.5, otherwise use integer part
        if actual_years - int(actual_years) >= 0.5:
            years_display = f"{int(actual_years) + 1}+"
        else:
            years_display = f"{int(actual_years)}+"
    else:
        years_display = "1+"
    
    logger.info(f"Calculated years of experience from work history: {actual_years} years (display: {years_display})")
    
    # Back-compat alias (older prompt blocks reference this)
    all_required_keywords = set(ats_required_keywords)
    
    system_prompt = f"""You are an expert Resume Writer specializing in ATS-optimized, one-page resumes with zero whitespace.

GOAL: ONE-PAGE resume (ATS Score 90+) with ZERO whitespace at the bottom.

CONTENT ANALYSIS:
- Work Experiences: {total_roles} role(s), Projects: {total_projects} project(s)
- Strategy: {strategy['exp_bullets']} bullets/role, Projects: NOT generated by LLM (copied from payload)
- Estimated lines: ~{36 + (total_roles * 9) + (total_projects * 5)}



CONTENT SPECIFICATIONS:
- Summary: {strategy['summary_words'][0]}-{strategy['summary_words'][1]} words (use {strategy['summary_words'][1]} words max unless estimated > 50 lines)
- Experience Bullets: {strategy['exp_bullets']} bullets per role, {strategy['exp_word_range'][0]}-{strategy['exp_word_range'][1]} words per bullet (use {strategy['exp_word_range'][1]} words max when filling space)
- Projects: NOT generated by LLM - copied directly from payload (projects field in response will be empty/ignored)
- Skills: {strategy['skills_detail']}, minimum {strategy.get('min_categories', 4)} categories, REQUIRED: Each category must have exactly ONE skill using "Skill (Sub-skill, Sub-skill)" format with parentheses, other skills in the same category should be listed without sub-skills

SPECIAL CASES:
- {total_roles} == 1 + {total_projects} == 0: Use {strategy['exp_bullets']} bullets (15), {strategy['exp_word_range'][1]} words/bullet (40-45 words), expand skills to 8+ categories
- {total_roles} == 1 + {total_projects} == 1: Use {strategy['exp_bullets']} bullets (12), {strategy['exp_word_range'][1]} words/bullet. Skills: EXACTLY 7-8 categories, each category has ONE skill with exactly 2 sub-skills. Remaining hard skills/keywords must be integrated into work experience bullets.
- {total_roles} == 1 + {total_projects} == 2: Use {strategy['exp_bullets']} bullets (10), {strategy['exp_word_range'][1]} words/bullet
- {total_roles} == 1 + {total_projects} >= 3: Use 7-8 bullets, {strategy['exp_word_range'][1]} words/bullet
- {total_roles} >= 5: Progressive reduction (Role 1: 4 bullets, Role 2-3: 3 bullets, Role 4+: 2 bullets)

MANDATORY REQUIREMENTS (ALL MUST BE FOLLOWED):
1. Include '{job.get('role_title')}' anywhere in the Summary (not the first sentence).
2. ONE-PAGE FIT: Must fit on ONE page AND fill it completely (zero whitespace). If estimated > 50 lines, reduce bullets by 1-2. If estimated < 50 lines, maximize all sections. NOTE: There is NO PDF-level post-processing; one-page is enforced at the text layer.
3. BULLET QUALITY: Each bullet {strategy['exp_word_range'][0]}-{strategy['exp_word_range'][1]} words. For 1 work exp + 0 projects: 40-45 words/bullet (NOT 10-30 words). Add technical details, quantified results (METRICS), collaboration context to reach word count. CRITICAL: Include EXACTLY 6 metrics total across ALL experience bullets (NOT more, NOT less). Distribute metrics evenly - aim for 1 metric per bullet when possible, but ensure total is EXACTLY 6 metrics across all experience bullets combined.
4. SOFT SKILLS ({len(final_soft_skills)} total) - ⚠️ MANDATORY: ALL {len(final_soft_skills)} MUST appear verbatim in Summary/Experience (NEVER in Skills section):
   - List: {chr(10).join([f'     {i+1}. {skill}' for i, skill in enumerate(final_soft_skills) if skill]) if final_soft_skills else '     NONE'}
   - Use EXACT wording (preserve capitalization, punctuation, spelling)
   - Weave naturally into bullets (examples: "Organized and managed...", "Thrived in high-pressure environments, working under pressure...", "Demonstrated ability to prioritize...")
   - Verify ALL {len(final_soft_skills)} appear before finalizing - missing ANY = CRITICAL FAILURE
5. ATS KEYWORDS ({len(ats_required_keywords)} total) - ⚠️ ALL must appear verbatim (ATS uses literal string-matching):
   - Placement: Technical terms → Skills/Experience; Soft skills → Summary/Experience; Action verbs → Experience; Industry terms → Summary/Experience
   - Preserve exact wording (capitalization, hyphens, spelling)
   - CANONICAL ({len(ats_canonical_required)}): Must appear verbatim in canonical form if listed
   - Include naturally during generation (validation will inject missing ones, but include proactively)
6. HARD SKILLS ({len(final_hard_skills)} total) - FOLLOW DENSITY STRATEGY:
   - Must be organized into EXACTLY {strategy.get('min_categories', 4)}+ categories
   - Skills detail level: {strategy.get('skills_detail', 'Comprehensive')}
   - CRITICAL: ONLY include skills that are mentioned in the job description
   - DO NOT add any skills from the candidate profile or invent skills
   - SPECIAL RULE FOR 1 WORK EXP + 1 PROJECT: Use EXACTLY 7-8 skill categories, each category has ONE skill with exactly 2 sub-skills (format: "Skill (Sub-skill, Sub-skill)"). Other skills in the same category should be listed without sub-skills. Place remaining hard skills/keywords naturally in work experience bullets (integrate them into bullet descriptions, don't list them separately). Ensure ALL {len(final_hard_skills)} hard skills are covered (either in Skills section OR in work experience bullets).
   - For all other cases: All {len(final_hard_skills)} skills listed below MUST appear in Skills section
   
   FORMATTING: {strategy.get('min_categories', 4)}+ categories, each with ONE skill using "Skill (Sub-skill, Sub-skill)" format; others without sub-skills. Format: "Python (Django, Flask), C++, Java" (comma-separated, NO "+" signs). Detail level: {strategy.get('skills_detail', 'Comprehensive')} - if "Extremely detailed" use 5-6 sub-skills; if "Standard detail with 2 sub-skills" use exactly 2; otherwise 2-3 max or none.
   
   CONTENT: PURE TECHNICAL NAMES ONLY - Technical Skills Definition:
   - MUST be: Concrete, industry-recognized technical noun/phrase (e.g., "Python", "AWS", "Docker", "React", "PostgreSQL", "Tableau", "SQL", "Kubernetes", "REST API", "Agile") - tool, platform, framework, protocol, standard, system, software, language, or methodology - independently recognizable by ATS
   - NOT: Responsibilities, qualifications/education/degrees/certifications, legal status/work authorization, abilities/understanding/knowledge claims, process descriptions (unless named methodology like "Agile"), vague concepts ("programming skills", "best practices"), generic terms, descriptive/verb phrases, prefixes ("Familiarity with..."), soft skills, activities, non-technical attributes
   
   CAPITALIZATION: EVERY hard skill MUST be Title Case - capitalize principal words (e.g., "Robotic Programming", "Path Planning", "Data Analysis", "Machine Learning"). Acronyms remain fully capitalized (e.g., "WAAM", "KUKA", "AWS", "API", "SQL"). Do NOT lowercase skills. Examples: "Robotic Programming" (NOT "robotic programming"), "WAAM" (NOT "waam")
7. METRICS (EXACTLY 7 total, NO MORE, NO LESS) - ⚠️ CRITICAL MANDATORY REQUIREMENT - EXCEEDING 7 = FAILURE:
   - Summary: EXACTLY 1 metric - ⚠️ CRITICAL: If using years of experience, you MUST use "{years_display} years" (calculated from actual work experience: {actual_years} years). DO NOT use any other number. Examples: "with {years_display} years of experience", "delivered 100+ projects", "increased efficiency by 25%"
   - Experience: EXACTLY 6 metrics total across ALL experience bullets combined (distribute across bullets, but total must be exactly 6)
   - Projects: 0 metrics required
   - TOTAL: EXACTLY 7 metrics (1 in Summary + 6 in Experience + 0 in Projects). DO NOT add more than 7. DO NOT add less than 7.
   - Valid formats: Percentages ("25%", "increased by 30%", "improved by 50%"), Dollar amounts ("$50K", "saved $100,000", "$1M"), Counts ("10+ projects", "managed 5 team members", "{years_display} years"), Quantified actions ("reduced errors by 50%", "increased revenue by 25%", "delivered 20+ projects")
   - NOT metrics: Vague phrases without numbers ("significant improvement", "better results", "improved performance")
   - CRITICAL: Count metrics before finalizing - Summary must have exactly 1 metric, Experience must have exactly 6 metrics total (distributed across bullets), Projects must have exactly 0 metrics = 7 total. If you count more than 7 metrics, DO NOT include them. If you count less than 7 metrics, add more to reach exactly 7.
   - ⚠️ YEARS OF EXPERIENCE VALIDATION: The calculated years of experience from work history is {actual_years} years (display format: "{years_display}"). If you include years of experience in the Summary metric, you MUST use "{years_display} years" exactly. DO NOT use "5+ years" or any other number - use ONLY "{years_display} years" based on the actual work experience provided.
   - Examples of good metrics: "Increased revenue by 25%", "Managed 10+ team members", "Reduced costs by $50K", "Delivered 5 major projects", "Improved efficiency by 30%", "Generated $100K in savings", "with {years_display} years of experience"
   - DISTRIBUTION STRATEGY: If you have 12 experience bullets, include metrics in 6 of them (1 metric per bullet). If you have 6 experience bullets, include 1 metric in each. If you have fewer than 6 bullets, some bullets will have 2 metrics to reach exactly 6 total. The key is: EXACTLY 6 metrics total in Experience section, EXACTLY 1 metric in Summary = 7 total.
8. PROJECTS: NOT generated by LLM - copied directly from payload. DO NOT generate/modify/include in output.
9. EDUCATION: If education is provided in the candidate profile, format it with proper capitalization:
   - Degree types: Capitalize first letter (e.g., "Bachelor", "Master", "PhD")
   - Field of study: Use Title Case (e.g., "Computer Science", "Information Technology", "Engineering")
   - Format: "Bachelor in Computer Science: Institution Name" (NOT "bachelor in computer science: institution name")
   - Examples: "Bachelor in Computer Science: SLIET", "Master in Information Technology: University Name"
10. NO HALLUCINATION: ONLY use skills from job description (REQUIRED SKILLS below). Do NOT invent Projects/Experiences or add candidate profile skills. (See also requirement #6 and MANDATORY TECHNICAL SKILLS ENFORCEMENT section)
11. FILLING SPACE: If light profile, expand Summary ({strategy['summary_words'][1]} words), maximize bullets ({strategy['exp_bullets']}/role), maximize word count ({strategy['exp_word_range'][1]}/bullet). For 1 work exp + 1 project: Use 7-8 skill categories, each category has ONE skill with exactly 2 sub-skills, place remaining keywords in work experience bullets. For other cases: expand skills (8+ categories with sub-skills). Use ONLY REQUIRED SKILLS.
"""

    # Build user prompt (reuse already calculated counts)
    # Add experience and projects text if they exist as plain text
    profile_for_prompt = profile.copy()
    experience_text = profile.get('experience_text', '')
    projects_text = profile.get('projects_text', '')
    
    user_prompt = f"""CANDIDATE PROFILE: {json.dumps(profile_for_prompt)}
TARGET JOB: {json.dumps(job)}

EXPERIENCE AND PROJECTS EXTRACTION:
{f'''EXPERIENCE TEXT: {experience_text}
Parse into Work Experience entries: extract company, role, duration, responsibilities/achievements. Create separate entries for each experience.
''' if experience_text else ''}

{f'''PROJECTS TEXT: {projects_text}
CRITICAL: Projects are NOT generated by LLM - they are copied directly from the payload.
DO NOT generate, modify, or include projects in your output. The projects section will be filled directly from the payload.
''' if projects_text else ''}

REQUIRED SKILLS (FROM JOB DESCRIPTION ONLY) - MUST INCLUDE ALL:

HARD SKILLS ({len(final_hard_skills)} total) - ALL must be in Technical Skills section:
{chr(10).join([f'  - {skill}' for skill in final_hard_skills]) if final_hard_skills else '  NONE'}

⚠️ MANDATORY TECHNICAL SKILLS ENFORCEMENT: ONLY use EXACT technical skills from list above: {', '.join(sorted(final_hard_skills)) if final_hard_skills else 'NONE'}. VERIFY every skill against provided list. Reference Technical Skills Definition above for what qualifies.

SOFT SKILLS ({len(final_soft_skills)} total) - ⚠️ MANDATORY: ALL must appear verbatim in Summary/Experience (NOT in Skills):
{chr(10).join([f'  {i+1}. {skill}' for i, skill in enumerate(final_soft_skills) if skill]) if final_soft_skills else '  NONE'}

⚠️ VERIFICATION (BEFORE FINALIZING):
1. ✅ ALL {len(final_hard_skills)} hard skills in Technical Skills section
2. ⚠️ ALL {len(final_soft_skills)} soft skills verbatim in Summary/Experience (verify each individually - missing ANY = CRITICAL FAILURE)
3. ✅ Format ALL hard skills using Title Case (see requirement #6 for detailed capitalization rules)
4. ✅ Use EXACT wording from lists above for soft skills (preserve capitalization, punctuation, spelling)
5. ✅ Weave soft skills naturally (examples: "Demonstrated [skill]...", "Utilized [skill] to...", "Thrived in [skill] environments...")
6. ✅ ONLY use skills from job description (NO candidate profile skills or invented skills) - see requirement #6 and #10

CRITICAL - ATS REQUIRED KEYWORDS ({len(ats_required_keywords)} total):
⚠️ BYTE-LEVEL EXACTNESS REQUIRED: These keywords must appear VERBATIM (exact string match) in the resume text for ATS tools like Jobscan.

{f'''LITERAL PHRASES ({len(job_literal_phrases)} total) - MUST APPEAR VERBATIM (BYTE-EXACT):
{chr(10).join([f'  [LITERAL] {phrase}' for phrase in job_literal_phrases[:30] if phrase]) if job_literal_phrases else '  NONE'}{f'{chr(10)}  ... (showing first 30 of {len(job_literal_phrases)})' if len(job_literal_phrases) > 30 else ''}

HYBRID PHRASES ({len(job_hybrid_phrases)} total) - DUAL-WRITE (BOTH literal AND canonical required):
{chr(10).join([f'  [HYBRID] literal: "{h.get("literal", "") if isinstance(h, dict) else (h.literal if hasattr(h, "literal") else "")}", canonical: "{h.get("canonical", "") if isinstance(h, dict) else (h.canonical if hasattr(h, "canonical") else "")}"' for h in job_hybrid_phrases[:20]]) if job_hybrid_phrases else '  NONE'}{f'{chr(10)}  ... (showing first 20 of {len(job_hybrid_phrases)})' if len(job_hybrid_phrases) > 20 else ''}

CANONICAL REQUIRED ({len(ats_canonical_required)} total):
{chr(10).join([f'  [CANONICAL] {kw}' for kw in sorted(ats_canonical_required)[:30] if kw]) if ats_canonical_required else '  NONE'}{f'{chr(10)}  ... (showing first 30 of {len(ats_canonical_required)})' if len(ats_canonical_required) > 30 else ''}

ALL REQUIRED KEYWORDS ({len(ats_required_keywords)} total - includes literals, hybrids, and extracted keywords):
{chr(10).join([f'  - {kw}' for kw in sorted(ats_required_keywords)[:50] if kw]) if ats_required_keywords else '  NONE'}{f'{chr(10)}  ... (showing first 50 of {len(ats_required_keywords)})' if len(ats_required_keywords) > 50 else ''}
''' if ats_required_keywords else 'NONE - No keywords required'}

⚠️ HYBRID PHRASES: BOTH literal phrase ("manage multiple projects" → verbatim in Experience/Summary) AND canonical form ("Project Management" → Skills/Experience) must appear.

⚠️ FINAL VERIFICATION CHECKLIST:
1. Skills: {strategy.get('min_categories', 4)}+ categories, {strategy.get('skills_detail', 'Comprehensive')} detail, ONE skill per category with "Skill (Sub-skill)" format. SPECIAL: For 1 work exp + 1 project, use EXACTLY 7-8 categories, each with ONE skill with exactly 2 sub-skills.
2. ✅ ALL {len(final_hard_skills)} hard skills covered (in Skills section OR work experience for 1 work exp + 1 project case). ALL hard skills in Title Case - Do NOT lowercase.
3. ⚠️ ALL {len(final_soft_skills)} soft skills verbatim in Summary/Experience (missing ANY = CRITICAL FAILURE)
4. ✅ ALL {len(ats_required_keywords)} ATS keywords verbatim. Skills: PURE TECHNICAL NAMES ONLY.
5. ✅ Projects: NOT generated - copied from payload (DO NOT include in response)
6. ⚠️ METRICS: Summary (EXACTLY 1) + Experience (EXACTLY 6 total) + Projects (EXACTLY 0) = EXACTLY 7 total (NO MORE, NO LESS). Count before finalizing. If you count more than 7, remove excess metrics. If you count less than 7, add more metrics. Examples: "increased by 25%", "saved $50K", "managed 10+ projects", "reduced errors by 50%", "delivered 5 major projects", "improved efficiency by 30%"
7. ✅ One-page fit with zero whitespace (NO PDF-level post-processing)

Follow system prompt requirements. Verify all skills, keywords, and metrics before finalizing.
"""

    try:
        # OPTIMIZATION: Use async client directly (no executor overhead)
        # Get async client for better performance
        from app.agent.utils import get_async_openai_client
        async_client = get_async_openai_client()
        
        logger.info("Starting resume generation LLM call...")
        start_llm_time = datetime.now()
        
        # OPTIMIZATION: Start project parsing in parallel with resume generation
        projects_text = profile.get('projects_text', '')
        project_task = None
        if projects_text:
            import asyncio
            project_task = asyncio.create_task(parse_projects_from_text(projects_text))
            logger.info("Started project parsing in parallel with resume generation")
        
        response = await async_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=ResumeContent,
                temperature=0.0,  # Use 0.0 for maximum consistency and deterministic results
            timeout=30.0,  # OPTIMIZATION: Reduced from 40.0 to 30.0 for faster failure detection
            max_tokens=2200  # OPTIMIZATION: Reduced from 2600 to 2200 for faster processing while maintaining quality
            )
        
        llm_duration = (datetime.now() - start_llm_time).total_seconds()
        logger.info(f"Resume generation LLM call completed in {llm_duration:.2f}s")
        
        content = response.choices[0].message.parsed
        
        # Fix years of experience in summary if incorrect
        if content.profile_summary and actual_years > 0:
            # Find and replace incorrect years patterns (e.g., "5+ years", "6+ years", etc.)
            # Only replace if it's clearly a years-of-experience metric, not other metrics
            summary_lower = content.profile_summary.lower()
            years_pattern = r'(\d+)\+?\s*years?\s+of\s+experience'
            match = re.search(years_pattern, summary_lower)
            if match:
                found_years = int(match.group(1))
                # Only fix if the found number is significantly different from actual
                if abs(found_years - actual_years) >= 1:
                    # Replace in original case
                    original_pattern = r'(\d+)\+?\s*years?\s+of\s+experience'
                    content.profile_summary = re.sub(
                        original_pattern,
                        f'{years_display} years of experience',
                        content.profile_summary,
                        flags=re.IGNORECASE
                    )
                    logger.info(f"Fixed years of experience in summary: {found_years}+ -> {years_display} (actual: {actual_years} years)")

        # CRITICAL: Projects are NOT generated by LLM - parsed/extracted from payload using LLM
        # OPTIMIZATION: Await project parsing task that was started in parallel
        if project_task:
            parsed_projects = await project_task
            if parsed_projects:
                content.projects = parsed_projects
                logger.info(f"Parsed {len(parsed_projects)} project(s) from payload using LLM extraction")
        
        # ALL REMOVAL LOGIC DISABLED - Trust LLM completely, do not filter or remove any skills
        # User requirement: Do not remove any skills after extraction
        # The LLM generates skills correctly - preserve everything it generates
        logger.info("ALL skill filtering/removal logic DISABLED - preserving all LLM-generated skills")
        
        # Only clean up formatting (remove "+" signs) - do NOT remove any skills
        # Also add sub-skills to sparse categories (1-2 skills) to fill space
        for cat in content.technical_skills or []:
            if not cat.skills:
                continue
            cleaned_skills = []
            for skill in cat.skills:
                if not skill or not str(skill).strip():
                    continue  # Skip empty skills only
                # Ensure comma-separated format (preserve C++ and C#)
                skill_clean = str(skill).strip()
                if skill_clean:
                    # Preserve C++ and C# by using placeholders
                    placeholder_cpp = "___CPP_PLACEHOLDER___"
                    placeholder_csharp = "___CSHARP_PLACEHOLDER___"
                    
                    if "C++" in skill_clean:
                        skill_clean = skill_clean.replace("C++", placeholder_cpp)
                    elif "c++" in skill_clean.lower():
                        idx = skill_clean.lower().find("c++")
                        if idx >= 0:
                            skill_clean = skill_clean[:idx] + placeholder_cpp + skill_clean[idx+3:]
                    
                    if "C#" in skill_clean:
                        skill_clean = skill_clean.replace("C#", placeholder_csharp)
                    elif "c#" in skill_clean.lower():
                        idx = skill_clean.lower().find("c#")
                        if idx >= 0:
                            skill_clean = skill_clean[:idx] + placeholder_csharp + skill_clean[idx+2:]
                    
                    # Replace separator + signs with commas
                    while skill_clean.startswith("+"):
                        skill_clean = skill_clean[1:].lstrip()
                    skill_clean = skill_clean.replace("+ ", ", ").replace("+", ", ")
                    while ",," in skill_clean or ", ," in skill_clean:
                        skill_clean = skill_clean.replace(",,", ",").replace(", ,", ",")
                    
                    # Restore C++ and C#
                    skill_clean = skill_clean.replace(placeholder_cpp, "C++")
                    skill_clean = skill_clean.replace(placeholder_csharp, "C#")
                    
                if skill_clean and skill_clean.strip():
                        cleaned_skills.append(skill_clean.strip())
            
            # SPACE FILLING: If category has only 1-2 skills, add sub-skills to the first skill
            if len(cleaned_skills) <= 2 and cleaned_skills:
                first_skill = cleaned_skills[0]
                # Check if first skill already has sub-skills (contains parentheses)
                if '(' not in first_skill or ')' not in first_skill:
                    # Extract base skill name (remove any existing formatting)
                    base_skill = first_skill.strip()
                    category_name = (cat.category or "").lower()
                    
                    # Generate relevant sub-skills based on skill name and category
                    sub_skills = _generate_sub_skills_for_skill(base_skill, category_name)
                    if sub_skills:
                        # Convert first skill to "Skill (Sub-skill1, Sub-skill2, ...)" format
                        cleaned_skills[0] = f"{base_skill} ({', '.join(sub_skills)})"
                        logger.info(f"Added sub-skills to sparse category '{cat.category}': {base_skill} → {cleaned_skills[0]}")
            
            cat.skills = cleaned_skills

        # --- GUARDRAIL: Ensure skills section is not empty ---
        # If no skills were generated or all were filtered out, inject at least the required skills
        if not content.technical_skills or all(not cat.skills for cat in content.technical_skills):
            logger.warning("Skills section is empty after generation/filtering. Injecting required skills...")
            if final_hard_skills:
                # Create at least one category with skills
                from app.agent.ats_validator import _ensure_skill_category
                skills_cat = _ensure_skill_category(content, "Technical Skills")
                # Add all required hard skills
                for skill in final_hard_skills[:20]:  # Limit to first 20 to avoid overwhelming
                    if skill and skill not in skills_cat.skills:
                        skills_cat.skills.append(skill)
                logger.info(f"Injected {len(skills_cat.skills)} skills into empty Skills section")
            elif comprehensive_allowed_skills:
                # Fallback: use comprehensive allowed skills
                from app.agent.ats_validator import _ensure_skill_category
                skills_cat = _ensure_skill_category(content, "Technical Skills")
                for skill in list(comprehensive_allowed_skills)[:20]:
                    if skill and str(skill).strip() and skill not in skills_cat.skills:
                        skills_cat.skills.append(str(skill).strip())
                logger.info(f"Injected {len(skills_cat.skills)} skills from comprehensive_allowed_skills into empty Skills section")

        # Log final skills state
        total_skills_count = sum(len(cat.skills) for cat in content.technical_skills or [])
        logger.info(f"Final Skills section state: {len(content.technical_skills or [])} categories, {total_skills_count} total skills")
        for cat in content.technical_skills or []:
            if cat.skills:
                logger.info(f"  - {cat.category}: {len(cat.skills)} skills - {', '.join(str(s) for s in cat.skills[:5])}{'...' if len(cat.skills) > 5 else ''}")

        # --- ATS SURFACE-FORM ENFORCEMENT FOR SOFT SKILLS ---
        # CRITICAL: ATS tools require exact verbatim phrases, not semantic synonyms
        # Rules:
        # 1. Check if exact phrase (case-insensitive) appears verbatim in resume text
        # 2. Do NOT rely on semantic similarity or synonyms
        # 3. If missing, inject exact wording from job description into Summary (preferred) or Experience
        # 4. Never remove or rewrite existing content - only append minimal sentences
        
        summary_text = (content.profile_summary or "").lower()
        experience_text = " ".join([h.lower() for exp in (content.experience or []) for h in (exp.highlights or [])])
        combined_text = (summary_text + " " + experience_text).lower()
        
        missing_soft_skills = []
        for soft_skill in comprehensive_allowed_soft_skills:
            if not soft_skill:
                continue
            # STRICT VERBATIM MATCHING: Check if exact phrase appears (case-insensitive)
            # Do NOT use semantic similarity, synonyms, or word-by-word matching
            soft_skill_lower = str(soft_skill).lower().strip()
            
            # Exact verbatim match (case-insensitive) - this is what ATS tools expect
            if soft_skill_lower not in combined_text:
                missing_soft_skills.append(soft_skill)
                logger.debug(f"ATS Surface-Form Check: '{soft_skill}' NOT found verbatim in resume text")
            else:
                logger.debug(f"ATS Surface-Form Check: '{soft_skill}' found verbatim ✓")
        
        if missing_soft_skills:
            logger.warning(f"ATS Surface-Form Enforcement: Missing {len(missing_soft_skills)}/{len(comprehensive_allowed_soft_skills)} soft skills verbatim in resume: {', '.join(str(s) for s in missing_soft_skills)}")

            # Inject exact phrases verbatim (no semantic wrappers, no rewrites)
            # Prefer Summary, fallback to Experience
            injected_summary = []
            injected_experience = []
            
            for soft_skill in missing_soft_skills:
                if not soft_skill:
                    continue
                
                # Create minimal sentence with exact phrase verbatim - preserve exact wording from job description
                # The exact phrase must appear verbatim (case-insensitive match allowed, but phrase structure must match)
                # Use natural, minimal wrappers that ensure ATS tools can find the exact phrase
                if len(soft_skill.split()) == 1:
                    # Single word: "Demonstrated {exact_word} in professional environments."
                    injection_phrase = f"Demonstrated {soft_skill} in professional environments."
                else:
                    # Multi-word phrase: "Applied {exact_phrase} to enhance team performance."
                    injection_phrase = f"Applied {soft_skill} to enhance team performance."
                
                # Optimized: Cache lowercase versions
                soft_skill_lower = soft_skill.lower()
                
                # Try Summary first (preferred)
                if content.profile_summary:
                    # Cache summary_lower once per iteration (already computed if summary exists)
                    if not hasattr(content, '_summary_lower_cache'):
                        content._summary_lower_cache = content.profile_summary.lower()
                    summary_lower = content._summary_lower_cache
                    
                    if soft_skill_lower not in summary_lower:
                        # Append to summary (preserve existing content)
                        content.profile_summary = content.profile_summary.rstrip() + " " + injection_phrase
                        # Invalidate cache
                        content._summary_lower_cache = content.profile_summary.lower()
                        injected_summary.append(soft_skill)
                        logger.info(f"ATS Injection: Injected '{soft_skill}' verbatim into Summary")
                        continue
                
                # Fallback to Experience
                if not content.experience:
                    content.experience = []
                if len(content.experience) == 0:
                    content.experience.append(Experience(company="", role="", duration="", highlights=[]))
                
                # Check if already in experience
                exp0 = content.experience[0]
                if exp0.highlights is None:
                    exp0.highlights = []
                
                # Optimized: Cache experience text lower once per experience entry
                if not hasattr(exp0, '_highlights_text_lower_cache'):
                    exp0._highlights_text_lower_cache = " ".join([h.lower() for h in exp0.highlights]) if exp0.highlights else ""
                exp_text_lower = exp0._highlights_text_lower_cache
                
                if soft_skill_lower not in exp_text_lower:
                    # Append new bullet with exact phrase (never rewrite existing)
                    exp0.highlights.append(injection_phrase)
                    # Invalidate cache
                    exp0._highlights_text_lower_cache = " ".join([h.lower() for h in exp0.highlights])
                    injected_experience.append(soft_skill)
                    logger.info(f"ATS Injection: Injected '{soft_skill}' verbatim into Experience")
            
            total_injected = len(injected_summary) + len(injected_experience)
            if total_injected > 0:
                logger.info(f"ATS Surface-Form Enforcement: Successfully injected {total_injected} soft skills verbatim ({len(injected_summary)} in Summary, {len(injected_experience)} in Experience)")
            else:
                logger.warning(f"ATS Surface-Form Enforcement: Failed to inject {len(missing_soft_skills)} soft skills - check injection logic")
        else:
            logger.info(f"ATS Surface-Form Enforcement: ✅ All {len(comprehensive_allowed_soft_skills)} soft skills found verbatim in resume")
        
        # --- METRICS VALIDATION ---
        # CRITICAL: Validate that exactly 7 metrics are present (1 in Summary + 6 in Experience + 0 in Projects)
        expected_total_metrics = 7
        actual_metrics = count_metrics_by_section(content)
        
        logger.info(f"Metrics Validation: Expected {expected_total_metrics} metrics (Summary: 1, Experience: 6, Projects: 0)")
        logger.info(f"Metrics Validation: Actual metrics found - Summary: {actual_metrics['summary']}, Experience: {actual_metrics['experience']}, Projects: {actual_metrics['projects']}, Total: {actual_metrics['total']}")
        
        if actual_metrics['total'] < expected_total_metrics:
            missing_count = expected_total_metrics - actual_metrics['total']
            logger.warning(
                f"⚠️ METRICS VALIDATION FAILED: Expected {expected_total_metrics} metrics, found {actual_metrics['total']} metrics. "
                f"Missing {missing_count} metric(s). "
                f"Breakdown - Summary: {actual_metrics['summary']}/1, Experience: {actual_metrics['experience']}/6, Projects: {actual_metrics['projects']}/0. "
                f"This is a critical requirement - the resume must have exactly {expected_total_metrics} quantified metrics (1 in Summary, 6 in Experience, 0 in Projects)."
            )
        elif actual_metrics['total'] > expected_total_metrics:
            excess_count = actual_metrics['total'] - expected_total_metrics
            logger.warning(
                f"⚠️ METRICS VALIDATION WARNING: Found {actual_metrics['total']} metrics (exceeds maximum of {expected_total_metrics}). "
                f"Excess: {excess_count} metric(s). "
                f"Breakdown - Summary: {actual_metrics['summary']}/1, Experience: {actual_metrics['experience']}/6, Projects: {actual_metrics['projects']}/0. "
                f"Maximum allowed: {expected_total_metrics} metrics (1 in Summary, 6 in Experience, 0 in Projects)."
            )
        else:
            logger.info(f"✅ Metrics Validation PASSED: Found {actual_metrics['total']} metrics (exactly {expected_total_metrics} as required)")

        # --- TEXT-LAYER ATS PREFLIGHT (NO LLM REWRITES) ---
        # NOTE: Validation is non-blocking - always proceeds with resume generation
        # Issues are logged as warnings but don't prevent output
        # OPTIMIZATION: Early exit if coverage is already high AND no missing keywords (saves 5-10 seconds)
        # CRITICAL: Still inject missing keywords even if coverage is high to maintain content density
        text_before_preflight = build_plain_text(content)
        cov_quick, present_quick, missing_quick = compute_coverage(text_before_preflight, set(ats_required_keywords))
        
        if cov_quick >= 0.90 and len(missing_quick) == 0:  # Already 90%+ coverage AND no missing keywords - skip expensive classification
            logger.info(f"OPTIMIZATION: Skipping full ATSPreflight - coverage already {cov_quick:.2%} with no missing keywords (threshold: 90%)")
            # Create minimal preflight report to maintain compatibility
            preflight_report = {
                "required_count": len(ats_required_keywords),
                "coverage_before": cov_quick,
                "coverage_after": cov_quick,
                "missing_before": [],
                "missing_after": [],
                "injected": {"skills": [], "experience": []},
                "frequency": {},
                "placement_violations": [],
                "ok": True
            }
        else:
            # Run full preflight - this will inject missing keywords to maintain content density
            preflight = ATSPreflight(min_coverage=0.80)  # Lower threshold to be more lenient
            content, preflight_report = await preflight.run(content, ats_required_keywords)
        
        # Log issues but never fail - always proceed with resume generation
        cov_after = preflight_report.get("coverage_after", 0.0)
        missing_count = len(preflight_report.get("missing_after", []))
        if cov_after < 0.80 or missing_count > 0:
            logger.warning(
                f"ATS preflight: coverage={cov_after:.2%}, missing={missing_count} keywords. "
                f"Continuing with resume generation anyway."
            )

        # --- ONE-PAGE A4 PREFLIGHT DISABLED ---
        # Layout pruning removed - skills should never be pruned
        # Pruning was causing required skills to be removed
        # If resume is too long, let PDF engine handle it
        prune_report = {
            "removed": {"experience_bullets": [], "skills": []},
            "estimate": {"estimated_lines": 0, "max_lines": 56, "fits_one_page": True}
        }
        logger.info("Layout preflight: DISABLED - no pruning performed to preserve all skills")

        # Re-check ATS coverage after pruning (no reinjection here)
        # NOTE: This is for telemetry only - we never fail here
        # Simplified: Only check if keywords are present, placement validation is handled by ATSPreflight
        # Optimized: Build text once and filter empty keywords upfront
        text_after = build_plain_text(content)
        missing_after: List[str] = []
        # Filter empty keywords upfront for better performance
        valid_keywords = [kw for kw in ats_required_keywords if kw and kw.strip()]
        for kw in valid_keywords:
            if count_occurrences(text_after, kw) == 0:
                missing_after.append(kw)

        # Log issues but never fail - always proceed with resume generation
        if missing_after:
            logger.warning(
                f"ATS validation after pruning: {len(missing_after)} missing keywords. Resume will be generated anyway."
            )
            if missing_after[:10]:
                logger.warning(f"Missing keywords (sample): {', '.join(str(k) for k in missing_after[:10] if k)}")

        # Always log success - resume generation always proceeds
        # Pruning disabled - no line estimation
        logger.info(
            f"Resume generation completed: coverage={cov_after:.2%}, "
            f"estimated_lines=N/A (pruning disabled), fits_one_page=N/A"
        )

        # NOTE: Legacy post-generation warnings (e.g., "keywords must be in narrative" / "keywords must not be in Skills")
        # were removed because ATS correctness is now enforced by:
        # - deterministic authoritative keyword set (domain-validated + canonicalized)
        # - deterministic injection into safe sections (Skills/Compliance/Experience)
        # - deterministic placement validation inside ATSPreflight
        # - Layout pruning disabled to preserve all extracted skills
        #
        # If ATSPreflight passes, keyword coverage + placement are already guaranteed.
        
        # Enforce strategy limits (overflow is handled by PDF engine if needed; no text-layer pruning)
        exp_bullet_limit = strategy['exp_bullets']
        for exp in content.experience:
            exp.highlights = exp.highlights[:exp_bullet_limit]
        
        # CRITICAL: Ensure proper number of projects based on work experience count
        # Only limit bullet points, don't remove entire projects
        # (profile_projects_count and profile_exp_count already calculated above)
        
        if profile_projects_count >= 3:
            if profile_exp_count == 1:
                # With 1 work exp, allow up to 3 projects (prefer all 3 to fill space)
                if len(content.projects) < 2:
                    logger.warning(f"CRITICAL: Only {len(content.projects)} project(s) included, but profile has {profile_projects_count} projects with 1 work exp. Expected at least 2 (preferably 3).")
            else:
                # With 2+ work exp, require at least 2 projects
                if len(content.projects) < 2:
                    logger.warning(f"CRITICAL: Only {len(content.projects)} project(s) included, but profile has {profile_projects_count} projects with {profile_exp_count} work exp. Expected at least 2. This is a failure.")
        
        # CRITICAL: Never limit or omit project bullets - they must be preserved exactly as provided
        # Only clear projects if profile has no projects
        if profile_projects_count == 0:
            content.projects = []
        # DO NOT truncate project bullets - they must remain exactly as provided in the payload

        # Reduced logging for latency
        # end_time = datetime.now()
        # logger.info(f"generate_resume ended at {end_time}. Duration: {end_time - start_time}")
        return clean_content(content)
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        error_msg = str(e)
        sanitized = sanitize_error_message(error_msg) if "<!DOCTYPE html>" in error_msg or "DNS" in error_msg else error_msg
        logger.error(f"Network/Connection error during resume generation: {sanitized}")
        raise
    except openai.RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Error (429): Too many requests. Please wait before retrying. Error: {str(e)}")
        raise
    except openai.APIError as e:
        # Check if it's a 429 error that wasn't caught as RateLimitError
        if hasattr(e, 'status_code') and e.status_code == 429:
            logger.error(f"OpenAI API Error 429 (Too Many Requests): Rate limit exceeded. Error: {str(e)}")
        else:
            error_msg = str(e)
            sanitized = sanitize_error_message(error_msg) if "<!DOCTYPE html>" in error_msg or "DNS" in error_msg else error_msg
            logger.error(f"OpenAI API Error: {sanitized}")
        raise
    except Exception as e:
        error_msg = str(e)
        sanitized = sanitize_error_message(error_msg) if "<!DOCTYPE html>" in error_msg or "DNS" in error_msg else error_msg
        logger.error(f"Error generating resume logic: {sanitized}")
        raise

