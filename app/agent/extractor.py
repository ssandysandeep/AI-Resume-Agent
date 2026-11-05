"""
Unified extraction module for skills and keywords from job descriptions.
Uses LLM for all extraction and filtering - no regex, no hardcoded lists.
"""
import asyncio
from typing import Dict, Any, List, Set, Tuple
from pydantic import BaseModel, Field
from app.agent.utils import logger, get_async_openai_client

client = get_async_openai_client()


class HybridPhrase(BaseModel):
    """A hybrid phrase with both literal (exact JD phrase) and canonical (noun-based form) for dual-write."""
    literal: str = Field(description="Exact verbatim phrase from job description (byte-level exact, no normalization)")
    canonical: str = Field(description="Noun-based canonical form for semantic matching")


class DetectedDomains(BaseModel):
    """Domains detected from job description by LLM."""
    domains: List[str] = Field(
        description="ALL applicable job domains/types inferred from the job description content. Use your knowledge to identify domains from ANY industry, role, or field - not limited to examples. Analyze the job description content, responsibilities, required skills, and context to intelligently detect domains. Multi-domain is allowed. Examples (but NOT limited to): project management, data analytics, software engineering, healthcare administration, finance, marketing, operations, manufacturing, education, legal, consulting, research, sales, customer service, engineering (any discipline), design, creative fields, technical fields, business fields, etc. CRITICAL: Infer domains intelligently from job content - do not rely only on explicit mentions. Never collapse to a single domain - if multiple domains apply, list them all. If the domain is unclear, return ['general'].",
        default_factory=list
    )


class ExtractedJobSkills(BaseModel):
    """Skills and keywords extracted from job description by LLM with ATS-grade accuracy."""
    domains: List[str] = Field(
        description="ALL applicable domains intelligently inferred from the job description content. Use your knowledge to identify domains from ANY industry, role, or field. Analyze responsibilities, required skills, and context to detect domains - not limited to explicit mentions. Multi-domain is allowed. Examples (but NOT limited to): project management, data analytics, software engineering, healthcare, finance, marketing, operations, manufacturing, education, legal, consulting, research, sales, engineering disciplines, design, creative fields, etc. Never collapse to a single domain.",
        default_factory=list
    )
    hard_skills: List[str] = Field(
        description="ALL technical capabilities, operational skills, domain-specific skills, tools, technologies, methodologies, and competencies mentioned or implied ANYWHERE in the job description (including role summary, responsibilities, requirements, preferred requirements, company description, mission, vision, benefits, branding sections). EXHAUSTIVE EXTRACTION: Scan the ENTIRE job description - if a skill appears in ANY section, it MUST be extracted. Do NOT filter based on section location or perceived importance. CRITICAL EXTRACTION RULE - VERBATIM ONLY: Do NOT rephrase, paraphrase, infer, or semantically substitute any skill. Use ONLY the exact wording found in the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Do NOT replace words with synonyms. Do NOT group skills or hide them in parentheses. Do NOT expand or compress phrases. UNIQUE SURFACE FORMS: Each unique surface form must be listed separately. If JD contains 'AI', 'AI/ML', and 'artificial intelligence', extract all three separately. If unsure whether a phrase is a skill, INCLUDE it. If a skill appears in the JD, it must appear verbatim in this list. Exact text matching > readability > style. CRITICAL CLASSIFICATION RULE: A skill is a HARD SKILL if it: (1) involves tools, technologies, software, platforms, systems, equipment, or machinery; (2) requires technical knowledge, domain expertise, or specialized training; (3) produces a deliverable, analysis, or technical outcome; (4) is industry-specific, domain-specific, or role-specific expertise; (5) involves data, analytics, or technical processes; (6) requires certifications, licenses, or standards knowledge. MUST EXTRACT AS HARD SKILLS (not soft skills): Advanced analytics, artificial intelligence, machine learning, AI/ML, business administration, Financial Planning, Technology Innovation, Security Solutions, logistics, S3 (AWS S3), data requirements, cloud computing, cybersecurity, programming languages, databases, frameworks, methodologies, strategic planning, operations management, financial analysis, supply chain management, warehouse operations, and ANY domain-specific technical or operational capability. Physical skills like 'manual dexterity' are HARD SKILLS if required for technical/operational tasks (operating machinery, equipment, tools). Extract comprehensively from ANY domain, industry, or role, and from ANY section of the JD, but ALWAYS use exact verbatim wording from the job description. CRITICAL: If the job mentions responsibilities or requirements that imply specific skills, extract the exact phrases used in the JD - do NOT create new phrases or canonical forms. Extract skills exactly as written: preserve capitalization (Advanced analytics, not advanced analytics), spacing (AI/ML, not AI ML), hyphens, and punctuation.",
        default_factory=list
    )
    soft_skills: List[str] = Field(
        description="ALL interpersonal skills, behavioral traits, leadership qualities, collaboration traits, communication abilities, and personal attributes mentioned in the job description. CRITICAL EXTRACTION RULE - VERBATIM ONLY: Do NOT rephrase, paraphrase, infer, or semantically substitute any skill. Use ONLY the exact wording found in the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Do NOT replace words with synonyms. Do NOT group skills or hide them in parentheses. Do NOT expand or compress phrases. If a skill appears in the JD, it must appear verbatim in this list. Exact text matching > readability > style. CRITICAL CLASSIFICATION RULE: A skill is a SOFT SKILL if it describes: personality traits, interpersonal abilities, communication style, behavioral characteristics, emotional intelligence, work ethic, leadership traits, strategic thinking, or how someone interacts with others. Examples of SOFT SKILLS: committed, consistent, creative, caring, collaborative, adaptable, proactive, organized, detail-oriented, resilient, empathetic, patient, reliable, trustworthy, efficient, effective, visionary, strategic thinker, innovative leader, strong communication skills, ability to prioritize, time management, problem-solving mindset, critical thinking, working well in teams, leadership abilities, etc. CRITICAL: Do NOT classify technical/domain skills as soft skills. For example: 'analytical' as a personality trait = soft skill, but 'Advanced analytics' or 'data analysis' = hard skill. 'strategic thinking' as behavior = soft skill, but 'strategic planning' or 'Financial Planning' = hard skill. 'innovative mindset' or 'visionary' = soft skill, but 'Technology Innovation' = hard skill. Extract EVERY soft skill mentioned, including adjectives, verb phrases, and behavioral descriptors. Use EXACT verbatim wording from the job description - preserve hyphens, spacing, capitalization, and punctuation exactly as written. Extract ALL variations as separate entries. Missing any soft skill = failure.",
        default_factory=list
    )
    documents_and_deliverables: List[str] = Field(
        description="ALL documentation artifacts and deliverables mentioned. CRITICAL EXTRACTION RULE - VERBATIM ONLY: Do NOT rephrase, paraphrase, or semantically substitute. Use ONLY the exact wording found in the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Extract examples like: BRD (Business Requirements Documents), SOP (Standard Operating Procedures), UAC (User Acceptance Criteria), RFP (Request for Proposal), process maps, process mapping tools, business rules, requirements documentation, etc. Use exact verbatim wording from the JD - no canonical forms, no normalization.",
        default_factory=list
    )
    methodologies: List[str] = Field(
        description="ALL methodologies and frameworks mentioned. CRITICAL EXTRACTION RULE - VERBATIM ONLY: Do NOT rephrase, paraphrase, or semantically substitute. Use ONLY the exact wording found in the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Extract examples like: Agile, Scrum, Project Management, Lean Six Sigma, etc. Use exact verbatim wording from the JD - no canonical forms, no normalization.",
        default_factory=list
    )
    compliance_and_regulations: List[str] = Field(
        description="ALL compliance terms and regulations mentioned. CRITICAL EXTRACTION RULE - VERBATIM ONLY: Do NOT rephrase, paraphrase, or semantically substitute. Use ONLY the exact wording found in the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Extract examples like: ERISA, ACA (Affordable Care Act), HIPAA, GDPR, SOX, NIST, FedRAMP, ISO, SOC, etc. Use exact verbatim wording from the JD - if the JD says 'ERISA', extract 'ERISA' (not 'ERISA Compliance'). If it says 'affordable care act', extract exactly as written. No normalization, no canonical forms.",
        default_factory=list
    )
    keywords: List[str] = Field(
        description="ALL keywords, phrases, action verbs, industry-specific terminology, domain expertise phrases, business concepts, qualifications, certifications, and experience requirements mentioned ANYWHERE in the job description (including role summary, responsibilities, requirements, preferred requirements, company description, mission, vision, benefits, branding sections) that will be woven into resume summary and work experience bullets. EXHAUSTIVE EXTRACTION: Scan the ENTIRE job description - if a keyword appears in ANY section, it MUST be extracted. Do NOT filter based on 'importance' or 'marketing' - presence alone is sufficient. CRITICAL EXTRACTION RULE - VERBATIM ONLY: Do NOT rephrase, paraphrase, infer, or semantically substitute any keyword. Use ONLY the exact wording found in the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Do NOT replace words with synonyms. Do NOT group keywords or hide them in parentheses. Do NOT expand or compress phrases. UNIQUE SURFACE FORMS: Each unique surface form must be listed separately. If unsure whether a phrase is a keyword, INCLUDE it. MUST include everything for ATS matching. Use EXACT verbatim wording from job description.",
        default_factory=list
    )
    literal_phrases: List[str] = Field(
        description="CRITICAL: ALL verb-based phrases and action phrases from the job description that MUST appear VERBATIM in the resume for ATS matching. Extract EXACTLY as written in JD - no normalization, no canonicalization, no lemmatization. Examples: 'manage multiple projects', 'effective communication', 'project delivery', 'collaborating with stakeholders', 'working under pressure', 'analyze data', 'document business rules'. These are byte-level string matches required by ATS tools like Jobscan. Preserve exact capitalization, spacing, punctuation, and word order.",
        default_factory=list
    )
    hybrid_phrases: List[HybridPhrase] = Field(
        description="DUAL-WRITE phrases: Verb-based phrases that need BOTH exact JD phrase (verbatim) AND canonical form (noun-based) for ATS matching. Each entry has 'literal' (VERBATIM exact JD phrase - preserve exact wording, capitalization, spacing, punctuation) and 'canonical' (noun-based canonical form created for dual-write purposes). CRITICAL: The 'literal' field must be EXACTLY as written in the JD - no rephrasing, no normalization. Examples: if JD says 'manage multiple projects', literal = 'manage multiple projects' (verbatim) with canonical 'Project Management' (noun form for dual-write); if JD says 'effective communication', literal = 'effective communication' (verbatim) with canonical 'Communication Skills' (noun form for dual-write). Both forms must appear in resume - literal (verbatim) for ATS byte-matching, canonical (noun form) for semantic understanding.",
        default_factory=list
    )


# Removed: FilteredSkills class - filtering now done directly in extract_job_skills


async def detect_domains(raw_text: str) -> List[str]:
    """Detect ALL applicable domains from job description using LLM. Multi-domain allowed."""
    if not raw_text or not raw_text.strip():
        return ["general"]
    
    try:
        system_prompt = """Detect ALL applicable job domains/types by intelligently analyzing the job description content.

INSTRUCTIONS:
- Analyze the job description: responsibilities, required skills, industry context, role type, and any domain indicators
- Use your knowledge to infer domains from ANY industry, field, or role - not limited to examples
- Consider: technical vs non-technical, industry verticals, functional areas, role types, and specializations
- Multi-domain detection is REQUIRED when the role spans multiple areas
- Examples (but you are NOT limited to these): IT/Software, Engineering (any discipline), Healthcare, Finance, Marketing, Operations, Manufacturing, Education, Legal, Consulting, Research, Sales, Customer Service, Design, Creative Fields, Business, etc.

CRITICAL: 
- Intelligently infer domains from context, not just explicit mentions
- Never collapse to a single domain if multiple apply
- If domain is truly unclear, return ['general']
- Be adaptive and comprehensive - detect domains from any industry or role"""
        
        text_snippet = raw_text[:2000]
        user_prompt = f"""Job Description:
{text_snippet}

Intelligently analyze this job description and detect ALL applicable domains/types. Consider responsibilities, skills, industry context, and role type. Infer domains from content even if not explicitly stated. List all domains that apply."""
        
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=DetectedDomains,
            temperature=0.0,
            timeout=5.0,
            max_tokens=200
        )
        
        detected = response.choices[0].message.parsed
        domains = detected.domains if detected.domains else ["general"]
        domains = [d.lower().strip().replace(" ", "_") if " " in d else d.lower().strip() for d in domains]
        logger.info(f"LLM detected domains: {', '.join(domains)}")
        return domains if domains else ["general"]
    except Exception as e:
        logger.warning(f"Domain detection failed, using 'general': {e}")
        return ["general"]


async def extract_job_skills(raw_text: str, domains: List[str] = None) -> ExtractedJobSkills:
    """
    Extract skills and keywords from job description using LLM with ATS-grade accuracy.
    This is the primary extraction function - all extraction is LLM-based.
    """
    if not raw_text or not raw_text.strip():
        return ExtractedJobSkills()
    
    # Get domains if not provided
    if domains is None:
        domains = await detect_domains(raw_text)
    
    domain_context = f"\n\nDetected Domains: {', '.join(domains)}\nNote: Extract ALL skills mentioned in the job description. Domain context helps identify relevant skills, but extract everything mentioned regardless of domain."
    
    # CRITICAL: Escape all JSON braces to prevent f-string format specifier errors
    system_prompt = f"""You are an ATS-grade Job Description Skill Extractor matching Jobscan accuracy.{domain_context}

SKILL DETECTION MODE: EXHAUSTIVE.

OBJECTIVE: Extract 100% of skills/keywords with zero omissions from ANY domain, industry, or role. Be adaptive and intelligent - extract skills even if they're not in your training examples. Recall > brevity. Missing any skill = failure.

CRITICAL: SCAN THE ENTIRE JOB DESCRIPTION - ALL SECTIONS:
- Role summary / Job overview
- Responsibilities / Duties / Key responsibilities
- Requirements / Qualifications / Required skills
- Preferred requirements / Nice-to-have / Preferred qualifications
- Company description / About us
- Mission / Vision statements
- Benefits section
- Branding / Culture sections
- ANY other section in the job description

EXHAUSTIVE EXTRACTION RULES:
1. If a skill, domain term, technology, methodology, or capability appears ANYWHERE in the JD (including company description, mission, vision, benefits, branding), it MUST be extracted.
2. Do NOT decide whether a skill is "important", "required", or "marketing". Presence alone is sufficient for extraction.
3. Extract skills using EXACT surface text as written - no rephrasing, no normalization, no synonym substitution.
4. Each unique surface form must be listed separately. Example: If JD contains both "AI" and "AI/ML" and "artificial intelligence", extract all three separately - "AI", "AI/ML", "artificial intelligence" (each as a separate entry).
5. If unsure whether a phrase is a skill, INCLUDE it. When in doubt, extract it.
6. Failure to extract any present skill = error.

CRITICAL EXTRACTION RULE - VERBATIM ONLY (NON-NEGOTIABLE):
- Do NOT rephrase, paraphrase, infer, or semantically substitute any skill or keyword
- Use ONLY the exact wording found in the job description
- Preserve surface form exactly (spelling, spacing, capitalization, punctuation, hyphens)
- Do NOT replace words with synonyms
- Do NOT group skills or hide them in parentheses
- Do NOT expand or compress phrases
- If a skill appears in the JD, it must appear verbatim in the extraction at least once
- Exact text matching > readability > style
- ATS systems do byte-level string matching - exact wording is critical

EXTRACTION RULES:
1. Domains: Intelligently detect ALL applicable domains from ANY industry or field. Use context, responsibilities, and required skills to infer domains - do not limit to examples. Multi-domain allowed.

2. EXHAUSTIVE EXTRACTION from ALL sections - VERBATIM ONLY (NO REPHRASING):
   Scan the ENTIRE job description (role summary, responsibilities, requirements, preferred requirements, company description, mission, vision, benefits, branding sections). If a skill appears ANYWHERE, extract it. Do NOT filter based on section or perceived importance. Each unique surface form must be listed separately.
   
   - Hard skills: Extract ALL technical, operational, domain-specific skills, tools, technologies, methodologies, equipment, software, platforms, frameworks, standards, protocols, certifications, licenses, and competencies mentioned or implied ANYWHERE in the JD. Extract from company descriptions, mission statements, benefits sections if they mention technologies/capabilities. CRITICAL: MUST extract as HARD SKILLS - Advanced analytics, artificial intelligence, machine learning, AI/ML, business administration, Financial Planning, Technology Innovation, Security Solutions, logistics, S3, data requirements, cloud computing, cybersecurity, strategic planning, operations management, financial analysis, supply chain management, and manual dexterity (if for technical/operational tasks). Extract these EXACTLY as written in the JD - do NOT normalize, do NOT create canonical forms, do NOT rephrase. UNIQUE SURFACE FORMS: If JD contains 'AI', 'AI/ML', and 'artificial intelligence', extract all three separately. Examples of domains (but extract verbatim from JD): IT, Business, Analytics, Innovation, Operations, Security, Engineering, Healthcare, Creative, etc. CRITICAL: If responsibilities imply specific skills mentioned elsewhere in the JD, extract the exact phrases used - do NOT create new phrases or canonical forms. CRITICAL EXCLUSION RULE: DO NOT extract experience-phrased items (e.g., "experience with X", "proficiency in X", "familiarity with X", "ability to use X") as HARD SKILLS. These descriptive phrases should be extracted as KEYWORDS instead (they will be routed to Experience/Summary sections). Extract the underlying skill name (e.g., "X" from "experience with X") as a HARD SKILL if it's a standalone noun-based technical skill. Hard Skills section must contain ONLY standalone noun-based skills (e.g., "Python", "AWS", "Data Analysis"), NOT descriptive phrases.
   - Soft skills: ONLY interpersonal/leadership/collaboration/communication/behavioral traits that describe personality, behavior, or how someone interacts. Extract from ALL sections of the JD. Extract EXACTLY as written: committed, organized, collaborative, adaptable, visionary, strategic thinker, innovative leader, collaborating with stakeholders, ability to prioritize, strong communication, etc. DO NOT include technical/domain capabilities as soft skills - those are HARD SKILLS. Extract verbatim - do NOT rephrase.
   - Documents/deliverables: ALL documentation artifacts, deliverables, reports, specifications mentioned ANYWHERE in the JD. Extract EXACTLY as written: BRD, SOP, UAC, RFP, process maps, etc. Do NOT expand abbreviations unless the JD explicitly expands them.
   - Methodologies: ALL methodologies, frameworks, approaches mentioned ANYWHERE in the JD. Extract EXACTLY as written: Agile, Scrum, Project Management, Lean Six Sigma, etc. Do NOT normalize.
   - Compliance: ALL compliance terms, regulations, standards, certifications mentioned ANYWHERE in the JD. Extract EXACTLY as written: ERISA, ACA, HIPAA, GDPR, ISO, etc. Do NOT normalize (e.g., if JD says 'erisa', extract 'erisa', not 'ERISA Compliance').
   - Keywords: ALL terms, phrases, qualifications, certifications, experience requirements, domain expertise, industry terminology mentioned ANYWHERE in the JD for ATS matching. Do NOT filter based on 'importance' or 'marketing' - presence alone is sufficient. Extract EXACTLY as written - do NOT rephrase, do NOT use synonyms. UNIQUE SURFACE FORMS: List each unique surface form separately. CRITICAL: Experience-phrased items (e.g., "experience with X", "proficiency in X", "familiarity with X", "ability to use X") MUST be extracted as KEYWORDS (NOT as hard_skills). These will be routed to Experience/Summary sections. Extract the full phrase verbatim (e.g., "experience with Python", "proficiency in data analysis") as keywords.

3. LITERAL PHRASES (byte-exact, NO normalization):
   - Extract verb-based and action phrases EXACTLY as written: "manage multiple projects", "effective communication", "collaborating with stakeholders"
   - Preserve capitalization, spacing, punctuation, word order for ATS byte-level matching
   - Examples: "manage multiple projects" (NOT "Project Management"), "working under pressure" (NOT "Work Under Pressure")

4. HYBRID PHRASES (dual-write: literal + canonical):
   - For verb-based phrases with canonical forms, extract BOTH
   - Format: {{"literal": "exact JD phrase", "canonical": "noun-based form"}}
   - Examples: {{"literal": "manage projects", "canonical": "Project Management"}}, {{"literal": "analyze data", "canonical": "Data Analysis"}}

5. Classification (CRITICAL - accurate classification AND verbatim extraction is essential):
   
   HARD SKILLS = Extract if it involves (classify correctly AND extract VERBATIM from JD):
   - Tools, technologies, software, platforms, systems, equipment, machinery (e.g., if JD says "S3", extract "S3" verbatim; if JD says "AWS", extract "AWS" verbatim)
   - Technical knowledge, domain expertise, specialized training (e.g., if JD says "machine learning", extract "machine learning" verbatim; if JD says "artificial intelligence", extract "artificial intelligence" verbatim; if JD says "Advanced analytics", extract "Advanced analytics" verbatim - preserve capitalization)
   - Produces deliverables, analysis, or technical outcomes (e.g., if JD says "data analysis", extract "data analysis" verbatim; if JD says "financial analysis", extract "financial analysis" verbatim)
   - Industry/domain-specific expertise (e.g., if JD says "business administration", extract "business administration" verbatim; if JD says "Financial Planning", extract "Financial Planning" verbatim - preserve capitalization)
   - Data, analytics, or technical processes (e.g., if JD says "data requirements", extract "data requirements" verbatim)
   - Certifications, licenses, standards knowledge (e.g., if JD says "ISO", extract "ISO" verbatim)
   - Operational capabilities (e.g., if JD says "logistics", extract "logistics" verbatim; if JD says "supply chain management", extract "supply chain management" verbatim)
   - Innovation as a capability (e.g., if JD says "Technology Innovation", extract "Technology Innovation" verbatim - preserve capitalization)
   - Physical skills for technical tasks (e.g., if JD says "manual dexterity", extract "manual dexterity" verbatim)
   
   CRITICAL EXCLUSION RULE FOR HARD SKILLS:
   - DO NOT extract experience-phrased items as HARD SKILLS (e.g., "experience with Python", "proficiency in data analysis", "familiarity with AWS", "ability to use Docker")
   - These descriptive phrases MUST be extracted as KEYWORDS instead (they will be routed to Experience/Summary sections)
   - Hard Skills section must contain ONLY standalone noun-based skills (e.g., "Python", "AWS", "Data Analysis"), NOT descriptive phrases
   - If JD says "experience with Python", extract "Python" as a HARD SKILL (if it's a technical skill) AND "experience with Python" as a KEYWORD (verbatim)
   - If JD says "proficiency in data analysis", extract "data analysis" as a HARD SKILL (if it's a technical skill) AND "proficiency in data analysis" as a KEYWORD (verbatim)
   
   EXPLICITLY EXTRACT AS HARD SKILLS (not soft skills) - EXTRACT VERBATIM:
   - If JD mentions: Advanced analytics, artificial intelligence, machine learning, AI/ML → extract exactly as written
   - If JD mentions: business administration, Financial Planning, financial analysis → extract exactly as written (preserve capitalization)
   - If JD mentions: Technology Innovation, innovation management → extract exactly as written
   - If JD mentions: Security Solutions, cybersecurity, information security → extract exactly as written (preserve capitalization)
   - If JD mentions: logistics, supply chain management → extract exactly as written
   - If JD mentions: S3, AWS S3, cloud computing → extract exactly as written (preserve capitalization, spacing, slashes)
   - If JD mentions: data requirements, data analysis, data modeling → extract exactly as written
   - If JD mentions: manual dexterity (for technical tasks) → extract "manual dexterity" verbatim
   
   SOFT SKILLS = Extract ALL personality traits, behavioral characteristics, ethical traits, and interpersonal qualities (classify correctly AND extract VERBATIM from JD):
   - Personality traits or behavioral characteristics (e.g., if JD says "committed", extract "committed" verbatim; if JD says "organized", extract "organized" verbatim; if JD says "trustworthy", extract "trustworthy" verbatim; if JD says "ethical", extract "ethical" verbatim; if JD says "honest", extract "honest" verbatim; if JD says "reliable", extract "reliable" verbatim; if JD says "dependable", extract "dependable" verbatim; if JD says "visionary", extract "visionary" verbatim; if JD says "proactive", extract "proactive" verbatim)
   - Ethical and integrity traits (e.g., "trustworthy", "ethical", "honest", "integrity", "transparent", "accountable", "responsible", "moral", "principled" - extract ALL verbatim if mentioned in JD)
   - Interpersonal abilities (e.g., if JD says "collaborative", extract "collaborative" verbatim; if JD says "team player", extract "team player" verbatim)
   - Communication style (e.g., if JD says "strong communication", extract "strong communication" verbatim; if JD says "excellent communication skills", extract "excellent communication skills" verbatim)
   - Work ethic or behavioral descriptors (e.g., if JD says "ability to prioritize", extract "ability to prioritize" verbatim; if JD says "detail-oriented", extract "detail-oriented" verbatim)
   - How someone interacts with others (e.g., if JD says "collaborating with stakeholders", extract "collaborating with stakeholders" verbatim; if JD says "working with teams", extract "working with teams" verbatim)
   - Leadership and strategic thinking traits (e.g., if JD says "visionary", extract "visionary" verbatim; if JD says "strategic thinker", extract "strategic thinker" verbatim; if JD says "innovative leader", extract "innovative leader" verbatim)
   - CRITICAL: Extract ALL soft skills comprehensively - if ANY personality trait, behavioral characteristic, ethical quality, work ethic descriptor, or interpersonal quality is mentioned in the JD, extract it verbatim. Do NOT skip soft skills because they're not in the examples above. Examples are illustrative, not exhaustive. Extract ALL soft skills mentioned in the JD regardless of whether they appear in examples.
   
   DO NOT classify as soft skills (these are HARD SKILLS - extract verbatim):
   - Technical capabilities (analytics, AI/ML, security) → these are HARD SKILLS, extract verbatim
   - Domain expertise (business administration, financial planning) → these are HARD SKILLS, extract verbatim
   - Operational capabilities (logistics, supply chain) → these are HARD SKILLS, extract verbatim
   - Innovation as a capability → HARD SKILL, extract verbatim
   
   VERBATIM EXTRACTION RULE: Extract ALL skills exactly as written in the JD - no rephrasing, no normalization, no canonical forms (except hybrid_phrases.canonical for dual-write). Preserve exact spelling, spacing, capitalization, punctuation, hyphens.

6. NEVER drop: ANY skill, tool, technology, methodology, compliance term, domain expertise, certification, or competency mentioned ANYWHERE in the JD (including company description, mission, vision, benefits, branding sections). Extract comprehensively from ANY domain or industry.

7. NO FILTERING: Do NOT filter out skills based on section location (company description, benefits, branding, etc.). Do NOT decide if a skill is "marketing" or "not important". If a skill appears in the JD, extract it regardless of which section it's in. ONLY filter: broken formatting, contact info (phone numbers, email addresses), and clearly non-skill employment terms (e.g., "full-time", "remote work" if not a skill requirement). Extract skills from company descriptions, mission statements, benefits sections, and branding sections if they mention technologies, methodologies, or capabilities.

ADAPTIVE EXTRACTION - VERBATIM ONLY: 
- Extract skills from ANY domain, even if unfamiliar - use context clues from the JD
- If responsibilities mention specific skills elsewhere in the JD, extract those exact phrases verbatim
- Extract industry-specific terminology and domain expertise EXACTLY as written in the JD
- Be comprehensive and intelligent - do not skip skills because they're not in examples
- CRITICAL: If a skill is implied, look for the exact wording used elsewhere in the JD - do NOT create new phrases or canonical forms

VALIDATION - EXHAUSTIVE MODE:
- Scan the ENTIRE job description from beginning to end
- Check ALL sections: role summary, responsibilities, requirements, preferred requirements, company description, mission, vision, benefits, branding
- Compare extracted items against the FULL JD text - if any skill/keyword appears ANYWHERE and is missing from extraction, re-run extraction until complete
- Verify that each unique surface form is listed separately (e.g., "AI", "AI/ML", "artificial intelligence" as separate entries if all appear)
- If unsure whether something is a skill, err on the side of inclusion - extract it
- Missing any present skill = error - re-extract until exhaustive

OUTPUT FORMAT: Return structured response with domains (intelligently inferred), hard_skills (VERBATIM - exact wording from JD), soft_skills (VERBATIM - exact wording from JD), documents_and_deliverables (VERBATIM), methodologies (VERBATIM), compliance_and_regulations (VERBATIM), keywords (VERBATIM), literal_phrases (byte-exact verbatim), hybrid_phrases (list of {{"literal": str (verbatim), "canonical": str}}).

CRITICAL: ALL fields (except hybrid_phrases.canonical) = VERBATIM byte-exact extraction from JD. Do NOT rephrase, paraphrase, normalize, or create canonical forms. Preserve exact spelling, spacing, capitalization, punctuation. hybrid_phrases should have literal (verbatim from JD) and canonical (noun form for dual-write). Completeness > conciseness. Exact text matching > readability > style. Be adaptive and extract from ANY domain, but ALWAYS use verbatim wording."""
    
    user_prompt = f"""Job Description (FULL TEXT - SCAN ENTIRELY):
{raw_text}

EXHAUSTIVE EXTRACTION MODE: Scan the ENTIRE job description from beginning to end. Extract skills from ALL sections:
- Role summary / Job overview
- Responsibilities / Duties
- Requirements / Qualifications
- Preferred requirements / Nice-to-have
- Company description / About us
- Mission / Vision statements
- Benefits section
- Branding / Culture sections
- ANY other section

CRITICAL RULES:
1. If a skill appears ANYWHERE in the JD (including company description, mission, vision, benefits, branding), it MUST be extracted
2. Do NOT filter based on "importance" or "marketing" - presence alone is sufficient
3. Each unique surface form must be listed separately (e.g., "AI", "AI/ML", "artificial intelligence" as separate entries if all appear)
4. If unsure whether a phrase is a skill, INCLUDE it
5. Extract skills from company descriptions if they mention technologies, methodologies, or capabilities

Intelligently extract ALL skills, keywords, and requirements from this FULL job description. Be adaptive and comprehensive - extract skills from ANY domain, industry, or role, even if mentioned in non-traditional sections.

CRITICAL EXTRACTION INSTRUCTIONS - VERBATIM ONLY (NO REPHRASING):
- VERBATIM RULE: Do NOT rephrase, paraphrase, infer, or semantically substitute ANY skill or keyword. Use ONLY exact wording from the job description. Preserve surface form exactly (spelling, spacing, capitalization, punctuation). Do NOT replace words with synonyms. Do NOT group skills or hide them in parentheses. Do NOT expand or compress phrases. Exact text matching > readability > style.
- LITERAL PHRASES: Extract verb-based and action phrases byte-exact to literal_phrases (e.g., "manage multiple projects", "effective communication", "working under pressure") - NO normalization, preserve exact wording from JD verbatim
- HYBRID PHRASES: For verb-based phrases with canonical forms, create entries with both "literal" (exact JD phrase verbatim) and "canonical" (noun form for dual-write) keys
- HARD SKILLS: Extract ALL technical capabilities, tools, technologies, methodologies, domain expertise, certifications, standards, and competencies EXACTLY as written in the JD. CRITICAL: MUST extract as HARD SKILLS (not soft skills): Advanced analytics, artificial intelligence, machine learning, AI/ML, business administration, Financial Planning, Technology Innovation, Security Solutions, logistics, S3, data requirements, cloud computing, cybersecurity, strategic planning, operations management, financial analysis, supply chain management, manual dexterity (for technical/operational tasks). Extract these EXACTLY as written - do NOT normalize, do NOT create canonical forms, do NOT rephrase. If mentioned in JD, extract verbatim from ANY domain (IT, engineering, business, healthcare, creative, operations, etc.). Extract comprehensively - if it involves tools, technology, domain expertise, or produces deliverables/analysis, it's a HARD SKILL, but extract it VERBATIM. CRITICAL EXCLUSION: DO NOT extract experience-phrased items (e.g., "experience with X", "proficiency in X", "familiarity with X") as HARD SKILLS - extract them as KEYWORDS instead. Extract the underlying skill (e.g., "X") as a HARD SKILL if it's a standalone noun-based technical skill. Hard Skills section must contain ONLY standalone noun-based skills.
- SOFT SKILLS: Extract ALL interpersonal, behavioral, communication, leadership, ethical, and integrity traits that describe personality, behavior, work ethic, or character EXACTLY as written. Examples include (but are NOT limited to): committed, organized, collaborative, adaptable, visionary, strategic thinker, innovative leader, collaborating with stakeholders, ability to prioritize, strong communication, trustworthy, ethical, honest, reliable, dependable, transparent, accountable, responsible, moral, principled, integrity. CRITICAL: Extract ALL soft skills comprehensively - if ANY personality trait, behavioral characteristic, ethical quality, work ethic descriptor, interpersonal quality, or character trait is mentioned in the JD, extract it verbatim. Do NOT skip soft skills because they're not in the examples. Examples are illustrative, not exhaustive. DO NOT classify technical/domain capabilities (analytics, AI/ML, business administration, logistics, etc.) as soft skills - those are HARD SKILLS. Extract verbatim - do NOT rephrase.
- DOCUMENTS/METHODOLOGIES/COMPLIANCE: Extract ALL mentioned documentation, methodologies, compliance terms, standards, certifications
- KEYWORDS: Extract ALL important terms, qualifications, domain expertise, industry terminology for ATS matching. CRITICAL: Experience-phrased items (e.g., "experience with X", "proficiency in X", "familiarity with X") MUST be extracted as KEYWORDS (verbatim) - they will be routed to Experience/Summary sections. Extract the full phrase verbatim.

VALIDATION - EXHAUSTIVE MODE: 
- Scan the ENTIRE job description from beginning to end - check ALL sections (role summary, responsibilities, requirements, preferred requirements, company description, mission, vision, benefits, branding)
- Intelligently detect domains from the FULL job content (not just requirements section)
- Extract ALL skills mentioned or implied from ANY section - be adaptive to ANY domain or role
- CRITICAL EXHAUSTIVE CHECK: Verify that skills appearing in company description, mission, vision, benefits, or branding sections are also extracted (e.g., if company description mentions "cloud computing", extract it even if not in requirements)
- CRITICAL UNIQUE SURFACE FORMS: Verify that each unique surface form is listed separately - if JD contains "AI", "AI/ML", and "artificial intelligence", all three must be extracted as separate entries
- CRITICAL CLASSIFICATION CHECK: Verify that Advanced analytics, artificial intelligence, machine learning, AI/ML, business administration, Financial Planning, Technology Innovation, Security Solutions, logistics, S3, data requirements are extracted as HARD SKILLS (not soft skills) - and extract them VERBATIM as written in the JD
- CRITICAL VERBATIM CHECK: Verify that ALL extracted skills/keywords match the exact wording from the JD - no rephrasing, no synonyms, no normalization, no canonical forms (except hybrid_phrases.canonical for dual-write)
- Preserve exact capitalization, spacing, punctuation, hyphens, spelling - byte-level exactness
- Validate completeness - if a skill appears ANYWHERE in the JD and is missing from extraction = failure (re-extract)
- If unsure whether a phrase is a skill, err on the side of inclusion - extract it
- If responsibilities imply skills, look for exact wording used elsewhere in the JD - extract that verbatim, do NOT create new phrases
- Double-check classification: Technical/domain capabilities = HARD SKILLS, personality/behavioral/ethical/character traits = SOFT SKILLS. CRITICAL: ALL personality traits, behavioral characteristics, ethical qualities (trustworthy, ethical, honest, integrity, transparent, accountable, responsible, moral, principled, etc.), work ethic descriptors, leadership traits (visionary, strategic thinker, innovative leader, proactive, etc.), and interpersonal qualities mentioned in the JD must be extracted as SOFT SKILLS verbatim. Do NOT skip any soft skills - extract comprehensively. If JD mentions "visionary", "proactive", "strategic thinker", or any leadership/behavioral trait, extract it as a SOFT SKILL.
- Final exhaustive check: Scan entire JD again - Exact text matching > readability > style - if a skill appears ANYWHERE in JD, it must appear verbatim in extraction. Verify ALL soft skills (including ethical traits like trustworthy, ethical, honest, and leadership traits like visionary, proactive, strategic thinker) are extracted."""
    
    # Attempt extraction with defensive parsing
    extracted = None
    raw_response_text = None
    
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ExtractedJobSkills,
            temperature=0.0,
            timeout=30.0,  # OPTIMIZATION: Reduced from 40.0 to 30.0 for faster failure detection
            max_tokens=2500  # OPTIMIZATION: Reduced from 3000 to 2500 for faster processing while maintaining quality
        )
        
        extracted = response.choices[0].message.parsed
        
        # Log extraction health
        hard_count = len(extracted.hard_skills) if extracted.hard_skills else 0
        soft_count = len(extracted.soft_skills) if extracted.soft_skills else 0
        keywords_count = len(extracted.keywords) if extracted.keywords else 0
        docs_count = len(extracted.documents_and_deliverables) if extracted.documents_and_deliverables else 0
        methods_count = len(extracted.methodologies) if extracted.methodologies else 0
        compliance_count = len(extracted.compliance_and_regulations) if extracted.compliance_and_regulations else 0
        
        # Log ERROR if soft skills are missing (critical for ATS matching)
        if soft_count == 0:
            logger.error(
                f"EXTRACTION HEALTH CHECK FAILED: soft_skills=0 - This is critical! Job descriptions typically contain soft skills. "
                f"Please check if soft skills are mentioned in the JD (e.g., committed, consistent, creative, caring, collaborative, etc.)."
            )
        
        # Log ERROR if any critical count is 0
        if hard_count == 0 or keywords_count == 0:
            logger.error(
                f"EXTRACTION HEALTH CHECK FAILED: hard_skills={hard_count}, soft_skills={soft_count}, "
                f"keywords={keywords_count}, documents={docs_count}, methodologies={methods_count}, "
                f"compliance={compliance_count}"
            )
        else:
            logger.info(
                f"Extracted: {hard_count} hard skills, {soft_count} soft skills, {docs_count} documents/deliverables, "
                f"{methods_count} methodologies, {compliance_count} compliance/regulations, {keywords_count} keywords"
            )
        
        # OPTIMIZATION: Removed fallback extraction to reduce latency (saves 5-8 seconds)
        # Main extraction is reliable - comprehensive prompt ensures all skills are extracted on first try
        # If hard_count is 0, it's likely the JD truly has no technical skills (e.g., pure soft-skill roles)
        if hard_count == 0 and len(raw_text) > 300:
            logger.warning(
                f"Main extraction returned 0 hard skills for JD length {len(raw_text)}. "
                f"This may indicate a non-technical role or JD without technical requirements. "
                f"Main extraction prompt is comprehensive and should extract all skills if present."
            )
            # Continue with original extraction - functionality preserved, no fallback needed
        
        # Final validation: Fail loudly if still empty
        if len(extracted.hard_skills) == 0 and len(raw_text) > 300:
            error_msg = (
                f"CRITICAL: Job description extraction returned 0 hard skills. "
                f"JD length: {len(raw_text)} chars. "
                f"This is a failure - resume generation should not proceed with empty job-derived skills."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return extracted
        
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Log the raw response if available for debugging
        if extracted is None:
            logger.error(f"LLM extraction failed completely - no parsed response. Error: {e}")
            try:
                # Try to get raw response text if available
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    raw_response_text = e.response.text
                    logger.error(f"Raw LLM response (first 500 chars): {raw_response_text[:500]}")
            except:
                pass
        else:
            logger.error(f"LLM extraction failed during parsing. Partial data may be available. Error: {e}")
        
        # Never silently return empty skills - raise the error
        raise ValueError(f"Job description extraction failed: {str(e)}") from e


# Removed: filter_noise_llm, FilteredSkillsResult - filtering now done directly in extract_job_skills via enhanced prompt
# Removed: dedupe_keywords - now using unified dedupe_items_set from utils.py


async def enrich_job_description(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point: Extract all skills and keywords with ATS-grade accuracy.
    Flow: raw_text → LLM extraction (with filtering built-in) → deduplication → return enriched job.
    """
    raw_text = job.get('raw_text', '')
    
    if not raw_text or not raw_text.strip():
        logger.warning("No raw_text provided, returning empty skill lists")
        return {
            **job,
            "domain": "general",
            "domains": ["general"],
            "required_skills": [],
            "soft_skills": [],
            "missing_keywords": [],
            "ats_required_keywords": [],
            "ats_canonical_required": [],
        }
    
    logger.info("Starting LLM-based job enrichment (ATS-grade extraction)...")
    
    # Step 1: Extract all skills/keywords in ONE LLM call with new format
    # CRITICAL: Never silently return empty skills - fail loudly if extraction fails
    try:
        extracted = await extract_job_skills(raw_text)
    except ValueError as e:
        # Validation errors - fail loudly, do not continue
        logger.error(f"CRITICAL: Job description extraction validation failed: {e}")
        raise RuntimeError(f"Job description extraction failed - cannot proceed with resume generation: {e}") from e
    except Exception as e:
        # Other extraction errors - fail loudly
        logger.error(f"CRITICAL: Job description extraction failed: {e}")
        raise RuntimeError(f"Job description extraction failed - cannot proceed with resume generation: {e}") from e
    
    # Validate extraction results
    hard_skills_count = len(extracted.hard_skills) if extracted.hard_skills else 0
    soft_skills_count = len(extracted.soft_skills) if extracted.soft_skills else 0
    all_keywords_count = (
        len(extracted.hard_skills) + len(extracted.soft_skills) + len(extracted.keywords) +
        len(extracted.documents_and_deliverables) + len(extracted.methodologies) +
        len(extracted.compliance_and_regulations)
    )
    
    if hard_skills_count == 0 and len(raw_text) > 300:
        error_msg = (
            f"CRITICAL: Job description extraction returned 0 hard skills for JD > 300 chars. "
            f"Cannot proceed with resume generation. JD length: {len(raw_text)}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if all_keywords_count == 0:
        error_msg = (
            f"CRITICAL: Job description extraction returned 0 total extracted items. "
            f"Cannot proceed with resume generation. JD length: {len(raw_text)}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Step 2: Get domains (from extraction or detect if not present)
    domains = extracted.domains if extracted.domains else await detect_domains(raw_text)
    domain = domains[0] if domains else "general"
    
    # Step 3: Combine all extracted items for keywords list
    # CRITICAL: DO NOT deduplicate across categories - ATS expects redundancy
    # Each category may have overlapping terms (e.g., "Project Management" in hard_skills and "manage projects" in literal_phrases)
    # This redundancy is REQUIRED for ATS matching
    all_keywords = (
        list(extracted.hard_skills) +
        list(extracted.soft_skills) +
        list(extracted.documents_and_deliverables) +
        list(extracted.methodologies) +
        list(extracted.compliance_and_regulations) +
        list(extracted.keywords) +
        list(extracted.literal_phrases if extracted.literal_phrases else [])
    )
    # Add canonical forms from hybrid phrases
    if extracted.hybrid_phrases:
        for hybrid in extracted.hybrid_phrases:
            # Handle both dict format (from JSON) and HybridPhrase model format
            if isinstance(hybrid, dict):
                # Add both literal and canonical forms
                if "literal" in hybrid and hybrid["literal"]:
                    all_keywords.append(hybrid["literal"])
                if "canonical" in hybrid and hybrid["canonical"]:
                    all_keywords.append(hybrid["canonical"])
            elif hasattr(hybrid, 'literal') and hasattr(hybrid, 'canonical'):
                # Handle HybridPhrase model instance
                if hybrid.literal:
                    all_keywords.append(hybrid.literal)
                if hybrid.canonical:
                    all_keywords.append(hybrid.canonical)
            elif isinstance(hybrid, str):
                # Fallback: if it's just a string, add it
                all_keywords.append(hybrid)
    
    # DO NOT deduplicate - preserve all phrases for ATS redundancy
    # Only remove truly empty entries
    all_keywords = [k for k in all_keywords if k and str(k).strip()]
    
    # Step 4: Build enriched job dict
    # CRITICAL: Include literal_phrases and hybrid_phrases for ATS byte-level matching
    enriched_job = {
        **job,
        "domain": domain,
        "domains": domains if domains else [domain],
        "required_skills": sorted(extracted.hard_skills) if extracted.hard_skills else [],
        "soft_skills": sorted(extracted.soft_skills) if extracted.soft_skills else [],
        "literal_phrases": sorted(extracted.literal_phrases) if extracted.literal_phrases else [],
        "hybrid_phrases": [{"literal": h.literal, "canonical": h.canonical} if hasattr(h, 'literal') else h for h in (extracted.hybrid_phrases if extracted.hybrid_phrases else [])],  # Convert HybridPhrase models to dicts for JSON serialization
        "missing_keywords": all_keywords,  # DO NOT sort - preserve order for redundancy
        "ats_required_keywords": all_keywords,  # Include all for ATS matching
        "ats_canonical_required": all_keywords,  # Include all (canonical forms are in hybrid_phrases)
    }
    
    # Log extraction health - use ERROR level if any critical count is 0
    hard_count = len(extracted.hard_skills)
    soft_count = len(extracted.soft_skills)
    docs_count = len(extracted.documents_and_deliverables)
    methods_count = len(extracted.methodologies)
    compliance_count = len(extracted.compliance_and_regulations)
    keywords_count = len(extracted.keywords)
    
    log_func = logger.error if (hard_count == 0 or soft_count == 0) else logger.info
    
    literal_count = len(extracted.literal_phrases) if extracted.literal_phrases else 0
    hybrid_count = len(extracted.hybrid_phrases) if extracted.hybrid_phrases else 0
    
    log_func(f"=== JOB ENRICHMENT COMPLETE - ALL EXTRACTED ITEMS ===")
    log_func(f"Domains ({len(domains)}): {', '.join(domains)}")
    log_func(f"Required Hard Skills ({hard_count}): {', '.join(str(s) for s in sorted(extracted.hard_skills)[:30] if s)}{'...' if hard_count > 30 else ''}")
    log_func(f"Soft Skills ({soft_count}): {', '.join(str(s) for s in sorted(extracted.soft_skills)[:30] if s)}{'...' if soft_count > 30 else ''}")
    log_func(f"Literal Phrases ({literal_count}) - BYTE-EXACT FOR ATS: {', '.join(str(s) for s in sorted(extracted.literal_phrases)[:20] if extracted.literal_phrases and s)}{'...' if literal_count > 20 else ''}")
    hybrid_phrases_display = []
    if extracted.hybrid_phrases:
        for h in extracted.hybrid_phrases[:10]:
            # Handle both dict format and HybridPhrase model format
            if isinstance(h, dict):
                literal = h.get('literal', '')
                canonical = h.get('canonical', '')
            elif hasattr(h, 'literal') and hasattr(h, 'canonical'):
                literal = h.literal
                canonical = h.canonical
            else:
                continue
            hybrid_phrases_display.append(f"{{literal: {literal}, canonical: {canonical}}}")
    hybrid_display_str = ', '.join(hybrid_phrases_display) if hybrid_phrases_display else 'NONE'
    if hybrid_count > 10:
        hybrid_display_str += '...'
    log_func(f"Hybrid Phrases ({hybrid_count}) - DUAL-WRITE: {hybrid_display_str}")
    log_func(f"Documents & Deliverables ({docs_count}): {', '.join(str(s) for s in sorted(extracted.documents_and_deliverables) if s)}")
    log_func(f"Methodologies ({methods_count}): {', '.join(str(s) for s in sorted(extracted.methodologies) if s)}")
    log_func(f"Compliance & Regulations ({compliance_count}): {', '.join(str(s) for s in sorted(extracted.compliance_and_regulations) if s)}")
    log_func(f"All Keywords (NO deduplication - ATS redundancy preserved) ({len(all_keywords)}): {', '.join(str(k) for k in all_keywords[:50] if k)}{'...' if len(all_keywords) > 50 else ''}")
    log_func("=== END JOB ENRICHMENT LOG ===")
    
    return enriched_job


