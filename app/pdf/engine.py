import os
import subprocess
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import jinja2
import openai
from app.schemas.resume import ClientProfile, ResumeContent

# Configure logging
logger = logging.getLogger(__name__)

# Function to escape LaTeX special chars
def escape_latex(text: str) -> str:
    if not text:
        return ""
    chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(chars.get(c, c) for c in text)


def _capitalize_education_degree(degree: str) -> str:
    """Capitalize education degree (e.g., 'bachelor in computer science' -> 'Bachelor in Computer Science')."""
    if not degree or not degree.strip():
        return degree or ""
    
    # Common degree types to capitalize
    degree_lower = degree.lower().strip()
    
    # Map common degree types
    degree_type_map = {
        'bachelor': 'Bachelor',
        'master': 'Master',
        'phd': 'PhD',
        'doctorate': 'Doctorate',
        'associate': 'Associate',
    }
    
    # Check if it starts with a degree type
    for key, value in degree_type_map.items():
        if degree_lower.startswith(key):
            # Extract the rest (field of study)
            remainder = degree_lower[len(key):].strip()
            if remainder.startswith('in ') or remainder.startswith('of '):
                # Format: "bachelor in computer science" -> "Bachelor in Computer Science"
                preposition = remainder[:3]  # "in " or "of "
                field = remainder[3:].strip()  # "computer science"
                # Capitalize field using title case
                field_title = ' '.join(word.capitalize() for word in field.split())
                return f"{value} {preposition}{field_title}"
            else:
                # Format: "bachelor" -> "Bachelor"
                return value
    
    # If no known degree type, capitalize each word (title case)
    return ' '.join(word.capitalize() for word in degree_lower.split())


def _capitalize_education_institution(institution: str) -> str:
    """Capitalize education institution name using title case."""
    if not institution or not institution.strip():
        return institution or ""
    
    # Capitalize each word, handling common acronyms
    words = institution.strip().split()
    capitalized_words = []
    
    for word in words:
        word_lower = word.lower()
        # Keep common acronyms fully capitalized
        if word_lower in ['sliet', 'mit', 'stanford', 'harvard']:
            capitalized_words.append(word.upper())
        else:
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

class PDFEngine:
    def __init__(self, template_dir: str = "app/pdf/templates", output_dir: str = "output"):
        self.template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.template_env = jinja2.Environment(
            loader=self.template_loader,
            block_start_string='\\BLOCK{',
            block_end_string='}',
            variable_start_string='\\VAR{',
            variable_end_string='}',
            comment_start_string='\\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False,
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def render_latex(self, profile: ClientProfile, content: ResumeContent) -> str:
        start_time = datetime.now()
        logger.info(f"render_latex started at {start_time}")
        template = self.template_env.get_template("resume.tex")

        # CRITICAL: Debug skills before rendering
        if content.technical_skills:
            logger.info(f"PDF Engine: Received {len(content.technical_skills)} skill categories")
            for i, cat in enumerate(content.technical_skills):
                logger.info(f"PDF Engine: Category {i+1} '{cat.category}' has {len(cat.skills) if cat.skills else 0} skills: {cat.skills[:3] if cat.skills else 'None'}")
        else:
            logger.warning("PDF Engine: content.technical_skills is None or empty!")

        # Build skills dict - filter out empty categories and empty skills
        skills_dict = {}
        for cat in content.technical_skills or []:
            if cat and cat.category and cat.skills and len(cat.skills) > 0:
                # Filter out empty/None skills and escape them
                escaped_skills = [escape_latex(str(s).strip()) for s in cat.skills if s and str(s).strip()]
                if escaped_skills:  # Only add category if it has non-empty skills
                    skills_dict[escape_latex(cat.category.strip())] = escaped_skills
        
        logger.info(f"PDF Engine: Built skills dict with {len(skills_dict)} categories: {list(skills_dict.keys())}")

        # Escape all string fields
        context = {
            "name": escape_latex(f"{profile.first_name} {profile.last_name}".strip()),
            "email": escape_latex(profile.contact_email),
            "phone": escape_latex(profile.phone_number),
            
            "linkedin": escape_latex(profile.linkedin_url),
            "github": escape_latex(profile.github_url),
            "summary": escape_latex(content.profile_summary),
            "location": escape_latex(content.location) if content.location else (escape_latex(profile.location) if profile.location else ""),
            "skills": skills_dict,
            "experience": [
                {
                    "company": escape_latex(e.company),
                    "role": escape_latex(e.role),
                    "duration": escape_latex(e.duration),
                    "highlights": [escape_latex(h) for h in e.highlights],
                }
                for e in content.experience
            ],
            "projects": [
                {
                    "name": escape_latex(p.name),
                    "tech_stack": [escape_latex(t) for t in p.tech_stack],
                    "description": [escape_latex(d) for d in p.description],
                    "link": escape_latex(p.link) if p.link else ""
                }
                for p in content.projects
            ],
            "education": [
                {
                    "institution": escape_latex(_capitalize_education_institution(e.institution)) if e.institution else "",
                    "degree": escape_latex(_capitalize_education_degree(e.degree)),
                    "year": escape_latex(e.year),
                }
                for e in profile.education
            ],
        }

        rendered = template.render(**context)
        end_time = datetime.now()
        logger.info(f"render_latex ended at {end_time}. Duration: {end_time - start_time}")
        return rendered

    def get_page_count(self, pdf_path: str) -> int:
        """Check the number of pages in a PDF using pdfinfo."""
        start_time = datetime.now()
        logger.info(f"get_page_count started at {start_time}")
        if not os.path.exists(pdf_path):
            return 0
        try:
            result = subprocess.run(
                ["/usr/bin/pdfinfo", pdf_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            for line in result.stdout.splitlines():
                if line.startswith("Pages:"):
                    count = int(line.split(":")[1].strip())
                    end_time = datetime.now()
                    logger.info(f"get_page_count ended at {end_time}. Duration: {end_time - start_time}")
                    return count
        except Exception as e:
            print(f"Error checking page count: {e}")
        return 0
    
    def _estimate_white_space_fallback(self, pdf_path: str) -> float:
        """
        Fallback method using LaTeX log analysis when pdfplumber is not available.
        """
        try:
            log_path = pdf_path.replace(".pdf", ".log")
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                    
                    # Check for \vfill usage (indicates white space being filled)
                    vfill_count = log_content.count("\\vfill")
                    underfull_count = log_content.count("Underfull")
                    
                    # If vfill is used and few underfull warnings, estimate white space
                    if vfill_count > 0 and underfull_count < 3:
                        # Estimate 5-15% white space if vfill is used
                        return min(0.15, vfill_count * 0.03)
                    
                    if underfull_count > 3:
                        return min(0.3, underfull_count * 0.05)
            
            # Conservative estimate if we can't detect
            return 0.05
        except Exception:
            return 0.05

    def estimate_white_space(self, pdf_path: str) -> float:
        """
        Accurately measure white space at the bottom of the first page by analyzing PDF content.
        Returns a ratio (0.0 = no white space, 1.0 = entire page is white space).
        Uses pdfplumber for accurate measurement, falls back to log analysis if unavailable.
        """
        if not os.path.exists(pdf_path):
            return 1.0  # Assume full white space if PDF doesn't exist
        
        # Try to use pdfplumber for accurate measurement
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    return 1.0
                
                page = pdf.pages[0]
                page_height = float(page.height)
                
                # Find the bottom-most content position
                bottom_most = 0.0
                content_found = False
                
                # Check text objects (words, characters)
                words = page.extract_words()
                if words:
                    for word in words:
                        bottom_most = max(bottom_most, float(word['bottom']))
                        content_found = True
                
                # Check for other content objects (rectangles, lines, etc.)
                # These might represent section dividers or formatting
                for obj_type in ['rect', 'line', 'curve']:
                    objects = page.objects.get(obj_type, [])
                    for obj in objects:
                        if 'bottom' in obj:
                            bottom_most = max(bottom_most, float(obj['bottom']))
                            content_found = True
                        elif 'y0' in obj:
                            # Some objects use y0/y1 instead of bottom/top
                            bottom_most = max(bottom_most, float(obj.get('y0', 0)))
                            content_found = True
                
                # Calculate white space ratio
                if content_found and bottom_most > 0:
                    # Account for bottom margin (0.3in = ~21.6 points)
                    # But also account for some tolerance - content might be slightly above margin
                    bottom_margin = 21.6
                    tolerance = 10.0  # 10pt tolerance for content near bottom
                    effective_page_height = page_height - bottom_margin
                    
                    # Calculate white space from bottom of content to bottom margin
                    # If content is within tolerance of bottom, consider it filled
                    if bottom_most >= (effective_page_height - tolerance):
                        white_space_ratio = 0.0  # Content is at bottom, no white space
                    else:
                        white_space_height = effective_page_height - bottom_most
                        white_space_ratio = white_space_height / page_height
                    
                    # Ensure ratio is between 0 and 1
                    white_space_ratio = max(0.0, min(1.0, white_space_ratio))
                    
                    logger.info(f"White space measurement: page_height={page_height:.2f}, bottom_most={bottom_most:.2f}, effective_height={effective_page_height:.2f}, white_space_ratio={white_space_ratio:.2%}")
                    return white_space_ratio
                
                # If no content found, assume full white space
                return 1.0
                
        except ImportError:
            # pdfplumber not available, use fallback method
            logger.warning("pdfplumber not available, using fallback log-based white space estimation")
            return self._estimate_white_space_fallback(pdf_path)
        except Exception as e:
            logger.warning(f"Error estimating white space with pdfplumber: {e}, falling back to log analysis")
            return self._estimate_white_space_fallback(pdf_path)

    async def _rewrite_resume_to_fill_whitespace(self, content: ResumeContent, profile: ClientProfile, white_space_ratio: float) -> ResumeContent:
        """
        Use LLM to rewrite resume content to fill white space.
        Only extends work experience bullet points, NOT the summary.
        """
        import json
        
        # Calculate how much content to add based on white space
        # 1% white space ≈ 8-10 points ≈ 1-2 lines ≈ 20-30 words
        # So 15% white space ≈ 300-450 words needed
        
        white_space_percent = white_space_ratio * 100
        bullets_to_add = max(1, int(white_space_percent / 5))  # ~1 bullet per 5% white space
        
        # Build current resume content as JSON
        # NOTE: Education comes from profile, not content, so we don't include it in the rewrite
        current_resume = {
            "profile_summary": content.profile_summary,
            "experience": [
                {
                    "role": exp.role,
                    "company": exp.company,
                    "duration": exp.duration,
                    "highlights": exp.highlights
                }
                for exp in content.experience
            ],
            "technical_skills": [
                {
                    "category": cat.category,
                    "skills": cat.skills
                }
                for cat in (content.technical_skills or [])
            ],
            "projects": [
                {
                    "name": proj.name,
                    "description": proj.description,
                    "link": proj.link
                }
                for proj in (content.projects or [])
            ]
        }
        
        total_experience_bullets = sum(len(exp.highlights) for exp in content.experience)
        
        rewrite_instruction = f"""The resume has {white_space_percent:.1f}% white space at the bottom. You need to fill this space by expanding content.

CURRENT RESUME CONTENT:
{json.dumps(current_resume, indent=2)}

INSTRUCTIONS TO FILL WHITE SPACE:
1. DO NOT MODIFY SUMMARY: Keep the profile_summary exactly as it is. Do NOT extend, modify, or change it in any way.

2. ADD EXPERIENCE BULLETS: Add approximately {bullets_to_add} new bullet points total across all work experience roles (currently {total_experience_bullets} bullets total). For each work experience role:
   - Add 1-2 new professional bullet points (distribute {bullets_to_add} bullets across all roles)
   - Each bullet should be 25-35 words long
   - Include quantified metrics (%, $, #) when possible
   - Match the style and format of existing bullets
   - Be relevant to the role and responsibilities
   - DO NOT remove any existing bullets, only ADD new ones

3. PRESERVE ALL EXISTING CONTENT: Do NOT remove, shorten, or modify any existing content. Only ADD new bullet points to work experience.

4. MAINTAIN FORMAT: Keep all existing formatting, structure, and style.

5. RETURN FORMAT: Return the complete rewritten resume content in JSON format with this structure:
{{
  "profile_summary": "keep summary exactly as is - DO NOT modify",
  "experience": [
    {{
      "role": "role name",
      "company": "company name",
      "duration": "duration",
      "highlights": ["bullet 1", "bullet 2", "bullet 3", ...]  // Include ALL existing bullets PLUS new ones
    }}
  ],
  "technical_skills": [...],  // Keep exactly as is
  "projects": [...]  // Keep exactly as is
}}

Return ONLY the JSON object, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert resume writer. Your task is to fill white space in resumes by adding experience bullets ONLY. Do NOT modify the summary. Preserve ALL existing content and maintain professional quality. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": rewrite_instruction
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000,  # OPTIMIZATION: Reduced from 3000 to 2000 for faster processing
                timeout=20.0  # OPTIMIZATION: Add timeout to prevent hanging
            )
            
            rewritten_json = json.loads(response.choices[0].message.content)
            
            # Update content with rewritten version - DO NOT update summary
            # Keep summary unchanged
            if "experience" in rewritten_json and content.experience:
                for i, exp_data in enumerate(rewritten_json["experience"]):
                    if i < len(content.experience) and "highlights" in exp_data:
                        original_count = len(content.experience[i].highlights)
                        content.experience[i].highlights = exp_data["highlights"]
                        new_count = len(content.experience[i].highlights)
                        if new_count > original_count:
                            logger.info(f"Added {new_count - original_count} bullet(s) to {content.experience[i].role} (now {new_count} total)")
            
            logger.info(f"Resume rewritten successfully to fill {white_space_percent:.1f}% white space (only work experience bullets extended, summary unchanged)")
            return content
            
        except Exception as e:
            logger.error(f"Failed to rewrite resume via LLM: {e}")
            raise

    def _compile_to_pdf(self, tex_file_path: str, filename: str) -> Optional[str]:
        """Runs pdflatex to compile a PDF from a .tex file."""
        start_time = datetime.now()
        logger.info(f"_compile_to_pdf started at {start_time}")
        tex_filename = f"{filename}.tex"
        pdf_file_path = os.path.join(self.output_dir, f"{filename}.pdf")
        
        try:
            for _ in range(2):
                process = subprocess.run(
                    [
                        "/usr/bin/pdflatex",
                        "-interaction=nonstopmode",
                        tex_filename,
                    ],
                    cwd=self.output_dir,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            
            if os.path.exists(pdf_file_path) and os.path.getsize(pdf_file_path) > 0:
                end_time = datetime.now()
                logger.info(f"_compile_to_pdf ended at {end_time}. Duration: {end_time - start_time}")
                return pdf_file_path
            else:
                return None
        except Exception as e:
            print(f"LaTeX compilation error: {e}")
            return None

    async def generate_pdf(self, profile: ClientProfile, content: ResumeContent, filename: str) -> Dict[str, Any]:
        """
        Generates a PDF resume and ensures it fits on one page.
        Retries up to 3 times by reducing content if necessary.
        """
        start_time = datetime.now()
        logger.info(f"generate_pdf started at {start_time}")
        
        # Initial LaTeX generation
        latex_content = self.render_latex(profile, content)
        tex_file_path = os.path.join(self.output_dir, f"{filename}.tex")
        
        max_retries = 3
        current_pdf_path = None
        content_expansion_attempted = False  # Track if we've tried expanding content for this PDF generation
        
        for attempt in range(max_retries + 1):
            # Write LaTeX content to file
            with open(tex_file_path, "w", encoding="utf-8") as f:
                f.write(latex_content)
            
            # Compile to PDF
            current_pdf_path = self._compile_to_pdf(tex_file_path, filename)
            
            if not current_pdf_path:
                print(f"Attempt {attempt}: Failed to generate PDF.")
                break
                
            # Check page count
            page_count = self.get_page_count(current_pdf_path)
            print(f"Attempt {attempt}: Generated PDF has {page_count} page(s).")
            
            if page_count <= 1:
                # Success: Fits on one page
                # Check for white space at the bottom
                white_space_ratio = self.estimate_white_space(current_pdf_path)
                if white_space_ratio > 0.03:  # More than 3% white space detected (acceptable: 3-4%)
                    logger.warning(f"White space detected (ratio: {white_space_ratio:.2%}). Attempting to fill... (Target: <4%)")
                    
                    # NEW APPROACH: Use LLM to rewrite resume content to fill white space
                    # Always use LLM rewrite when white space > 3% (instead of looping with incremental additions)
                    logger.info(f"White space {white_space_ratio:.2%} > 3% - Using LLM to rewrite resume content to fill space")
                    try:
                        content = await self._rewrite_resume_to_fill_whitespace(content, profile, white_space_ratio)
                        # Re-render LaTeX with rewritten content
                        latex_content = self.render_latex(profile, content)
                        
                        # CRITICAL FIX: Disable \flushbottom and reduce all stretchability to prevent content spreading
                        import re
                        # Disable \flushbottom - it causes content to spread vertically
                        latex_content = latex_content.replace("\\flushbottom", "%\\flushbottom  % DISABLED to prevent content spreading")
                        
                        # Reduce ALL stretchability to minimum to prevent spreading
                        # Reduce vspace stretchability
                        latex_content = re.sub(
                            r'\\vspace\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                            lambda m: f"\\vspace{{{m.group(1)}pt plus 1pt minus {m.group(3)}pt}}",
                            latex_content
                        )
                        
                        # Reduce section spacing stretchability
                        latex_content = re.sub(
                            r'\\titlespacing\*\{\\section\}\s*\{[^}]*\}\s*\{[^}]*\}\s*\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                            lambda m: f"\\titlespacing*{{\\section}}{{0pt}}{{8pt plus 1pt minus 2pt}}{{{m.group(1)}pt plus 1pt minus {m.group(3)}pt}}",
                            latex_content
                        )
                        
                        # Reduce parskip stretchability
                        latex_content = re.sub(
                            r'\\setlength\{\\parskip\}\{(\d+)pt plus (\d+)pt',
                            lambda m: f"\\setlength{{\\parskip}}{{{m.group(1)}pt plus 1pt",
                            latex_content
                        )
                        
                        # Reduce itemize stretchability
                        latex_content = re.sub(
                            r'itemsep=(\d+)pt plus (\d+)pt',
                            lambda m: f"itemsep={m.group(1)}pt plus 1pt",
                            latex_content
                        )
                        latex_content = re.sub(
                            r'topsep=(\d+)pt plus (\d+)pt',
                            lambda m: f"topsep={m.group(1)}pt plus 1pt",
                            latex_content
                        )
                        
                        logger.info("Resume rewritten by LLM to fill white space, disabled \\flushbottom and reduced stretchability")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to rewrite resume via LLM: {e}, falling back to incremental approach")
                    
                    # FALLBACK: Incremental content addition (if LLM rewrite failed or not attempted)
                    # STEP 1: Always add content when white space > 3% (only work experience bullets, NOT summary)
                    content_modified = False
                    
                    # ALWAYS add bullet points to work experience when white space > 4%
                    if content.experience and len(content.experience) > 0:
                        # For high white space (>10%), add more bullets; otherwise add 1 per role
                        max_bullets_per_role = 12 if white_space_ratio > 0.10 else 10  # Higher limit for high white space
                        bullets_per_role = 2 if white_space_ratio > 0.10 else 1  # Add 2 bullets if >10% white space
                        
                        for exp in content.experience:
                            if len(exp.highlights) < max_bullets_per_role:
                                bullets_added_this_role = 0
                                for _ in range(bullets_per_role):
                                    if len(exp.highlights) >= max_bullets_per_role:
                                        break
                                    
                                    # Add a new bullet using LLM
                                    try:
                                        expansion_instruction = (
                                            f"The resume has white space. Add ONE new professional bullet point to the work experience "
                                            f"role: '{exp.role} at {exp.company}' ({exp.duration}).\n"
                                            f"The new bullet should:\n"
                                            f"1. Be 20-25 words long (same length as existing bullets)\n"
                                            f"2. Match the style and format of existing bullets\n"
                                            f"3. Include quantified metrics (%, $, #) if possible\n"
                                            f"4. Be relevant to the role and responsibilities\n"
                                            f"5. Use action verbs and professional language\n\n"
                                            f"Existing bullets:\n" + "\n".join([f"- {h}" for h in exp.highlights])
                                        )
                                        
                                        response = self.client.chat.completions.create(
                                            model="gpt-4o",
                                            messages=[
                                                {"role": "system", "content": "You are an expert resume writer. Generate professional resume bullet points that are 20-25 words long, include metrics when possible, and match the style of existing bullets."},
                                                {"role": "user", "content": expansion_instruction}
                                            ],
                                            temperature=0.3,
                                            max_tokens=150
                                        )
                                        
                                        new_bullet = response.choices[0].message.content.strip()
                                        # Clean up the bullet (remove markdown, ensure it's a single line)
                                        new_bullet = new_bullet.replace("*", "").replace("-", "").strip()
                                        if new_bullet.startswith('"') and new_bullet.endswith('"'):
                                            new_bullet = new_bullet[1:-1]
                                        
                                        if new_bullet and len(new_bullet.split()) >= 15:  # Ensure it's substantial
                                            exp.highlights.append(new_bullet)
                                            content_modified = True
                                            bullets_added_this_role += 1
                                            logger.info(f"Added new bullet to {exp.role}: {new_bullet[:50]}...")
                                    except Exception as e:
                                        logger.warning(f"Failed to add bullet via LLM: {e}")
                                        break
                                
                                if bullets_added_this_role > 0:
                                    logger.info(f"Added {bullets_added_this_role} bullet(s) to {exp.role} to fill white space")
                    
                    # Re-render LaTeX if content was modified
                    if content_modified:
                        logger.info("Content modified to fill white space (bullets added to work experience), re-rendering LaTeX...")
                        latex_content = self.render_latex(profile, content)
                        
                        # CRITICAL FIX: Reduce stretchability when adding content to prevent spreading
                        # When new content is added, reduce stretch values so content doesn't spread out
                        import re
                        # Reduce vspace stretchability (keep base spacing but limit stretch to prevent spreading)
                        latex_content = re.sub(
                            r'\\vspace\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                            lambda m: f"\\vspace{{{m.group(1)}pt plus {min(int(m.group(2)), 3)}pt minus {m.group(3)}pt}}",
                            latex_content
                        )
                        
                        # Reduce section spacing stretchability
                        latex_content = re.sub(
                            r'\\titlespacing\*\{\\section\}\s*\{[^}]*\}\s*\{[^}]*\}\s*\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                            lambda m: f"\\titlespacing*{{\\section}}{{0pt}}{{8pt plus 3pt minus 2pt}}{{{m.group(1)}pt plus {min(int(m.group(2)), 2)}pt minus {m.group(3)}pt}}",
                            latex_content
                        )
                        
                        # Reduce itemize stretchability
                        latex_content = re.sub(
                            r'itemsep=(\d+)pt plus (\d+)pt',
                            lambda m: f"itemsep={m.group(1)}pt plus {min(int(m.group(2)), 2)}pt",
                            latex_content
                        )
                        
                        # Reduce parskip stretchability
                        latex_content = re.sub(
                            r'\\setlength\{\\parskip\}\{(\d+)pt plus (\d+)pt',
                            lambda m: f"\\setlength{{\\parskip}}{{{m.group(1)}pt plus {min(int(m.group(2)), 3)}pt",
                            latex_content
                        )
                        
                        logger.info("Reduced stretchability to prevent content spreading after addition")
                        # Recompile with new content
                        continue
                    else:
                        # If content expansion failed, mark as attempted to avoid infinite loops
                        content_expansion_attempted = True
                    
                    # STEP 2: Only apply spacing adjustments if content expansion FAILED
                    # DO NOT apply spacing adjustments if content was just added (it will make it worse)
                    if not content_modified:
                        # Content expansion failed, try spacing adjustments as last resort
                        if "\vfill" not in latex_content:
                            # Add vfill if missing
                            latex_content = latex_content.replace("\\end{document}", "\\vfill\n\\end{document}")
                        
                        # Increase stretchability of vspace commands (only if content addition failed)
                        import re
                        # Replace limited stretchability with more aggressive stretching
                        latex_content = re.sub(
                            r'\\vspace\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                            lambda m: f"\\vspace{{{m.group(1)}pt plus {max(int(m.group(2)) * 2, 8)}pt minus {m.group(3)}pt}}",
                            latex_content
                        )
                        
                        # Increase section spacing stretchability
                        latex_content = re.sub(
                            r'\\titlespacing\*\{\\section\}\s*\{[^}]*\}\s*\{[^}]*\}\s*\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                            lambda m: f"\\titlespacing*{{\\section}}{{0pt}}{{8pt plus 6pt minus 2pt}}{{{m.group(1)}pt plus {max(int(m.group(2)) * 2, 4)}pt minus {m.group(3)}pt}}",
                            latex_content
                        )
                        
                        logger.info("Applied white space filling adjustments to LaTeX (spacing) - content expansion failed")
                        # Recompile with adjusted spacing
                        continue
                elif white_space_ratio > 0.03:  # Between 3-4%, try spacing adjustments only
                    # Try spacing adjustments for marginal white space
                    if "\vfill" not in latex_content:
                        latex_content = latex_content.replace("\\end{document}", "\\vfill\n\\end{document}")
                    
                    # Increase stretchability more aggressively
                    import re
                    latex_content = re.sub(
                        r'\\vspace\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                        lambda m: f"\\vspace{{{m.group(1)}pt plus {max(int(m.group(2)) * 3, 12)}pt minus {m.group(3)}pt}}",
                        latex_content
                    )
                    
                    latex_content = re.sub(
                        r'\\titlespacing\*\{\\section\}\s*\{[^}]*\}\s*\{[^}]*\}\s*\{(\d+)pt plus (\d+)pt minus (\d+)pt\}',
                        lambda m: f"\\titlespacing*{{\\section}}{{0pt}}{{8pt plus 8pt minus 2pt}}{{{m.group(1)}pt plus {max(int(m.group(2)) * 3, 6)}pt minus {m.group(3)}pt}}",
                        latex_content
                    )
                    
                    logger.info(f"Applied aggressive spacing adjustments for marginal white space (ratio: {white_space_ratio:.2%})")
                    continue
                else:
                    # White space is acceptable (≤3%), success
                    logger.info(f"PDF fits on one page with acceptable white space (ratio: {white_space_ratio:.2%}, target: <4%)")
                    break
            
            if attempt < max_retries:
                # The PDF exceeds 1 page, send back to OpenAI for reduction
                print(f"PDF exceeds 1 page. Reduction attempt {attempt + 1}...")
                
                reduction_instruction = (
                    "The current LaTeX resume exceeds one page. Your task is to trim it to EXACTLY one page while keeping it dense and professional.\n"
                    "1. DO NOT shorten the length of bullet points. Every bullet point MUST remain 20-25 words long.\n"
                    "2. REDUCTION PRIORITY (CRITICAL - FOLLOW THIS ORDER STRICTLY):\n"
                    "   - STEP 1: REDUCE WORK EXPERIENCE BULLETS FIRST. Reduce bullets from work experience before touching projects.\n"
                    "     * If there are multiple work experience roles, reduce bullets EQUALLY from EACH role (one bullet from each role, not all from one role).\n"
                    "     * Example: If you have 2 work experience roles with 6 bullets each and need to reduce, remove 1 bullet from EACH role (result: 5 bullets each).\n"
                    "     * Example: If you have 3 work experience roles with 5 bullets each and need to reduce, remove 1 bullet from EACH role (result: 4 bullets each).\n"
                    "     * Example: If you have 4 work experience roles with 4 bullets each and need to reduce, remove 1 bullet from EACH role (result: 3 bullets each).\n"
                    "     * Continue reducing proportionally (1 bullet from each role per reduction cycle) until the page fits.\n"
                    "     * PRIORITIZE keeping bullets that contain quantified metrics (%, $, #) when deciding which bullet to remove from each role.\n"
                    "   - STEP 2: ONLY IF work experience bullets have been reduced to minimum (2-3 bullets per role) AND still need reduction, THEN reduce Summary length:\n"
                    "     * Slightly reduce the Summary paragraph by removing less critical phrases while keeping all keywords and soft skills.\n"
                    "   - CRITICAL: NEVER reduce, omit, or modify project bullets - they must remain exactly as provided in the original content.\n"
                    "   - CRITICAL: NEVER omit entire projects - all projects must remain with all their bullets intact.\n"
                    "   - CRITICAL: NEVER omit sub-skills from the Skills section - all skills must maintain their sub-skills format.\n"
                    "3. Ensure the font size does NOT go below 9pt (current is 10pt).\n"
                    "4. Maintain all technical skills and keyword density.\n"
                    "5. If only a few lines over, you can slightly reduce vertical spacing (e.g., \\vspace commands) if present, but priority is reducing work experience bullets.\n"
                    "Return ONLY the updated LaTeX code."
                )
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert resume editor. You must return only the valid LaTeX code, starting with \\documentclass and ending with \\end{document}. No markdown wrappers."},
                            {"role": "user", "content": f"Current LaTeX content:\n\n{latex_content}\n\nInstruction: {reduction_instruction}"}
                        ],
                        temperature=0.1
                    )
                    
                    new_latex = response.choices[0].message.content.strip()
                    # Basic cleaning of the response
                    if new_latex.startswith("```"):
                        lines = new_latex.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines[-1].startswith("```"):
                            lines = lines[:-1]
                        new_latex = "\n".join(lines).strip()
                    
                    latex_content = new_latex
                    
                except openai.RateLimitError as e:
                    logger.error(f"OpenAI Rate Limit Error (429) during PDF reduction: Too many requests. Error: {str(e)}")
                    break
                except openai.APIError as e:
                    if hasattr(e, 'status_code') and e.status_code == 429:
                        logger.error(f"OpenAI API Error 429 (Too Many Requests) during PDF reduction: {str(e)}")
                    else:
                        logger.error(f"OpenAI API Error during PDF reduction: {str(e)}")
                    break
                except Exception as e:
                    logger.error(f"Error during OpenAI reduction: {str(e)}")
                    break
        
        end_time = datetime.now()
        end_time_str = end_time.isoformat()
        
        # Cleanup intermediate files
        for ext in [".aux", ".log", ".out"]:
            path = os.path.join(self.output_dir, f"{filename}{ext}")
            if os.path.exists(path):
                os.remove(path)
        
        # Also remove the .tex file as per original cleanup
        if os.path.exists(tex_file_path):
             os.remove(tex_file_path)

        end_time = datetime.now()
        logger.info(f"generate_pdf ended at {end_time}. Duration: {end_time - start_time}")

        return {
            "pdf_path": current_pdf_path,
            "latex_content": latex_content,
            "page_count": self.get_page_count(current_pdf_path) if current_pdf_path else 0
        }
