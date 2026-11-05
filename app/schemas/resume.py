from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices

class Education(BaseModel):
    institution: Optional[str] = None
    degree: str = ""
    year: str = ""

class Experience(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    highlights: List[str] = Field(default_factory=list)

class Project(BaseModel):
    name: str = ""
    tech_stack: List[str] = Field(default_factory=list)
    description: List[str] = Field(default_factory=list)
    link: Optional[str] = None

class ClientProfile(BaseModel):
    first_name: str = Field(default="", validation_alias=AliasChoices("first_name", "ffirst_name"))
    last_name: str = ""
    contact_email: str = ""
    phone_number: str = ""
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    location: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: str = ""  # Changed from List[Experience] to string
    projects: str = ""  # Changed from List[Project] to string

class JobDescription(BaseModel):
    raw_text: str = ""
    role_title: str = ""
    company_name: str = ""
    # Note: required_skills, soft_skills, and missing_keywords are now automatically extracted from raw_text

class SkillCategory(BaseModel):
    category: str = Field(default="", description="The category of the skill (e.g., 'Programming Languages', 'Frameworks').")
    skills: List[str] = Field(default_factory=list)

class ResumeContent(BaseModel):
    """The AI-generated content optimized for the specific job."""
    profile_summary: str = ""
    technical_skills: List[SkillCategory] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    location: Optional[str] = None
    ats_score: int = 0
    matched_keywords: List[str] = Field(
        default=[],
        description="List of keywords from the job description that are included in this resume"
    )
