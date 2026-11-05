from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging
from datetime import datetime

from app.schemas.resume import ClientProfile, JobDescription, ResumeContent
from app.agent.graph import generate_resume
from app.pdf.engine import PDFEngine
from app.agent.utils import sanitize_error_message
from app.services.s3_uploader import S3Uploader  # Commented out for testing

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Novadrix AI Resume Agent",
    description="Microservice for generating high-quality, ATS-optimized resumes using AI",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PDF Engine
# Initialize services
pdf_engine = PDFEngine()
s3_uploader = S3Uploader()  # Commented out for testing


@app.get("/")
async def root():
    return {"message": "Novadrix AI Resume Agent Service is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/generate_resume")
async def generate_resume_endpoint(profile: ClientProfile, job: JobDescription):
    """
    Main endpoint to generate a resume.
    1. Runs the AI agent to tailor content.
    2. Generates a PDF using LaTeX.
    3. Returns the PDF file (mocking S3 upload for now).
    """
    # 1. Generate Resume Content
    # Reduced logging for latency optimization
    start_time = datetime.now()
    logger.info(f"generate_resume_endpoint started at {start_time}")
    try:
        tailored_content = await generate_resume(profile.model_dump(), job.model_dump())

        if not tailored_content:
            return {"status": "error", "message": "Failed to generate resume content"}
    except ValueError as e:
        # ATS/text-layer preflight failures and one-page failures are surfaced as ValueError
        return {
            "status": "error",
            "message": "ATS/text-layer validation failed (keyword coverage and/or one-page constraint).",
            "details": str(e),
        }
    except Exception as e:
        error_msg = sanitize_error_message(str(e))
        return {"status": "error", "message": f"AI Generation Error: {error_msg}"}

    # 2. Generate PDF
    request_id = str(uuid.uuid4())
    generation_result = await pdf_engine.generate_pdf(
        profile, tailored_content, filename=request_id
    )
    pdf_path = generation_result.get("pdf_path")
    pdf_status = generation_result.get("status")
    page_count = generation_result.get("page_count")

    if not pdf_path or not os.path.exists(pdf_path):
        # Reduced logging for latency
        end_time = datetime.now()
        logger.info(
            f"generate_resume_endpoint ended (partial success) at {end_time}. Duration: {end_time - start_time}"
        )
        return {
            "status": "partial_success",
            "message": "Resume content generated, but PDF compilation failed.",
            "pdf_status": pdf_status,
            "page_count": page_count,
            "s3_url": None,
        }

    if page_count > 1:
        # Compiled but violates one-page constraint. Caller must adjust content before rendering.
        return {
            "status": "error",
            "message": "PDF generated but exceeds one page constraint.",
            "pdf_status": "page_limit_exceeded",
            "page_count": page_count,
            "pdf_path": pdf_path,
            "s3_url": None,
        }

    # 3. Upload to S3 and Return response (COMMENTED OUT FOR TESTING)
    try:
        s3_key = f"resumes/{request_id}.pdf"
        s3_url = s3_uploader.upload_file(pdf_path, s3_key)

        # Cleanup local PDF after successful upload
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        end_time = datetime.now()
        logger.info(
            f"generate_resume_endpoint ended (success) at {end_time}. Duration: {end_time - start_time}"
        )

        return {
            "status": "success",
            "message": "Resume generated and uploaded successfully.",
            "s3_url": s3_url,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"S3 Upload Error: {str(e)}",
            "s3_url": None,
        }

    # For testing: Return local PDF path
    # Reduced logging for latency
    # end_time = datetime.now()
    # logger.info(f"generate_resume_endpoint ended (success) at {end_time}. Duration: {end_time - start_time}")

    # return {
    #     "status": "success",
    #     "message": "Resume generated successfully.",
    #     "pdf_path": pdf_path,
    #     "page_count": page_count,
    #     "s3_url": None
    # }
