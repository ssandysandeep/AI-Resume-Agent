# Novadrix AI Resume Agent

## Project Goal
Develop an AI Agent for Resume Creation that integrates with Novadrix's existing recruitment platform. The agent will automatically generate professional, job-specific resumes that meet strict quality criteria.

## Core Features
1. **ATS Optimization**: Minimum score of 90%
2. **Single Page Format**: Strict single page constraint
3. **Smart Skills Alignment**: Context-aware skill categorization
4. **Quantified Metrics**: AI-generated impact metrics
5. **Human-Like Quality**: Natural language generation
6. **High Performance**: Generation time ≤ 20-30 seconds

## Tech Stack
- **Language**: Python
- **Framework**: FastAPI
- **AI Agent**: LangGraph + LangChain + Claude LLM
- **PDF Generation**: LaTeX
- **Storage**: AWS S3
- **Database**: Vector DB (for optimization strategies)

## setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run server: `uvicorn main:app --reload`
