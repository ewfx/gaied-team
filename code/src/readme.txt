AI-Powered Email Processor

Overview:
This project processes incoming emails, extracts key details, classifies content using AI, and routes requests accordingly.

Tech Stack:
- FastAPI (Backend)
- Hugging Face BART (facebook/bart-large-mnli)
- Libraries: pdfplumber, python-docx, FastAPI, uvicorn, transformers

Installation:
1. Clone the repository:
   git clone <repo-link>
   cd <repo-name>

2. Install dependencies:
   pip install -r requirements.txt

3. Run the FastAPI server:
   uvicorn main:app --reload

4. API available at: http://127.0.0.1:8000/docs

API Usage:
- POST /process_email/
  Upload an email (.eml) file for processing.

Requirements:
- Python 3.x
- Required libraries are listed in `requirements.txt`.