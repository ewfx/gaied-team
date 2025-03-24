import os
import email
import pdfplumber
from email import policy
from email.parser import BytesParser
from docx import Document
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File
import uvicorn
import re
from io import BytesIO
from transformers import pipeline

# FastAPI app
app = FastAPI()

# Load Hugging Face model for classification (e.g., a general-purpose text classification model)
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Root Endpoint to avoid 404 error
@app.get("/")
def home():
    return {"message": "FastAPI Email Processor is running!"}

# Load email and parse content
def parse_email(file):
    file_content = BytesIO(file.read())
    msg = BytesParser(policy=policy.default).parse(file_content)
    body = msg.get_body(preferencelist=("plain", "html"))
    email_body = body.get_content() if body else ""
    
    attachments = []
    for part in msg.iter_attachments():
        file_name = part.get_filename()
        if file_name:
            attachments.append((file_name, BytesIO(part.get_payload(decode=True))))
    return email_body, attachments

# Extract text from attachments
def extract_text_from_attachment(file_name, content):
    text = ""
    try:
        if file_name.endswith(".pdf"):
            with pdfplumber.open(content) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file_name.endswith(".docx"):
            doc = Document(content)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif file_name.endswith(".txt"):
            text = content.read().decode("utf-8")
    except Exception as e:
        text = f"Error reading attachment {file_name}: {str(e)}"
    return text

# Extract key attributes
def extract_fields(text):
    fields = {
        "deal_name": re.search(r"Deal Name: (.+)", text),
        "amount": re.search(r"Amount: \$?(\d+[,.\d]*)", text),
        "expiration_date": re.search(r"Expiration Date: (\d{2}/\d{2}/\d{4})", text)
    }
    return {key: (match.group(1) if match else "N/A") for key, match in fields.items()}

# Classify Email and Extract Fields using Hugging Face classifier
def classify_email(email_body, attachment_text):
    combined_text = email_body + " " + attachment_text
    labels = ["Loan Modification", "Payment Inquiry", "Fraud Report", "General Inquiry"]
    
    response = classifier(combined_text, candidate_labels=labels)
    primary_request = response['labels'][0]
    confidence = response['scores'][0]  # confidence level for the primary request
    
    sub_labels = {
        "Loan Modification": ["Interest Rate Change", "Term Extension"],
        "Payment Inquiry": ["Payment Status", "Payment Method Change"],
        "Fraud Report": ["Unauthorized Transaction", "Identity Theft"],
        "General Inquiry": ["Product Information", "Account Details"]
    }
    
    sub_request_options = sub_labels.get(primary_request, [])
    sub_request = "N/A"
    
    if sub_request_options:
        sub_response = classifier(combined_text, candidate_labels=sub_request_options)
        sub_request = sub_response['labels'][0]
    
    reasoning_fields = {
        "Loan Modification": "Customer requests a change in loan terms.",
        "Payment Inquiry": "Customer seeks details on payments.",
        "Fraud Report": "Customer reports suspicious activity.",
        "General Inquiry": "General customer query."
    }
    reasoning = reasoning_fields.get(primary_request, "No specific reasoning available.")
    
    return primary_request, sub_request, confidence, reasoning

# Priority Mapping
def assign_priority(request_type):
    priority_mapping = {
        "Fraud Report": "High",
        "Loan Modification": "Medium",
        "Payment Inquiry": "Medium",
        "General Inquiry": "Low"
    }
    return priority_mapping.get(request_type, "Low")

# Duplicate Indicator with Reasoning
seen_emails = set()
def is_duplicate(email_body):
    if email_body in seen_emails:
        return True, "Duplicate detected: Similar email content found in the system."
    seen_emails.add(email_body)
    return False, "Unique email."

# Skill-Based Routing
def route_request(request_type):
    routing_table = {
        "Loan Modification": "Modification Team",
        "Payment Inquiry": "Payments Team",
        "Fraud Report": "Fraud Team"
    }
    return routing_table.get(request_type, "General Support Team")

# FastAPI Endpoint to process emails
@app.post("/process_email/")
async def process_email(file: UploadFile = File(...)):
    email_body, attachments = parse_email(file.file)
    attachment_texts = [extract_text_from_attachment(name, content) for name, content in attachments]
    full_text = email_body + "\n".join(attachment_texts)

    extracted_fields = extract_fields(full_text)
    primary_request, sub_request, confidence, reasoning = classify_email(email_body, "\n".join(attachment_texts))
    assigned_team = route_request(primary_request)
    priority = assign_priority(primary_request)
    duplicate, duplicate_reason = is_duplicate(email_body)

    return {
        "primary_request": primary_request,
        "sub_request_type": sub_request,
        "confidence": confidence,
        "reasoning": reasoning,
        "extracted_fields": extracted_fields,
        "duplicate_indicator": duplicate,
        "duplicate_reason": duplicate_reason,
        "assigned_team": assigned_team,
        "priority": priority
    }

# Run the API Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)