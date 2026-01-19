# Flask app with CORS fixes deployed on January 18, 2026
import os
import io
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import re
from datetime import datetime
import threading
import time

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
import stripe

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Image processing and OCR imports
from PIL import Image
import pytesseract


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes with comprehensive headers
CORS(app,
     origins=['http://localhost:5173', 'https://disputemyhoa.com', 'https://www.disputemyhoa.com'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'apikey', 'X-Webhook-Secret', 'X-Requested-With', 'Accept', 'Origin'],
     supports_credentials=True)

# Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
DOC_EXTRACT_WEBHOOK_SECRET = os.environ.get('DOC_EXTRACT_WEBHOOK_SECRET')

# Stripe Configuration
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')
STRIPE_PRICE_ID = os.environ.get('STRIPE_PRICE_ID')
SITE_URL = os.environ.get('SITE_URL', 'https://disputemyhoa.com')

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Configure Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# SMTP Configuration - Optional, only needed for email functionality
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
# Clean SMTP credentials to handle non-ASCII characters like non-breaking spaces
SMTP_USER = (os.environ.get("SMTP_USER") or "").strip().replace('\xa0', ' ')
SMTP_PASS = (os.environ.get("SMTP_PASS") or "").strip().replace('\xa0', ' ')
SMTP_FROM = os.environ.get("SMTP_FROM", "support@disputemyhoa.com")

SMTP_SENDER_WEBHOOK_SECRET = os.environ.get("SMTP_SENDER_WEBHOOK_SECRET")
SMTP_SENDER_WEBHOOK_URL = os.environ.get("SMTP_SENDER_WEBHOOK_URL")

# Request timeouts
TIMEOUT = (5, 60)  # (connect, read)

def supabase_headers() -> Dict[str, str]:
    """Return headers for Supabase API requests."""
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json'
    }

def fetch_ready_documents_by_token(token: str, limit: int = 3) -> List[Dict]:
    """Query Supabase for ready documents by case token."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        params = {
            'token': f'eq.{token}',
            'status': 'eq.ready',
            'select': 'id,filename,mime_type,page_count,char_count,updated_at,extracted_text',
            'order': 'updated_at.desc',
            'limit': str(limit)
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        documents = response.json()
        logger.info(f"Found {len(documents)} ready documents for token {token[:12]}...")
        return documents

    except Exception as e:
        logger.error(f"Failed to fetch ready documents for token {token[:12]}...: {str(e)}")
        return []

def fetch_any_documents_status_by_token(token: str) -> List[Dict]:
    """Check for any documents (including processing/pending) by token."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        params = {
            'token': f'eq.{token}',
            'select': 'id,status,updated_at',
            'order': 'updated_at.desc'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        logger.error(f"Failed to fetch document status for token {token[:12]}...: {str(e)}")
        return []

def summarize_doc_text_with_openai(token: str, raw_text: str) -> str:
    """Summarize document text using OpenAI gpt-4o-mini."""
    try:
        # Clip text to avoid token limits
        clipped_text = raw_text[:12000] if raw_text else ""

        if not clipped_text.strip():
            return ""

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        prompt = """Summarize the HOA notice into: (a) alleged violation, (b) what HOA demands, (c) deadlines/fines/next actions, (d) evidence/rules cited. Be factual, quote exact deadlines/amounts when present. 8-12 bullets max."""

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a document summarizer for HOA notices. Extract key facts concisely."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nDocument text:\n{clipped_text}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()

        # Limit summary length
        if len(summary) > 1200:
            summary = summary[:1200] + "..."

        logger.info(f"Successfully summarized document for token {token[:12]}...")
        return summary

    except Exception as e:
        logger.error(f"Failed to summarize document text for token {token[:12]}...: {str(e)}")
        # Fallback to clipped excerpt
        return raw_text[:1200] + "..." if len(raw_text) > 1200 else raw_text

def build_doc_brief(docs: List[Dict]) -> Dict:
    """Build document brief from ready documents."""
    if not docs:
        return {
            "doc_status": "none",
            "doc_count": 0,
            "sources": [],
            "brief_text": ""
        }

    sources = []
    raw_texts = []

    for doc in docs:
        sources.append({
            "filename": doc.get("filename", "unknown"),
            "page_count": doc.get("page_count", 0),
            "char_count": doc.get("char_count", 0)
        })

        # Extract text with limits
        extracted_text = doc.get("extracted_text", "")
        if extracted_text:
            # Limit per document to 6000 chars
            doc_text = extracted_text[:6000]
            raw_texts.append(doc_text)

    if not raw_texts:
        return {
            "doc_status": "ready",
            "doc_count": len(docs),
            "sources": sources,
            "brief_text": ""
        }

    # Combine texts with total limit of 12000 chars
    combined_text = " ".join(raw_texts)
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    # Try to summarize with OpenAI
    token = docs[0].get("token", "unknown") if docs else "unknown"
    brief_text = summarize_doc_text_with_openai(token, combined_text)

    return {
        "doc_status": "ready",
        "doc_count": len(docs),
        "sources": sources,
        "brief_text": brief_text
    }


def fetch_document_status(document_id: str) -> Optional[Dict[str, Any]]:
    """Fetch current document status from Supabase."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        params = {
            'id': f'eq.{document_id}',
            'select': 'id,token,status'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()
        return data[0] if data else None

    except Exception as e:
        logger.error(f"Failed to fetch document status for {document_id}: {str(e)}")
        return None

def update_document(document_id: str, token: str, updates: Dict[str, Any]) -> bool:
    """Update document in Supabase database."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        params = {
            'id': f'eq.{document_id}',
            'token': f'eq.{token}'
        }
        headers = supabase_headers()
        headers['Prefer'] = 'return=representation'

        response = requests.patch(url, params=params, headers=headers,
                                json=updates, timeout=TIMEOUT)
        response.raise_for_status()

        logger.info(f"Updated document {document_id} with: {updates}")
        return True

    except Exception as e:
        logger.error(f"Failed to update document {document_id}: {str(e)}")
        return False

def download_storage_object(bucket: str, path: str) -> Optional[bytes]:
    """Download file from Supabase Storage."""
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
        headers = {
            'apikey': SUPABASE_SERVICE_ROLE_KEY,
            'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}'
        }

        response = requests.get(url, headers=headers, timeout=TIMEOUT)

        if response.status_code == 404:
            logger.error(f"File not found: {bucket}/{path}")
            return None

        response.raise_for_status()

        logger.info(f"Downloaded {len(response.content)} bytes from {bucket}/{path}")
        return response.content

    except Exception as e:
        logger.error(f"Failed to download {bucket}/{path}: {str(e)}")
        return None

def extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, int, int, Optional[str]]:
    """
    Extract text from PDF bytes.
    Returns: (extracted_text, page_count, char_count, error_message)
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)

        text_parts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page: {str(e)}")
                text_parts.append("")

        extracted_text = "\n\n".join(text_parts)
        char_count = len(extracted_text)

        # Check if text is empty or only whitespace
        if not extracted_text or extracted_text.strip() == "":
            return "", page_count, 0, "No text layer found - document may be scanned and require OCR"

        logger.info(f"Extracted {char_count} characters from {page_count} pages")
        return extracted_text, page_count, char_count, None

    except Exception as e:
        error_msg = f"Failed to extract text from PDF: {str(e)}"
        logger.error(error_msg)
        return "", 0, 0, error_msg


def extract_image_text(image_bytes: bytes, filename: str = "") -> Tuple[str, int, int, Optional[str]]:
    """
    Extract text from image bytes using OCR.
    Returns: (extracted_text, page_count=1, char_count, error_message)
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (some formats like RGBA or P need conversion)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        logger.info(f"Processing image: {image.size[0]}x{image.size[1]} pixels, mode: {image.mode}")

        # Set TESSDATA_PREFIX if not already set (for Heroku compatibility)
        if 'TESSDATA_PREFIX' not in os.environ:
            possible_paths = [
                '/usr/share/tesseract-ocr/5/tessdata',
                '/usr/share/tesseract-ocr/tessdata',
                '/usr/share/tesseract-ocr/4.00/tessdata',
                '/usr/share/tesseract-ocr/4/tessdata',
                '/usr/share/tesseract-ocr/tessdata',
                '/usr/share/tessdata',
                '/app/.apt/usr/share/tesseract-ocr/5/tessdata',
                '/app/.apt/usr/share/tesseract-ocr/tessdata'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    os.environ['TESSDATA_PREFIX'] = path
                    logger.info(f"Set TESSDATA_PREFIX to: {path}")
                    break
            else:
                logger.warning("Could not find tessdata directory in any expected location")

        # Try multiple OCR configurations for better compatibility
        configs_to_try = [
            r'--oem 1 --psm 1 -l eng',   # Automatic page segmentation with OSD
            r'--oem 1 --psm 3 -l eng',   # Fully automatic page segmentation, but no OSD
            r'--oem 1 --psm 4 -l eng',   # Assume a single column of text of variable sizes
            r'--oem 1 --psm 6 -l eng',   # Assume a single uniform block of text
            r'--oem 3 --psm 1 -l eng',   # LSTM with automatic page segmentation
            r'--oem 3 --psm 3 -l eng',   # LSTM with fully automatic page segmentation
            r'--oem 3 --psm 6 -l eng',   # LSTM standard config
            r'--oem 3 --psm 11 -l eng',  # Sparse text - find as much text as possible
            r'--oem 3 --psm 12 -l eng',  # Sparse text with OSD
            r'--psm 6',                  # No language specified fallback
        ]

        extracted_text = ""
        best_text = ""
        best_char_count = 0
        last_error = None

        for config in configs_to_try:
            try:
                logger.info(f"Trying OCR with config: {config}")
                current_text = pytesseract.image_to_string(image, config=config)
                current_text = current_text.strip()
                char_count = len(current_text);

                # Keep track of the best result (most text that looks reasonable)
                if char_count > best_char_count:
                    # Basic heuristic: prefer results with more alphanumeric content
                    alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in current_text) / max(len(current_text), 1)
                    if alphanumeric_ratio > 0.3:  # At least 30% should be readable characters
                        best_text = current_text
                        best_char_count = char_count
                        logger.info(f"New best result: {char_count} chars, {alphanumeric_ratio:.2f} alphanumeric ratio")

                # If we got a decent amount of readable text, we can stop
                if char_count > 50 and best_text:
                    extracted_text = best_text
                    break

            except Exception as e:
                last_error = str(e)
                logger.warning(f"OCR config failed: {config}, error: {str(e)}")
                continue

        # Use the best result we found
        if not extracted_text and best_text:
            extracted_text = best_text

        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        char_count = len(extracted_text)

        if char_count == 0:
            error_msg = f"No text found in image - image may be blank or contain no readable text. Last OCR error: {last_error}"
            return "", 1, 0, error_msg

        logger.info(f"OCR extracted {char_count} characters from image {filename}")
        return extracted_text, 1, char_count, None

    except Exception as e:
        error_msg = f"Failed to extract text from image: {str(e)}"
        logger.error(error_msg)
        return "", 0, 0, error_msg


def is_image_file(filename: str, mime_type: str = "") -> bool:
    """Check if file is a supported image format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    image_mime_types = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
        'image/bmp', 'image/tiff', 'image/webp'
    }

    # Check by file extension
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in image_extensions:
            return True

    # Check by MIME type
    if mime_type and mime_type.lower() in image_mime_types:
        return True

    return False


def is_pdf_file(filename: str, mime_type: str = "") -> bool:
    """Check if file is a PDF."""
    if filename and filename.lower().endswith('.pdf'):
        return True
    if mime_type and mime_type.lower() == 'application/pdf':
        return True
    return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/debug/env', methods=['GET'])
def debug_env():
    """Return presence of critical env vars (gated by DOC_EXTRACT_WEBHOOK_SECRET).

    This does NOT return any secret values, only boolean flags indicating whether each
    required configuration item is set. Intended for quick diagnostics on deployed app.
    """
    secret = request.headers.get('X-Webhook-Secret')
    if not secret or secret != DOC_EXTRACT_WEBHOOK_SECRET:
        logger.warning("Unauthorized request to /debug/env")
        return jsonify({'error': 'Unauthorized'}), 401

    keys = [
        'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'DOC_EXTRACT_WEBHOOK_SECRET',
        'SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASS', 'SMTP_FROM', 'SMTP_SENDER_WEBHOOK_SECRET'
    ]
    presence = {k: bool(os.environ.get(k)) for k in keys}
    logger.info(f"/debug/env requested; presence: { {k: presence[k] for k in presence} }")
    return jsonify({'env_presence': presence}), 200

@app.route('/webhooks/doc-extract', methods=['POST'])
def doc_extract_webhook():
    """Main webhook endpoint for document extraction."""
    # Validate webhook secret
    webhook_secret = request.headers.get('X-Webhook-Secret')
    if not webhook_secret or webhook_secret != DOC_EXTRACT_WEBHOOK_SECRET:
        logger.warning("Invalid or missing webhook secret")
        return jsonify({'error': 'Unauthorized'}), 401

    document_id = None
    token = None

    try:
        # Parse JSON body
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        # Validate required fields
        required_fields = ['token', 'document_id', 'bucket', 'path']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        token = data['token']
        document_id = data['document_id']
        bucket = data['bucket']
        path = data['path']
        filename = data.get('filename', '') or ''  # Handle null values
        mime_type = data.get('mime_type', '') or ''  # Handle null values

        logger.info(f"Processing document extraction - ID: {document_id}, Token: {token[:8]}...")

        # Check if document is already processed
        current_doc = fetch_document_status(document_id)
        if current_doc and current_doc.get('status') == 'ready':
            logger.info(f"Document {document_id} already processed")
            return jsonify({
                'message': 'Document already processed',
                'document_id': document_id,
                'status': 'ready'
            }), 200

        # Mark document as processing
        if not update_document(document_id, token, {'status': 'processing'}):
            return jsonify({
                'error': 'Failed to update document status to processing',
                'document_id': document_id
            }), 500

        # Download file from Supabase Storage
        file_bytes = download_storage_object(bucket, path)
        if file_bytes is None:
            error_msg = f"Failed to download file from {bucket}/{path}"
            update_document(document_id, token, {
                'status': 'failed',
                'error': error_msg[:2000]
            })
            return jsonify({
                'error': error_msg,
                'document_id': document_id
            }), 500

        # Determine file type with multiple fallback strategies
        logger.info(f"Initial file detection - filename: '{filename}', mime_type: '{mime_type}'")

        # Strategy 1: If filename is empty/null, extract from path
        if not filename or filename.lower() == 'null':
            filename = os.path.basename(path)
            logger.info(f"Extracted filename from path: '{filename}'")

        # Strategy 2: If mime_type is empty/null, guess from filename
        if not mime_type or mime_type.lower() == 'null':
            if filename:
                ext = os.path.splitext(filename.lower())[1]
                mime_type_map = {
                    '.pdf': 'application/pdf',
                    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.png': 'image/png', '.gif': 'image/gif',
                    '.bmp': 'image/bmp', '.tiff': 'image/tiff', '.tif': 'image/tiff',
                    '.webp': 'image/webp'
                }
                mime_type = mime_type_map.get(ext, '')
                logger.info(f"Guessed MIME type from extension '{ext}': '{mime_type}'")

        # Strategy 3: If still no filename, try extracting just the filename from the full path
        if not filename:
            # Handle paths like "dmhoa-docs/case_xxx/original/image.jpg"
            path_parts = path.split('/')
            if path_parts:
                filename = path_parts[-1]  # Get the last part
                logger.info(f"Extracted filename from path parts: '{filename}'")

        # Strategy 4: If we still have no clear type, try to detect from file content
        detected_type = None
        if not (is_pdf_file(filename, mime_type) or is_image_file(filename, mime_type)):
            # Check file magic bytes as last resort
            if file_bytes and len(file_bytes) >= 4:
                # PDF magic bytes
                if file_bytes.startswith(b'%PDF'):
                    detected_type = 'pdf'
                    logger.info("Detected PDF from file magic bytes")
                # JPEG magic bytes
                elif file_bytes.startswith(b'\xff\xd8\xff'):
                    detected_type = 'image'
                    mime_type = 'image/jpeg'
                    logger.info("Detected JPEG from file magic bytes")
                # PNG magic bytes
                elif file_bytes.startswith(b'\x89PNG'):
                    detected_type = 'image'
                    mime_type = 'image/png'
                    logger.info("Detected PNG from file magic bytes")

        logger.info(f"Final file detection - filename: '{filename}', mime_type: '{mime_type}', detected_type: {detected_type}")

        # Process based on detected file type
        if is_pdf_file(filename, mime_type) or detected_type == 'pdf':
            logger.info(f"Processing as PDF: {filename}")
            extracted_text, page_count, char_count, extraction_error = extract_pdf_text(file_bytes)
        elif is_image_file(filename, mime_type) or detected_type == 'image':
            logger.info(f"Processing as image: {filename}")
            extracted_text, page_count, char_count, extraction_error = extract_image_text(file_bytes, filename)
        else:
            error_msg = f"Unsupported file type: {filename} (MIME: {mime_type})"
            logger.error(error_msg)
            update_document(document_id, token, {
                'status': 'failed',
                'error': error_msg[:2000]
            })
            return jsonify({
                'error': error_msg,
                'document_id': document_id
            }, 400)

        # Update document with extraction results
        update_data = {
            'status': 'ready',
            'extracted_text': extracted_text[:50000],  # Limit text size
            'page_count': page_count,
            'char_count': char_count
        }

        if extraction_error:
            update_data['error'] = extraction_error[:2000]
            logger.warning(f"Document {document_id} processed with warning: {extraction_error}")
        else:
            logger.info(f"Document {document_id} processed successfully: {char_count} chars, {page_count} pages")

        if not update_document(document_id, token, update_data):
            return jsonify({
                'error': 'Failed to save extraction results',
                'document_id': document_id
            }), 500

        return jsonify({
            'message': 'Document processed successfully',
            'document_id': document_id,
            'status': 'ready',
            'page_count': page_count,
            'char_count': char_count,
            'has_error': bool(extraction_error)
        }), 200

    except Exception as e:
        error_msg = f"Unexpected error processing document: {str(e)}"
        logger.error(error_msg)

        # Try to update document status to failed if we have the IDs
        if document_id and token:
            try:
                update_document(document_id, token, {
                    'status': 'failed',
                    'error': error_msg[:2000]
                })
            except:
                pass  # Don't fail the response if we can't update status

        return jsonify({
            'error': error_msg,
            'document_id': document_id
        }), 500


def read_case_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Fetch case details from Supabase by token."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {
            'token': f'eq.{token}',
            'select': '*'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        cases = response.json()
        if not cases:
            logger.warning(f"No case found for token: {token[:12]}...")
            return None

        case = cases[0]
        logger.info(f"Found case for token {token[:12]}...: ID {case.get('id')}")
        return case

    except Exception as e:
        logger.error(f"Failed to fetch case for token {token[:12]}...: {str(e)}")
        return None


def generate_case_preview_with_openai(case_data: Dict, doc_brief: Dict) -> str:
    """Generate case preview using OpenAI."""
    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
            return "Preview generation unavailable - OpenAI not configured"

        # Extract case information
        hoa_name = case_data.get('hoa_name', 'Unknown HOA')
        violation_type = case_data.get('violation_type', 'Unknown violation')
        case_description = case_data.get('case_description', 'No description provided')

        # Get document brief
        doc_text = doc_brief.get('brief_text', '')
        doc_count = doc_brief.get('doc_count', 0)

        # Prepare the prompt
        prompt = f"""Generate a professional case preview for this HOA dispute:

HOA: {hoa_name}
Violation Type: {violation_type}
Case Description: {case_description}
Documents Analyzed: {doc_count}

Document Summary:
{doc_text[:2000] if doc_text else 'No documents available'}

Create a concise case preview that includes:
1. Brief case summary
2. Key issues identified
3. Potential legal considerations
4. Recommended next steps

Keep it professional, factual, and under 800 words."""

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a legal case analyst specializing in HOA disputes. Provide professional, factual analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1200
        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        preview = result['choices'][0]['message']['content'].strip()

        logger.info(f"Generated case preview: {len(preview)} characters")
        return preview

    except Exception as e:
        logger.error(f"Failed to generate case preview: {str(e)}")
        return f"Error generating preview: {str(e)}"


def save_case_preview(token: str, preview_text: str, doc_brief: Dict) -> bool:
    """Save generated case preview to Supabase."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {'token': f'eq.{token}'}
        headers = supabase_headers()
        headers['Prefer'] = 'return=representation'

        update_data = {
            'preview_text': preview_text[:10000],  # Limit size
            'preview_generated_at': datetime.utcnow().isoformat(),
            'doc_summary': doc_brief
        }

        response = requests.patch(url, params=params, headers=headers,
                                json=update_data, timeout=TIMEOUT)
        response.raise_for_status()

        logger.info(f"Saved case preview for token {token[:12]}...")
        return True

    except Exception as e:
        logger.error(f"Failed to save case preview for token {token[:12]}...: {str(e)}")
        return False


@app.route('/api/case-preview', methods=['POST'])
def case_preview_endpoint():
    """Generate and return case preview with document analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        token = data.get('token')
        if not token:
            return jsonify({'error': 'Missing required field: token'}), 400

        logger.info(f"Processing case preview request for token: {token[:12]}...")

        # Fetch case details
        case = read_case_by_token(token)
        if not case:
            return jsonify({
                'error': 'Case not found',
                'token': token[:12] + '...'
            }), 404

        # Check if preview already exists and is recent
        existing_preview = case.get('preview_text')
        preview_generated_at = case.get('preview_generated_at')

        if existing_preview and preview_generated_at:
            try:
                # If preview was generated less than 1 hour ago, return it
                from datetime import datetime, timedelta
                generated_time = datetime.fromisoformat(preview_generated_at.replace('Z', '+00:00'))
                if datetime.now().replace(tzinfo=generated_time.tzinfo) - generated_time < timedelta(hours=1):
                    logger.info(f"Returning existing preview for token {token[:12]}...")
                    return jsonify({
                        'preview': existing_preview,
                        'cached': True,
                        'generated_at': preview_generated_at
                    }), 200
            except:
                pass  # If there's any issue with date parsing, continue with new generation

        # Fetch and analyze documents
        documents = fetch_ready_documents_by_token(token, limit=5)
        doc_brief = build_doc_brief(documents)

        # Generate case preview
        preview_text = generate_case_preview_with_openai(case, doc_brief)

        # Save the preview
        if save_case_preview(token, preview_text, doc_brief):
            logger.info(f"Successfully generated and saved preview for token {token[:12]}...")
        else:
            logger.warning(f"Generated preview but failed to save for token {token[:12]}...")

        return jsonify({
            'preview': preview_text,
            'doc_summary': doc_brief,
            'cached': False,
            'generated_at': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in case preview endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/case/<token>/status', methods=['GET'])
def case_status(token: str):
    """Get case and document status."""
    try:
        # Fetch case
        case = read_case_by_token(token)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        # Fetch document status
        documents = fetch_any_documents_status_by_token(token)

        # Count documents by status
        doc_counts = {}
        for doc in documents:
            status = doc.get('status', 'unknown')
            doc_counts[status] = doc_counts.get(status, 0) + 1

        return jsonify({
            'case_id': case.get('id'),
            'token': token[:12] + '...',
            'hoa_name': case.get('hoa_name'),
            'violation_type': case.get('violation_type'),
            'documents': {
                'total': len(documents),
                'by_status': doc_counts
            },
            'preview_available': bool(case.get('preview_text'))
        }), 200

    except Exception as e:
        logger.error(f"Error in case status endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Stripe webhook endpoints
@app.route('/webhooks/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events."""
    try:
        payload = request.get_data()
        sig_header = request.headers.get('Stripe-Signature')

        if not STRIPE_WEBHOOK_SECRET:
            logger.error("Stripe webhook secret not configured")
            return jsonify({'error': 'Webhook not configured'}), 500

        # Verify webhook signature
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            return jsonify({'error': 'Invalid payload'}), 400
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            return jsonify({'error': 'Invalid signature'}), 400

        # Handle the event
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            handle_successful_payment(session)
        else:
            logger.info(f"Unhandled Stripe event type: {event['type']}")

        return jsonify({'received': True}), 200

    except Exception as e:
        logger.error(f"Error handling Stripe webhook: {str(e)}")
        return jsonify({'error': str(e)}), 500


def handle_successful_payment(session):
    """Handle successful Stripe payment."""
    try:
        customer_email = session.get('customer_details', {}).get('email')
        metadata = session.get('metadata', {})

        logger.info(f"Processing successful payment for email: {customer_email}")

        # You can add logic here to update user permissions, send confirmation emails, etc.

    except Exception as e:
        logger.error(f"Error handling successful payment: {str(e)}")


@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """Create Stripe checkout session."""
    try:
        if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
            return jsonify({'error': 'Stripe not configured'}), 500

        data = request.get_json()

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_PRICE_ID,
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{SITE_URL}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{SITE_URL}/cancel",
            customer_email=data.get('email') if data else None,
        )

        return jsonify({'checkout_url': session.url}), 200

    except Exception as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Email functionality
def send_email(to_email: str, subject: str, body: str, html_body: str = None) -> bool:
    """Send email via SMTP."""
    try:
        if not all([SMTP_HOST, SMTP_USER, SMTP_PASS]):
            logger.warning("SMTP not configured - cannot send email")
            return False

        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_FROM
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add text part
        text_part = MIMEText(body, 'plain')
        msg.attach(text_part)

        # Add HTML part if provided
        if html_body:
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, to_email, msg.as_string())

        logger.info(f"Email sent successfully to {to_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False


@app.route('/api/save-case', methods=['POST'])
def save_case_endpoint():
    """Save or update case details."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        token = data.get('token')
        if not token:
            return jsonify({'error': 'Missing required field: token'}), 400

        logger.info(f"Processing save case request for token: {token[:12]}...")
        logger.info(f"Request data keys: {list(data.keys())}")

        # Extract and clean case fields from request
        case_data = {}

        # Required fields
        case_data['token'] = token

        # Optional fields - only add if they exist and are not empty
        if data.get('hoa_name') and str(data.get('hoa_name')).strip():
            case_data['hoa_name'] = str(data.get('hoa_name')).strip()

        if data.get('violation_type') and str(data.get('violation_type')).strip():
            case_data['violation_type'] = str(data.get('violation_type')).strip()

        if data.get('case_description') and str(data.get('case_description')).strip():
            case_data['case_description'] = str(data.get('case_description')).strip()

        logger.info(f"Cleaned case data: {case_data}")

        # Check if case exists
        existing_case = read_case_by_token(token)
        if existing_case:
            # Update existing case
            logger.info(f"Updating existing case ID {existing_case.get('id')} for token {token[:12]}...")

            url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            params = {'id': f"eq.{existing_case.get('id')}"}
            headers = supabase_headers()
            headers['Prefer'] = 'return=representation'

            response = requests.patch(url, params=params, headers=headers,
                                    json=case_data, timeout=TIMEOUT)

            logger.info(f"Update response status: {response.status_code}")
            if response.status_code >= 400:
                logger.error(f"Update failed - Response: {response.text}")
                return jsonify({
                    'error': f'Failed to update case: {response.text}',
                    'status_code': response.status_code
                }), 500

            response.raise_for_status()
            updated_case = response.json()
            return jsonify({'message': 'Case updated successfully', 'case': updated_case}), 200
        else:
            # Create new case - add timestamp for creation
            case_data['created_at'] = datetime.utcnow().isoformat() + 'Z'  # Add Z for UTC timezone

            logger.info(f"Creating new case for token {token[:12]}...")
            logger.info(f"Final case data for creation: {case_data}")

            url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            headers = supabase_headers()
            headers['Prefer'] = 'return=representation'

            response = requests.post(url, headers=headers, json=case_data, timeout=TIMEOUT)

            logger.info(f"Create response status: {response.status_code}")
            logger.info(f"Create response headers: {dict(response.headers)}")

            if response.status_code >= 400:
                logger.error(f"Create failed - Response: {response.text}")
                logger.error(f"Request URL: {url}")
                logger.error(f"Request headers: {headers}")
                logger.error(f"Request data: {case_data}")

                # Try to parse the error response
                try:
                    error_details = response.json()
                    logger.error(f"Parsed error details: {error_details}")
                    return jsonify({
                        'error': 'Failed to create case in database',
                        'supabase_error': error_details,
                        'status_code': response.status_code
                    }), 500
                except:
                    return jsonify({
                        'error': 'Failed to create case in database',
                        'supabase_error': response.text,
                        'status_code': response.status_code
                    }), 500

            response.raise_for_status()
            new_case = response.json()

            # Handle both single object and array responses from Supabase
            case_id = None
            if isinstance(new_case, list) and len(new_case) > 0:
                case_id = new_case[0].get('id')
            elif isinstance(new_case, dict):
                case_id = new_case.get('id')

            logger.info(f"New case created successfully with ID: {case_id}")
            return jsonify({
                'message': 'Case saved successfully',
                'case': new_case,
                'case_id': case_id
            }), 201

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error in save case endpoint: {str(e)}")
        error_response = e.response.text if e.response else 'No response details'
        logger.error(f"Error response: {error_response}")
        return jsonify({
            'error': f'Database request failed: {str(e)}',
            'details': error_response
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error in save case endpoint: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
