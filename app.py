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
# NEW: Updated CORS configuration with comprehensive headers and methods
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "apikey", "x-client-info", "x-supabase-api-version", "X-Webhook-Secret"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

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

# NEW: In-process lock to prevent duplicate preview generation
preview_generation_locks = {}
preview_lock = threading.Lock()

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
                char_count = len(current_text)

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
            }), 400

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

        # Trigger preview update now that document is ready
        try:
            trigger_preview_update_after_document_processing(token)
        except Exception as e:
            logger.warning(f"Failed to trigger preview update after document processing: {str(e)}")
            # Don't fail the response if preview update fails

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


def trigger_preview_update_after_document_processing(token: str):
    """Trigger preview regeneration after a document becomes ready."""
    try:
        case = read_case_by_token(token)
        if not case:
            logger.warning(f"Cannot trigger preview update - case not found for token {token[:8]}...")
            return

        case_id = case.get('id')
        if not case_id:
            logger.warning(f"Cannot trigger preview update - case ID not found for token {token[:8]}...")
            return

        # Trigger preview regeneration with force=True since we have new document data
        success = auto_generate_case_preview(token, case_id, force_regenerate=True)

        if success:
            logger.info(f"Successfully triggered preview update for case {token[:8]}... after document processing")
        else:
            logger.warning(f"Failed to trigger preview update for case {token[:8]}... after document processing")

    except Exception as e:
        logger.error(f"Error triggering preview update after document processing for {token[:8]}...: {str(e)}")


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


def read_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    """Fetch case details from Supabase by case ID."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {
            'id': f'eq.{case_id}',
            'select': '*'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        cases = response.json()
        if not cases:
            logger.warning(f"No case found for ID: {case_id}")
            return None

        case = cases[0]
        logger.info(f"Found case for ID {case_id}: Token {case.get('token', '')[:8]}...")
        return case

    except Exception as e:
        logger.error(f"Failed to fetch case for ID {case_id}: {str(e)}")
        return None


def read_active_preview(case_id: str) -> Optional[Dict]:
    """Read the active preview for a case from dmhoa_case_previews table."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_previews"
        params = {
            'case_id': f'eq.{case_id}',
            'is_active': 'eq.true',
            'select': '*',
            'order': 'created_at.desc',
            'limit': '1'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        previews = response.json()
        if previews:
            logger.info(f"Found active preview for case {case_id}")
            return previews[0]

        logger.info(f"No active preview found for case {case_id}")
        return None

    except Exception as e:
        logger.error(f"Failed to read active preview for case {case_id}: {str(e)}")
        return None


def deactivate_previews(case_id: str) -> bool:
    """Deactivate all existing previews for a case."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_previews"
        params = {'case_id': f'eq.{case_id}'}
        headers = supabase_headers()

        update_data = {'is_active': False}

        response = requests.patch(url, params=params, headers=headers,
                                json=update_data, timeout=TIMEOUT)
        response.raise_for_status()

        logger.info(f"Deactivated existing previews for case {case_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to deactivate previews for case {case_id}: {str(e)}")
        return False


def insert_preview(case_id: str, preview_content: Dict, preview_snippet: str = None,
                  prompt_version: str = None, model: str = "gpt-4o-mini",
                  token_input: int = None, token_output: int = None,
                  cost_usd: float = None, latency_ms: int = None) -> Optional[str]:
    """Insert a new preview into dmhoa_case_previews table."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_previews"
        headers = supabase_headers()
        headers['Prefer'] = 'return=representation'

        preview_data = {
            'case_id': case_id,
            'preview_content': preview_content,
            'preview_snippet': preview_snippet,
            'prompt_version': prompt_version,
            'model': model,
            'token_input': token_input,
            'token_output': token_output,
            'cost_usd': cost_usd,
            'latency_ms': latency_ms,
            'is_active': True
        }

        # Remove None values
        preview_data = {k: v for k, v in preview_data.items() if v is not None}

        response = requests.post(url, headers=headers, json=preview_data, timeout=TIMEOUT)
        response.raise_for_status()

        result = response.json()
        if result:
            preview_id = result[0]['id']
            logger.info(f"Inserted new preview {preview_id} for case {case_id}")
            return preview_id

        return None

    except Exception as e:
        logger.error(f"Failed to insert preview for case {case_id}: {str(e)}")
        return None


def save_case_preview_to_new_table(case_id: str, preview_text: str, doc_brief: Dict,
                                  token_usage: Dict = None, latency_ms: int = None, preview_json: Optional[Dict] = None) -> bool:
    """Save generated case preview to the new dmhoa_case_previews table."""
    try:
        # Deactivate existing previews first
        deactivate_previews(case_id)

        # Prepare preview content as JSONB with preview_json
        preview_content = {
            'preview_text': preview_text,
            'doc_summary': doc_brief,
            'generated_at': datetime.utcnow().isoformat(),
            'preview_json': preview_json
        }

        # Create preview_snippet from preview_json if available, otherwise fallback
        if preview_json:
            headline = preview_json.get('headline', 'HOA Case Analysis')
            deadline = preview_json.get('your_situation', {}).get('deadline', 'Not stated')
            risks = preview_json.get('risk_if_wrong', [])
            first_risk = risks[0] if risks else 'Various consequences'

            preview_snippet = f"{headline} | Deadline: {deadline} | Risk: {first_risk}"
            # Limit snippet length
            if len(preview_snippet) > 200:
                preview_snippet = preview_snippet[:197] + "..."
        else:
            # Fallback to old method
            preview_snippet = preview_text[:200] + "..." if len(preview_text) > 200 else preview_text

        # Extract token usage if available
        token_input = token_usage.get('prompt_tokens') if token_usage else None
        token_output = token_usage.get('completion_tokens') if token_usage else None
        cost_usd = token_usage.get('cost_usd') if token_usage else None

        # Insert new preview with updated prompt version
        preview_id = insert_preview(
            case_id=case_id,
            preview_content=preview_content,
            preview_snippet=preview_snippet,
            prompt_version="v2.0_sales",  # Updated to indicate new conversion-optimized format
            model="gpt-4o-mini",
            token_input=token_input,
            token_output=token_output,
            cost_usd=cost_usd,
            latency_ms=latency_ms
        )

        return preview_id is not None

    except Exception as e:
        logger.error(f"Failed to save case preview to new table for case {case_id}: {str(e)}")
        return False


def render_preview_markdown(preview_json: Dict) -> str:
    """Convert preview_json into a clean markdown string for existing UI."""
    try:
        markdown_parts = []

        # Headline
        headline = preview_json.get('headline', 'HOA Case Analysis')
        markdown_parts.append(f"# {headline}\n")

        # Why Now section
        why_now = preview_json.get('why_now', '')
        if why_now:
            markdown_parts.append(f"## Urgent Situation\n{why_now}\n")

        # Your Situation
        situation = preview_json.get('your_situation', {})
        if situation:
            markdown_parts.append("## Your Current Situation\n")

            alleged_violation = situation.get('alleged_violation')
            if alleged_violation:
                markdown_parts.append(f"**Alleged Violation:** {alleged_violation}\n")

            deadline = situation.get('deadline', 'Not stated')
            markdown_parts.append(f"**Deadline:** {deadline}\n")

            hoa_demands = situation.get('hoa_demands', [])
            if hoa_demands:
                markdown_parts.append("**HOA Demands:**")
                for demand in hoa_demands:
                    markdown_parts.append(f"- {demand}")
                markdown_parts.append("")

            rules_cited = situation.get('rules_cited', [])
            if rules_cited:
                markdown_parts.append("**Rules/Regulations Cited:**")
                for rule in rules_cited:
                    markdown_parts.append(f"- {rule}")
                markdown_parts.append("")

        # NEW: Critical Detail (Locked) section
        critical_detail = preview_json.get('critical_detail_locked', {})
        if critical_detail:
            title = critical_detail.get('title', 'Critical Detail (Locked)')
            body = critical_detail.get('body', '')
            if body:
                markdown_parts.append(f"## {title}\n{body}\n")

        # Risk if wrong
        risks = preview_json.get('risk_if_wrong', [])
        if risks:
            markdown_parts.append("## Risk If You Handle This Wrong\n")
            for risk in risks:
                markdown_parts.append(f"- {risk}")
            markdown_parts.append("")

        # What you get when you unlock
        unlock_items = preview_json.get('what_you_get_when_you_unlock', [])
        if unlock_items:
            markdown_parts.append("## What You Get When You Unlock Full Response\n")
            for item in unlock_items:
                markdown_parts.append(f"- {item}")
            markdown_parts.append("")

        # Hard stop
        hard_stop = preview_json.get('hard_stop', '')
        if hard_stop:
            markdown_parts.append(f"## Next Steps\n{hard_stop}\n")

        # CTA
        cta = preview_json.get('cta', {})
        if cta:
            primary = cta.get('primary', 'Unlock full response package')
            secondary = cta.get('secondary', 'See exactly what proof the HOA will accept')
            markdown_parts.append(f"**{primary}**\n*{secondary}*\n")

        # Join all parts and limit to ~600 words
        full_markdown = '\n'.join(markdown_parts)

        # Simple word count check - if too long, truncate at reasonable point
        words = full_markdown.split()
        if len(words) > 600:
            truncated_words = words[:600]
            full_markdown = ' '.join(truncated_words) + "..."

        return full_markdown

    except Exception as e:
        logger.error(f"Error rendering preview markdown: {str(e)}")
        return "Error rendering preview. Please try again."


# NEW: Helper function to clean up rules_cited narrative text
def clean_rules_cited(rules_cited_list):
    """Clean up rules_cited to prefer citations over narrative."""
    if not rules_cited_list:
        return rules_cited_list

    cleaned = []
    for rule in rules_cited_list:
        if isinstance(rule, str):
            # Convert narrative "previous notices" to compressed form
            if "previous notices" in rule.lower():
                # Extract year if present, otherwise use generic
                import re
                year_match = re.search(r'\b(20\d{2})\b', rule)
                if year_match:
                    cleaned.append(f"Prior notices ({year_match.group(1)})")
                else:
                    cleaned.append("Prior notices (2024)")
            else:
                cleaned.append(rule)
        else:
            cleaned.append(rule)

    return cleaned

def generate_case_preview_with_openai(case_data: Dict, doc_brief: Dict) -> Tuple[str, Dict, int, Optional[Dict]]:
    """Generate case preview using OpenAI and return preview, token usage, latency, and preview_json."""
    start_time = time.time()

    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
            return "Preview generation unavailable - OpenAI not configured", {}, 0, None

        # Extract case information from payload first, then fallback to top level
        payload = case_data.get('payload', {})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except:
                payload = {}

        # Try payload first, then case_data directly
        hoa_name = (payload.get('hoaName') or
                   payload.get('hoa_name') or
                   case_data.get('hoa_name') or
                   case_data.get('hoaName') or
                   'Unknown HOA')

        violation_type = (payload.get('violationType') or
                         payload.get('violation_type') or
                         payload.get('noticeType') or
                         case_data.get('case_description') or
                         case_data.get('violationType') or
                         'Unknown violation')

        case_description = (payload.get('caseDescription') or
                           payload.get('case_description') or
                           payload.get('description') or
                           case_data.get('case_description') or
                           case_data.get('caseDescription') or
                           'No description provided')

        property_address = (payload.get('propertyAddress') or
                           payload.get('property_address') or
                           case_data.get('property_address') or
                           case_data.get('propertyAddress') or
                           '')

        owner_name = (payload.get('ownerName') or
                     payload.get('owner_name') or
                     case_data.get('owner_name') or
                     case_data.get('ownerName') or
                     '')

        # Get document brief
        doc_text = doc_brief.get('brief_text', '')
        doc_count = doc_brief.get('doc_count', 0)
        doc_status = doc_brief.get('doc_status', 'none')

        # Prepare the conversion-optimized prompt based on document availability
        if doc_status == "ready" and doc_text:
            # NEW: Updated prompt for documents ready - improved headline, why_now, rules_cited, hard_stop, and critical_detail_locked
            user_prompt = f"""Create a conversion-optimized preview for this HOA dispute case. You must extract specific facts from the document analysis and output ONLY valid JSON.

Case Details:
- HOA: {hoa_name}
- Violation Type: {violation_type}
- Property Address: {property_address}
- Owner: {owner_name}
- Case Description: {case_description}

Document Analysis ({doc_count} documents ready):
{doc_text[:3000] if doc_text else 'No document content available'}

Output ONLY valid JSON with this exact structure:
{{
  "version": "preview_v2_sales",
  "headline": "string (8-14 words, specific to this case. If a deadline is present, headline must include it as 'X-Day Deadline' or the exact date)",
  "why_now": "string (1-2 sentences, tie urgency to the required action: inspection + written report + video, and the deadline)",
  "your_situation": {{
    "alleged_violation": "string (extracted from documents)",
    "hoa_demands": ["string", "string", "..."],
    "deadline": "string (use exact date if present; else 'Not stated')",
    "rules_cited": ["string (each entry must be either 'Paragraph <...>' / 'Section <...>' / 'Article <...>' if present, otherwise 'Not stated'. Do NOT include vague narrative like 'previous notices' unless no citations exist; if included, compress to 'Prior notices (YYYY)')", "..."]
  }},
  "critical_detail_locked": {{
    "title": "Critical Response Wording (Locked)",
    "body": "The exact clause/paragraph language to cite, proof checklist (what the HOA will accept: report + video format), and extension request wording (preserves rights without admitting fault) are locked. Our analysis shows specific language about [evidence acceptance/rule interpretation/compliance deadlines] that could weaken your position if worded incorrectly. This critical phrasing is included in the unlock package."
  }},
  "risk_if_wrong": [
    "string (specific consequence)",
    "string",
    "string"
  ],
  "what_you_get_when_you_unlock": [
    "Professional response letter tailored to your specific violation",
    "Certified mail template with proper legal language",
    "Evidence checklist to document your defense",
    "Extension request template if deadline is tight",
    "Negotiation strategies for your specific situation"
  ],
  "hard_stop": "string (1-2 lines that create unfinished business - must mention 2-3 concrete items such as: exact paragraph language to quote, proof checklist, extension request template)",
  "cta": {{
    "primary": "Unlock full response package",
    "secondary": "See exactly what proof the HOA will accept"
  }}
}}

RULES:
- Extract specific facts from document analysis for your_situation fields
- Use "you" voice throughout
- Be concrete and specific to this case
- Include at least 5 concrete deliverables in what_you_get_when_you_unlock
- Make hard_stop create genuine unfinished business with concrete locked items
- For critical_detail_locked body, reference exact clause language, proof checklist, extension request wording
- Avoid 'admit liability' language unless phrased as 'without admitting fault' or 'without admitting liability'"""
        else:
            # NEW: Updated prompt for documents pending - improved hard_stop and critical_detail_locked
            docs_pending_text = "docs are still being processed" if doc_status == "processing" else "no documents uploaded yet"
            user_prompt = f"""Create a conversion-optimized preview for this HOA dispute case. Documents are pending but you must still create a persuasive preview. Output ONLY valid JSON.

Case Details:
- HOA: {hoa_name}
- Violation Type: {violation_type}
- Property Address: {property_address}
- Owner: {owner_name}
- Case Description: {case_description}
- Document Status: {docs_pending_text}

Output ONLY valid JSON with this exact structure:
{{
  "version": "preview_v2_sales",
  "headline": "string (8-14 words, specific to this case type)",
  "why_now": "string (1-2 sentences, mention documents pending but emphasize urgency)",
  "your_situation": {{
    "alleged_violation": "{violation_type} (details pending document analysis)",
    "hoa_demands": ["Pending document analysis"],
    "deadline": "Not stated - will be extracted from documents",
    "rules_cited": ["Pending document analysis"]
  }},
  "critical_detail_locked": {{
    "title": "Critical Response Wording (Locked)",
    "body": "Exact rule language, deadline extraction, proof checklist, and extension request template are being extracted from your documents. The precise response phrasing that avoids admitting liability and preserves your rights will be available after processing. This includes the exact language needed for compliance responses and extension requests."
  }},
  "risk_if_wrong": [
    "Missing critical response deadlines",
    "Accepting invalid HOA demands without challenge",
    "Paying unnecessary fines or agreeing to unreasonable compliance"
  ],
  "what_you_get_when_you_unlock": [
    "Professional response letter once documents are analyzed",
    "Certified mail template with proper legal language",
    "Evidence checklist for your specific violation type",
    "Deadline tracking and extension strategies",
    "Complete document analysis and defense strategy"
  ],
  "hard_stop": "Your documents are being analyzed to identify exact rule language, deadline extraction, proof checklist, and extension request template. Once complete, you'll get these specific locked items needed for your response.",
  "cta": {{
    "primary": "Unlock full response package",
    "secondary": "Get complete analysis once documents are ready"
  }}
}}

RULES:
- Use "you" voice throughout
- Be specific about what the unlock will provide once docs are ready
- Make it clear docs are pending but tool is still valuable
- Create urgency around not missing opportunities
- For critical_detail_locked, emphasize that exact response phrasing comes after document analysis"""

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You write conversion-optimized previews for HOA dispute tool. Output ONLY valid JSON. No explanation, no markdown, just the JSON object."
                },
                {
                    "role": "user",
                    "content": user_prompt
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
        json_response = result['choices'][0]['message']['content'].strip()

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract token usage
        token_usage = {}
        if 'usage' in result:
            usage = result['usage']
            token_usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }

            # Estimate cost (approximate rates for gpt-4o-mini)
            input_cost = (token_usage['prompt_tokens'] / 1000) * 0.00015  # $0.15 per 1K input tokens
            output_cost = (token_usage['completion_tokens'] / 1000) * 0.0006  # $0.60 per 1K output tokens
            token_usage['cost_usd'] = round(input_cost + output_cost, 6)

        # Parse JSON safely and create markdown
        preview_json = None
        try:
            # Clean the response in case there's extra text
            json_start = json_response.find('{')
            json_end = json_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                clean_json = json_response[json_start:json_end]
                preview_json = json.loads(clean_json)

                # NEW: Post-processing to clean up rules_cited narrative
                if preview_json and 'your_situation' in preview_json and 'rules_cited' in preview_json['your_situation']:
                    preview_json['your_situation']['rules_cited'] = clean_rules_cited(preview_json['your_situation']['rules_cited'])

                # Create markdown from JSON
                markdown_preview = render_preview_markdown(preview_json)

                logger.info(f"Generated case preview: {len(markdown_preview)} characters, {latency_ms}ms")
                return markdown_preview, token_usage, latency_ms, preview_json

            else:
                raise json.JSONDecodeError("No valid JSON found", json_response, 0)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            logger.warning(f"Raw response: {json_response}")

            # Fallback to original response as markdown but no preview_json
            markdown_preview = f"# HOA Case Analysis\n\n{json_response}"
            logger.info(f"Generated fallback case preview: {len(markdown_preview)} characters, {latency_ms}ms")
            return markdown_preview, token_usage, latency_ms, None

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Failed to generate case preview: {str(e)}")
        return f"Error generating preview: {str(e)}", {}, latency_ms, None


def generate_preview_without_documents(case_data: Dict) -> Tuple[str, Dict, int, Optional[Dict]]:
    """Generate a conversion-optimized case preview when no documents are available yet."""
    start_time = time.time()

    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
            return "Preview generation unavailable - OpenAI not configured", {}, 0, None

        # Extract case information from payload
        payload = case_data.get('payload', {})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except:
                payload = {}

        hoa_name = payload.get('hoaName', payload.get('hoa_name', 'Unknown HOA'))
        violation_type = payload.get('violationType', payload.get('noticeType', 'Unknown violation'))
        case_description = payload.get('caseDescription', payload.get('case_description', 'No description provided'))
        property_address = payload.get('propertyAddress', payload.get('property_address', ''))
        owner_name = payload.get('ownerName', payload.get('owner_name', ''))

        # NEW: Updated prompt for no documents case - improved hard_stop and critical_detail_locked
        user_prompt = f"""Create a conversion-optimized preview for this HOA dispute case. No documents have been uploaded yet, but you must still create a persuasive preview. Output ONLY valid JSON.

Case Details:
- HOA: {hoa_name}
- Violation Type: {violation_type}
- Property Address: {property_address}
- Owner: {owner_name}
- Case Description: {case_description}
- Document Status: No documents uploaded yet

Output ONLY valid JSON with this exact structure:
{{
  "version": "preview_v2_sales",
  "headline": "string (8-14 words, specific to this {violation_type} case)",
  "why_now": "string (1-2 sentences, create urgency about acting before it's too late)",
  "your_situation": {{
    "alleged_violation": "{violation_type} by {hoa_name}",
    "hoa_demands": ["Upload documents to see specific demands"],
    "deadline": "Unknown - upload documents to identify deadlines",
    "rules_cited": ["Upload documents to see specific rules cited"]
  }},
  "critical_detail_locked": {{
    "title": "Critical Response Wording (Locked)",
    "body": "Exact rule language, deadline extraction, proof checklist, and extension request template will be extracted from your documents. The precise response phrasing that avoids admitting liability and preserves your rights will be available after processing. This includes the exact language needed for compliance responses and extension requests."
  }},
  "risk_if_wrong": [
    "Missing critical response deadlines that could escalate penalties",
    "Accepting invalid HOA demands without proper challenge",
    "Paying unnecessary fines or agreeing to unreasonable compliance"
  ],
  "what_you_get_when_you_unlock": [
    "Professional response letter template for {violation_type} cases",
    "Certified mail template with proper legal language",
    "Evidence checklist for {violation_type} violations",
    "Extension request strategies if deadlines are tight",
    "Complete analysis once you upload your HOA documents"
  ],
  "hard_stop": "Upload your HOA notice and we'll analyze exact rule language, deadline extraction, proof checklist, and extension request template. You'll get these specific locked items needed for your response.",
  "cta": {{
    "primary": "Unlock full response package",
    "secondary": "Upload documents for complete analysis"
  }}
}}

RULES:
- Use "you" voice throughout
- Be specific about {violation_type} violations
- Create urgency around not delaying action
- Make it clear uploading documents unlocks much more value
- For critical_detail_locked, emphasize that exact response phrasing comes after document analysis"""

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You write conversion-optimized previews for HOA dispute tool. Output ONLY valid JSON. No explanation, no markdown, just the JSON object."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        json_response = result['choices'][0]['message']['content'].strip()

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract token usage
        token_usage = {}
        if 'usage' in result:
            usage = result['usage']
            token_usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }

            # Estimate cost (approximate rates for gpt-4o-mini)
            input_cost = (token_usage['prompt_tokens'] / 1000) * 0.00015  # $0.15 per 1K input tokens
            output_cost = (token_usage['completion_tokens'] / 1000) * 0.0006  # $0.60 per 1K output tokens
            token_usage['cost_usd'] = round(input_cost + output_cost, 6)

        # Parse JSON safely and create markdown
        preview_json = None
        try:
            # Clean the response in case there's extra text
            json_start = json_response.find('{')
            json_end = json_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                clean_json = json_response[json_start:json_end]
                preview_json = json.loads(clean_json)

                # NEW: Post-processing to clean up rules_cited narrative
                if preview_json and 'your_situation' in preview_json and 'rules_cited' in preview_json['your_situation']:
                    preview_json['your_situation']['rules_cited'] = clean_rules_cited(preview_json['your_situation']['rules_cited'])

                # Create markdown from JSON
                markdown_preview = render_preview_markdown(preview_json)

                logger.info(f"Generated case preview: {len(markdown_preview)} characters, {latency_ms}ms")
                return markdown_preview, token_usage, latency_ms, preview_json

            else:
                raise json.JSONDecodeError("No valid JSON found", json_response, 0)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            logger.warning(f"Raw response: {json_response}")

            # Fallback to original response as markdown but no preview_json
            markdown_preview = f"# HOA Case Analysis\n\n{json_response}"
            logger.info(f"Generated fallback case preview: {len(markdown_preview)} characters, {latency_ms}ms")
            return markdown_preview, token_usage, latency_ms, None

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Failed to generate case preview: {str(e)}")
        return f"Error generating preview: {str(e)}", {}, latency_ms, None


def auto_generate_case_preview(token: str, case_id: str, force_regenerate: bool = False) -> bool:
    """Generate ONLY final preview when documents are ready or after 30-second timeout."""
    try:
        # Use improved concurrency guard to prevent duplicate active previews
        if not force_regenerate and not upsert_active_preview_lock(case_id):
            return True  # Skip generation, already handled or in progress

        try:
            # Fetch case data first
            case = read_case_by_token(token)
            if not case:
                logger.error(f"Case not found for token {token[:12]}...")
                return False

            # Check document status
            documents = fetch_ready_documents_by_token(token, limit=5)
            all_documents = fetch_any_documents_status_by_token(token)

            has_processing_documents = any(doc.get('status') in ['pending', 'processing'] for doc in all_documents)

            # Check existing preview
            existing_preview = read_active_preview(case_id)

            # NEW: Only generate FINAL preview - wait for documents OR timeout
            should_generate = False
            preview_type = "final"

            if force_regenerate:
                # Forced regeneration (usually after document processing completion)
                should_generate = True
                logger.info(f"Force regenerating FINAL preview for case {token[:12]}...")
            elif not existing_preview:
                # No preview exists - check if we should generate final preview
                if documents:
                    # Documents are ready - generate final preview immediately
                    should_generate = True
                    logger.info(f"Documents ready - generating FINAL preview immediately for case {token[:12]}...")
                elif has_processing_documents:
                    # Documents still processing - wait (don't generate preliminary)
                    logger.info(f"Documents processing for case {token[:12]}... - waiting for FINAL preview")
                    return True  # Return success but don't generate preview yet
                elif not all_documents:
                    # No documents at all - generate final basic preview
                    should_generate = True
                    logger.info(f"No documents uploaded - generating FINAL basic preview for case {token[:12]}...")
                else:
                    # Documents exist but not ready yet - wait
                    logger.info(f"Waiting for documents to be ready for case {token[:12]}...")
                    return True
            else:
                # Preview already exists - check if we need to upgrade
                existing_content = existing_preview.get('preview_content', {})
                existing_doc_summary = existing_content.get('doc_summary', {})
                existing_doc_status = existing_doc_summary.get('doc_status', 'none')

                if documents and existing_doc_status in ['none']:
                    # We have ready documents but existing preview doesn't - upgrade to final
                    should_generate = True
                    preview_type = "final_upgrade"
                    logger.info(f"Upgrading to FINAL preview with documents for case {token[:12]}...")
                else:
                    logger.info(f"FINAL preview already exists for case {token[:12]}... - skipping generation")
                    return True

            if not should_generate:
                return True

            # Generate FINAL preview based on available data
            preview_json = None
            if documents:
                # Documents are ready - generate final preview with document analysis
                logger.info(f"Generating FINAL preview with {len(documents)} ready documents for case {token[:12]}...")
                doc_brief = build_doc_brief(documents)
                preview_text, token_usage, latency_ms, preview_json = generate_case_preview_with_openai(case, doc_brief)
                preview_type = "final_with_docs"
            else:
                # No documents - generate final basic preview
                logger.info(f"Generating FINAL basic preview (no documents) for case {token[:12]}...")
                doc_brief = {
                    "doc_status": "none",
                    "doc_count": 0,
                    "sources": [],
                    "brief_text": "No documents have been uploaded for analysis."
                }
                preview_text, token_usage, latency_ms, preview_json = generate_preview_without_documents(case)
                preview_type = "final_basic"

            # Save final preview (this will deactivate existing ones automatically)
            success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms, preview_json)

            if success:
                logger.info(f"Successfully generated {preview_type} FINAL preview for case {token[:12]}...")
            else:
                logger.error(f"Failed to save {preview_type} FINAL preview for case {token[:12]}")

            return success

        finally:
            # Always release the lock when done
            release_preview_lock(case_id)

    except Exception as e:
        # Release lock on exception
        release_preview_lock(case_id)
        logger.error(f"Error auto-generating FINAL preview for case {token[:12]}...: {str(e)}")
        # Update status to error so it's not stuck at pending
        try:
            error_data = {
                'case_token': token,
                'status': 'error',
                'error': str(e)[:500],  # Truncate long errors
                'updated_at': datetime.utcnow().isoformat()
            }
            upsert_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
            upsert_headers_err = supabase_headers()
            upsert_headers_err['Prefer'] = 'resolution=merge-duplicates'
            requests.post(upsert_url, headers=upsert_headers_err, json=error_data, timeout=TIMEOUT)
        except Exception as db_err:
            logger.error(f"Failed to update error status in DB: {str(db_err)}")

def trigger_case_analysis_after_payment(token: str):
    """
    Trigger case analysis generation in a background thread after payment.
    This creates a record in dmhoa_case_outputs with the full analysis.
    """
    def run_analysis():
        try:
            logger.info(f"Starting case analysis generation for token {token[:8]}... after payment")

            # 1) Ensure case exists and is unlocked/paid
            case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            case_params = {
                'token': f'eq.{token}',
                'select': 'token,unlocked,status,payload'
            }
            case_headers = supabase_headers()

            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None

            if not case_row:
                logger.error(f"Case not found for token {token[:8]}... during post-payment analysis")
                return

            if not case_row.get('unlocked'):
                logger.warning(f"Case {token[:8]}... not unlocked yet during post-payment analysis, skipping")
                return

            # 2) Load extracted documents for this case
            docs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
            docs_params = {
                'token': f'eq.{token}',
                'select': 'id,filename,path,status,extracted_text,page_count,char_count,updated_at,error',
                'order': 'updated_at.desc',
                'limit': '10'
            }
            docs_headers = supabase_headers()

            docs_response = requests.get(docs_url, params=docs_params, headers=docs_headers, timeout=TIMEOUT)
            docs_response.raise_for_status()
            docs = docs_response.json()

            docs_newest = newest_updated_at(docs)

            usable_docs = [
                d for d in docs
                if isinstance(d.get('extracted_text'), str) and d['extracted_text'].strip()
            ]

            logger.info(f"Post-payment analysis: {len(docs)} docs found, {len(usable_docs)} usable for token {token[:8]}...")

            if usable_docs:
                docs_block = '\n'.join([
                    f"DOCUMENT {i + 1}: {d.get('filename') or d.get('path') or d['id']}\n"
                    f"---\n{(d.get('extracted_text', '') or '')[:12000]}\n---\n"
                    for i, d in enumerate(usable_docs[:5])
                ])
            else:
                statuses = ', '.join([d.get('status', 'unknown') for d in docs]) or 'none'
                errors = ' | '.join([d.get('error', '') for d in docs if d.get('error')])[:100] or 'none'
                docs_block = f"No document text available yet.\nDocs found: {len(docs)}\nStatuses: {statuses}\nErrors: {errors}"

            # 3) Check if outputs already exist
            outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
            outputs_params = {
                'case_token': f'eq.{token}',
                'select': 'case_token,status,outputs,error,updated_at'
            }
            outputs_headers = supabase_headers()

            outputs_response = requests.get(outputs_url, params=outputs_params, headers=outputs_headers, timeout=TIMEOUT)
            outputs_response.raise_for_status()
            existing_outputs = outputs_response.json()
            existing_out = existing_outputs[0] if existing_outputs else None

            out_updated = safe_iso(existing_out.get('updated_at') if existing_out else None)
            docs_are_newer = (
                docs_newest and out_updated and
                datetime.fromisoformat(docs_newest) > datetime.fromisoformat(out_updated)
            )

            if (existing_out and existing_out.get('status') == 'ready' and
                existing_out.get('outputs') and not docs_are_newer):
                logger.info(f"Case outputs already exist for token {token[:8]}..., skipping generation")
                return

            # 4) Mark outputs as pending (upsert)
            pending_data = {
                'case_token': token,
                'status': 'pending',
                'error': None,
                'model': 'gpt-4o-mini',
                'prompt_version': 'v3_post_payment_auto',
                'updated_at': datetime.utcnow().isoformat()
            }

            upsert_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
            upsert_headers_local = supabase_headers()
            upsert_headers_local['Prefer'] = 'resolution=merge-duplicates'

            upsert_response = requests.post(upsert_url, headers=upsert_headers_local, json=pending_data, timeout=TIMEOUT)
            upsert_response.raise_for_status()

            payload = case_row.get('payload') or {}
            draft_titles = get_draft_titles(payload)

            # 5) OpenAI API call
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary_html": {"type": "string"},
                    "letter_summary": {"type": "string"},
                    "draft_titles": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "clarification": {"type": "string"},
                            "extension": {"type": "string"},
                            "compliance": {"type": "string"}
                        },
                        "required": ["clarification", "extension", "compliance"]
                    },
                    "risks_and_deadlines": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "deadlines": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "risks": {"type": "array", "items": {"type": "string"}, "minItems": 3}
                        },
                        "required": ["deadlines", "risks"]
                    },
                    "action_plan": {"type": "array", "items": {"type": "string"}, "minItems": 6},
                    "drafts": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "clarification": {"type": "string"},
                            "extension": {"type": "string"},
                            "compliance": {"type": "string"}
                        },
                        "required": ["clarification", "extension", "compliance"]
                    },
                    "questions_to_ask": {"type": "array", "items": {"type": "string"}, "minItems": 6},
                    "lowest_cost_path": {"type": "array", "items": {"type": "string"}, "minItems": 4}
                },
                "required": [
                    "summary_html", "letter_summary", "draft_titles", "risks_and_deadlines",
                    "action_plan", "drafts", "questions_to_ask", "lowest_cost_path"
                ]
            }

            doc_fingerprint = {
                'count': len(docs),
                'usableCount': len(usable_docs),
                'newestUpdatedAt': docs_newest,
                'ids': [d['id'] for d in docs],
                'statuses': [d.get('status') for d in docs],
                'charCounts': [d.get('char_count') for d in docs]
            }

            system_content = """You generate HOA dispute assistance for a homeowner. This is educational drafting help, not legal advice.
OUTPUT RULES: ONLY summary_html may contain HTML using <div>, <strong>, <ul>, <li>. ALL drafts MUST be PLAIN TEXT ONLY with newlines.
DRAFT QUALITY: Each draft must be complete, ready-to-send. Include Subject line, opening, 3-6 bullet requests, timeline, closing.
DEPTH: action_plan >= 6 steps, risks >= 3, questions_to_ask >= 6, lowest_cost_path >= 4.
STYLE: Calm, professional, firm, factual."""

            user_content = f"""Case payload JSON:
{json.dumps(payload)}

Document fingerprint:
{json.dumps(doc_fingerprint)}

Extracted documents:
{docs_block}

Draft types:
- drafts.clarification: "{draft_titles['clarification']}"
- drafts.extension: "{draft_titles['extension']}"
- drafts.compliance: "{draft_titles['compliance']}"

Make this feel like a $30 deliverable: concrete, specific, complete."""

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            openai_payload = {
                "model": "gpt-4o-mini",
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "dmhoa_case_outputs",
                        "strict": True,
                        "schema": schema
                    }
                }
            }

            openai_response = requests.post(
                'https://api.openai.com/v1/responses',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json=openai_payload,
                timeout=(10, 120)
            )

            if not openai_response.ok:
                error_text = openai_response.text
                logger.error(f'OpenAI call failed: {openai_response.status_code}, {error_text}')
                error_data = {
                    'case_token': token,
                    'status': 'error',
                    'error': error_text or 'OpenAI call failed',
                    'updated_at': datetime.utcnow().isoformat()
                }
                requests.post(upsert_url, headers=upsert_headers_local, json=error_data, timeout=TIMEOUT)
                return

            openai_json = openai_response.json()
            structured = extract_structured_result(openai_json)

            if structured:
                outputs_to_store = {
                    **structured,
                    'draft_titles': structured.get('draft_titles', draft_titles),
                    'doc_fingerprint': doc_fingerprint
                }
            else:
                outputs_to_store = {
                    'raw': openai_json,
                    'draft_titles': draft_titles,
                    'doc_fingerprint': doc_fingerprint
                }

            success_data = {
                'case_token': token,
                'status': 'ready',
                'outputs': outputs_to_store,
                'error': None,
                'model': 'gpt-4o-mini',
                'prompt_version': 'v3_post_payment_auto',
                'updated_at': datetime.utcnow().isoformat()
            }

            requests.post(upsert_url, headers=upsert_headers_local, json=success_data, timeout=TIMEOUT)
            logger.info(f"Successfully generated case analysis for token {token[:8]}... after payment")

        except Exception as e:
            logger.error(f"Error in post-payment case analysis for token {token[:8]}...: {str(e)}")
            # Update status to error so it's not stuck at pending
            try:
                error_data = {
                    'case_token': token,
                    'status': 'error',
                    'error': str(e)[:500],  # Truncate long errors
                    'updated_at': datetime.utcnow().isoformat()
                }
                upsert_url_err = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
                upsert_headers_err = supabase_headers()
                upsert_headers_err['Prefer'] = 'resolution=merge-duplicates'
                requests.post(upsert_url_err, headers=upsert_headers_err, json=error_data, timeout=TIMEOUT)
                logger.info(f"Updated case outputs status to 'error' for token {token[:8]}...")
            except Exception as db_err:
                logger.error(f"Failed to update error status in DB: {str(db_err)}")

    # Run in background thread
    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.daemon = True
    analysis_thread.start()
    logger.info(f"Started background case analysis thread for token {token[:8]}...")


@app.route('/webhooks/stripe', methods=['POST'])
def stripe_webhook():
    """Stripe webhook handler for processing payment events (converted from Deno/TypeScript)"""
    try:
        # Environment variables validation
        if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET or not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            logger.error("Missing required environment variables", extra={
                'hasStripeKey': bool(STRIPE_SECRET_KEY),
                'hasWebhookSecret': bool(STRIPE_WEBHOOK_SECRET),
                'hasSupabaseUrl': bool(SUPABASE_URL),
                'hasServiceRole': bool(SUPABASE_SERVICE_ROLE_KEY),
            })
            return jsonify({'error': 'Missing environment variables'}), 500

        # Get raw body and signature
        payload = request.get_data()
        sig_header = request.headers.get('stripe-signature')

        if not sig_header:
            return jsonify({'error': 'No signature'}), 400

        # Verify webhook signature
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            logger.error("Invalid payload")
            return jsonify({'error': 'Invalid payload'}), 400
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid signature")
            return jsonify({'error': 'Invalid signature'}), 400

        logger.info(f"Webhook event type: {event['type']}")

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']

            # Get token from client_reference_id or metadata (check both token and case_token)
            token = (
                session.get('client_reference_id') or
                session.get('metadata', {}).get('token') or
                session.get('metadata', {}).get('case_token')
            )
            if not token:
                logger.error(f"No token found in session. Session data: {session.get('metadata', {})}")
                return jsonify({'error': 'No token in session'}), 400

            # Get email (prefer customer_details.email)
            email = (
                session.get('customer_details', {}).get('email') or
                session.get('customer_email') or
                None
            )

            logger.info(f"Processing payment completion for token: {token}")

            # Update case to unlocked
            case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            case_params = {'token': f'eq.{token}'}
            case_data = {
                'unlocked': True,
                'status': 'paid',
                'stripe_checkout_session_id': session['id'],
                'stripe_payment_intent_id': session.get('payment_intent'),
                'amount_total': session.get('amount_total'),
                'currency': session.get('currency'),
                'updated_at': datetime.utcnow().isoformat()
            }
            case_headers = supabase_headers()
            case_headers['Prefer'] = 'return=representation'

            try:
                case_response = requests.patch(case_url, params=case_params, headers=case_headers,
                                             json=case_data, timeout=TIMEOUT)
                case_response.raise_for_status()
                updated_cases = case_response.json()

                if not updated_cases:
                    logger.error(f"Case not found for token: {token}")
                    return jsonify({'error': 'Case not found'}), 404

                updated_case = updated_cases[0]
                logger.info(f"Successfully updated case: {updated_case.get('id')}")

                # Fallback email from DB if Stripe didn't provide it
                if not email:
                    email = updated_case.get('email')

            except Exception as e:
                logger.error(f"Failed to update case: {str(e)}")
                return jsonify({'error': 'Database update failed'}), 500

            # --- Send receipt email (non-fatal) ---
            if not email:
                logger.warning("No email available (Stripe + DB). Skipping receipt email send.")
            elif not SMTP_SENDER_WEBHOOK_URL or not SMTP_SENDER_WEBHOOK_SECRET:
                logger.warning("SMTP sender webhook env vars missing; skipping email send")
            else:
                case_url_link = f"{SITE_URL}/case.html?case={token}"
                email_payload = {
                    'token': token,
                    'email': email,
                    'case_url': case_url_link,
                    'amount_total': session.get('amount_total'),
                    'currency': session.get('currency'),
                    'customer_name': session.get('customer_details', {}).get('name'),
                    'stripe_session_id': session['id']
                }

                try:
                    email_response = requests.post(
                        SMTP_SENDER_WEBHOOK_URL,
                        headers={
                            'Content-Type': 'application/json',
                            'X-Webhook-Secret': SMTP_SENDER_WEBHOOK_SECRET
                        },
                        json=email_payload,
                        timeout=TIMEOUT
                    )

                    if not email_response.ok:
                        error_text = email_response.text
                        logger.warning(f"Receipt email send failed (non-fatal): {email_response.status_code}, {error_text}")

                        # Log failed email event
                        try:
                            event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
                            event_data = {
                                'token': token,
                                'type': 'receipt_email_failed',
                                'data': {
                                    'status': email_response.status_code,
                                    'body': error_text[:1000]
                                }
                            }
                            event_headers = supabase_headers()
                            requests.post(event_url, headers=event_headers, json=event_data, timeout=TIMEOUT)
                        except Exception:
                            pass  # Best effort
                    else:
                        # Log successful email event
                        try:
                            event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
                            event_data = {
                                'token': token,
                                'type': 'receipt_email_sent',
                                'data': {
                                    'to': email,
                                    'case_url': case_url_link
                                }
                            }
                            event_headers = supabase_headers()
                            requests.post(event_url, headers=event_headers, json=event_data, timeout=TIMEOUT)
                        except Exception:
                            pass  # Best effort

                except Exception as e:
                    logger.warning(f"Receipt email send threw (non-fatal): {str(e)}")
                    # Log error event
                    try:
                        event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
                        event_data = {
                            'token': token,
                            'type': 'receipt_email_failed',
                            'data': {
                                'error': str(e)[:1000]
                            }
                        }
                        event_headers = supabase_headers()
                        requests.post(event_url, headers=event_headers, json=event_data, timeout=TIMEOUT)
                    except Exception:
                        pass  # Best effort

            # Log payment completion event (also non-fatal)
            try:
                event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
                event_data = {
                    'token': token,
                    'type': 'payment_completed',
                    'data': {
                        'session_id': session['id'],
                        'payment_intent': session.get('payment_intent'),
                        'amount_total': session.get('amount_total'),
                        'currency': session.get('currency'),
                        'customer_email': session.get('customer_email'),
                        'payment_status': session.get('payment_status')
                    }
                }
                event_headers = supabase_headers()
                requests.post(event_url, headers=event_headers, json=event_data, timeout=TIMEOUT)
            except Exception as e:
                logger.warning(f"Failed to log payment completion event: {str(e)}")

            # Trigger case analysis generation in background thread
            try:
                trigger_case_analysis_after_payment(token)
                logger.info(f"Triggered case analysis for token: {token}")
            except Exception as e:
                logger.warning(f"Failed to trigger case analysis (non-fatal): {str(e)}")

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        logger.error(f"Error handling Stripe webhook: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
