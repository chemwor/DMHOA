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
CORS(app)  # Enable CORS for all routes

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
                                  token_usage: Dict = None, latency_ms: int = None) -> bool:
    """Save generated case preview to the new dmhoa_case_previews table."""
    try:
        # Deactivate existing previews first
        deactivate_previews(case_id)

        # Prepare preview content as JSONB
        preview_content = {
            'preview_text': preview_text,
            'doc_summary': doc_brief,
            'generated_at': datetime.utcnow().isoformat()
        }

        # Extract first paragraph or 200 chars as snippet
        preview_snippet = preview_text[:200] + "..." if len(preview_text) > 200 else preview_text

        # Extract token usage if available
        token_input = token_usage.get('prompt_tokens') if token_usage else None
        token_output = token_usage.get('completion_tokens') if token_usage else None
        cost_usd = token_usage.get('cost_usd') if token_usage else None

        # Insert new preview
        preview_id = insert_preview(
            case_id=case_id,
            preview_content=preview_content,
            preview_snippet=preview_snippet,
            prompt_version="v1.0",
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


def generate_case_preview_with_openai(case_data: Dict, doc_brief: Dict) -> Tuple[str, Dict, int]:
    """Generate case preview using OpenAI and return preview, token usage, and latency."""
    start_time = time.time()

    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
            return "Preview generation unavailable - OpenAI not configured", {}, 0

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

        logger.info(f"Generated case preview: {len(preview)} characters, {latency_ms}ms")
        return preview, token_usage, latency_ms

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Failed to generate case preview: {str(e)}")
        return f"Error generating preview: {str(e)}", {}, latency_ms


def generate_preview_without_documents(case_data: Dict) -> Tuple[str, Dict, int]:
    """Generate a basic case preview when no documents are available yet."""
    start_time = time.time()

    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
            return "Preview generation unavailable - OpenAI not configured", {}, 0

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

        # Create a basic prompt without documents
        prompt = f"""Generate a preliminary case preview for this HOA dispute case (documents still being processed):

Case Details:
- HOA: {hoa_name}
- Violation Type: {violation_type}
- Property Address: {property_address}
- Owner: {owner_name}
- Case Description: {case_description}
- Document Status: Documents are still being processed and analyzed

Create a preliminary case preview that includes:
1. Case overview based on provided information
2. General guidance for this type of HOA violation
3. Common legal considerations for similar cases
4. Recommended next steps
5. Note that detailed analysis will be available once documents are processed

Keep it professional, factual, and under 600 words. Mention that this is a preliminary preview pending document analysis."""

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a legal case analyst specializing in HOA disputes. Provide professional, factual analysis even with limited information."
                },
                {
                    "role": "user",
                    "content": prompt
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
        preview = result['choices'][0]['message']['content'].strip()

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

        logger.info(f"Generated preliminary case preview: {len(preview)} characters, {latency_ms}ms")
        return preview, token_usage, latency_ms

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Failed to generate preliminary case preview: {str(e)}")
        return f"Error generating preliminary preview: {str(e)}", {}, latency_ms


def auto_generate_case_preview(token: str, case_id: str) -> bool:
    """Automatically generate case preview - immediate or deferred based on document status."""
    try:
        # Check if preview already exists
        existing_preview = read_active_preview(case_id)
        if existing_preview:
            logger.info(f"Preview already exists for case {token[:12]}... - skipping generation")
            return True

        # Fetch case data
        case = read_case_by_token(token)
        if not case:
            logger.error(f"Case not found for token {token[:12]}...")
            return False

        # Check document status
        documents = fetch_ready_documents_by_token(token, limit=5)
        all_documents = fetch_any_documents_status_by_token(token)

        has_processing_documents = any(doc.get('status') in ['pending', 'processing'] for doc in all_documents)

        if documents:
            # Documents are ready - generate full preview with document analysis
            logger.info(f"Generating full preview with {len(documents)} ready documents for case {token[:12]}...")
            doc_brief = build_doc_brief(documents)
            preview_text, token_usage, latency_ms = generate_case_preview_with_openai(case, doc_brief)

            # Save to new table
            success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms)

            if success:
                logger.info(f"Successfully generated full preview for case {token[:12]}...")
            else:
                logger.error(f"Failed to save full preview for case {token[:12]}...")

            return success

        elif has_processing_documents:
            # Documents are still processing - generate preliminary preview
            logger.info(f"Documents still processing for case {token[:12]}... - generating preliminary preview")
            doc_brief = {
                "doc_status": "processing",
                "doc_count": len(all_documents),
                "sources": [],
                "brief_text": "Documents are currently being processed and analyzed."
            }

            preview_text, token_usage, latency_ms = generate_preview_without_documents(case)

            # Save preliminary preview (will be updated when documents are ready)
            success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms)

            if success:
                logger.info(f"Successfully generated preliminary preview for case {token[:12]}...")
            else:
                logger.error(f"Failed to save preliminary preview for case {token[:12]}...")

            return success

        else:
            # No documents at all - generate basic preview
            logger.info(f"No documents found for case {token[:12]}... - generating basic preview")
            doc_brief = {
                "doc_status": "none",
                "doc_count": 0,
                "sources": [],
                "brief_text": "No documents have been uploaded for analysis."
            }

            preview_text, token_usage, latency_ms = generate_preview_without_documents(case)

            # Save basic preview
            success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms)

            if success:
                logger.info(f"Successfully generated basic preview for case {token[:12]}...")
            else:
                logger.error(f"Failed to save basic preview for case {token[:12]}...")

            return success

    except Exception as e:
        logger.error(f"Error auto-generating preview for case {token[:12]}...: {str(e)}")
        return False


def trigger_preview_update_after_document_processing(token: str) -> bool:
    """Trigger preview update when documents become ready (called from doc-extract webhook)."""
    try:
        # Find the case
        case = read_case_by_token(token)
        if not case:
            logger.warning(f"Case not found for document processing completion: {token[:12]}...")
            return False

        case_id = case.get('id')
        if not case_id:
            logger.warning(f"Case ID not found for token: {token[:12]}...")
            return False

        # Check if we have ready documents now
        documents = fetch_ready_documents_by_token(token, limit=5)
        if not documents:
            logger.info(f"No ready documents yet for case {token[:12]}... - keeping existing preview")
            return True

        # Check if we already have a full preview (not preliminary)
        existing_preview = read_active_preview(case_id)
        if existing_preview:
            preview_content = existing_preview.get('preview_content', {})
            doc_summary = preview_content.get('doc_summary', {})
            if doc_summary.get('doc_status') == 'ready' and doc_summary.get('doc_count', 0) > 0:
                logger.info(f"Full preview already exists for case {token[:12]}... - skipping update")
                return True

        # Generate updated preview with documents
        logger.info(f"Updating preview with newly ready documents for case {token[:12]}...")

        # Deactivate existing preliminary preview
        deactivate_previews(case_id)

        # Generate new full preview
        doc_brief = build_doc_brief(documents)
        preview_text, token_usage, latency_ms = generate_case_preview_with_openai(case, doc_brief)

        # Save updated preview
        success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms)

        if success:
            logger.info(f"Successfully updated preview with documents for case {token[:12]}...")
        else:
            logger.error(f"Failed to update preview with documents for case {token[:12]}...")

        return success

    except Exception as e:
        logger.error(f"Error updating preview after document processing for {token[:12]}...: {str(e)}")
        return False

@app.route('/webhooks/generate-preview', methods=['POST'])
def generate_preview_webhook():
    """Webhook endpoint to generate case preview - can be called after case creation."""
    # Validate webhook secret
    webhook_secret = request.headers.get('X-Webhook-Secret')
    if not webhook_secret or webhook_secret != DOC_EXTRACT_WEBHOOK_SECRET:
        logger.warning("Invalid or missing webhook secret for generate-preview")
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Parse JSON body
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        # Validate required fields
        token = data.get('token')
        case_id = data.get('case_id')

        if not token:
            return jsonify({'error': 'Missing required field: token'}), 400

        logger.info(f"Generating preview for case - Token: {token[:8]}..., Case ID: {case_id}")

        # If case_id not provided, look it up by token
        if not case_id:
            case = read_case_by_token(token)
            if not case:
                return jsonify({'error': 'Case not found for token'}), 404
            case_id = case.get('id')
            if not case_id:
                return jsonify({'error': 'Case ID not found'}), 404

        # Generate preview
        success = auto_generate_case_preview(token, case_id)

        if success:
            return jsonify({
                'message': 'Preview generated successfully',
                'token': token,
                'case_id': case_id
            }), 200
        else:
            return jsonify({
                'error': 'Failed to generate preview',
                'token': token,
                'case_id': case_id
            }), 500

    except Exception as e:
        error_msg = f"Unexpected error generating preview: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


def schedule_delayed_preview_generation(token: str, case_id: str, delay_seconds: int = 30):
    """Schedule preview generation with a delay to allow documents to be uploaded and processed."""
    def delayed_preview():
        time.sleep(delay_seconds)
        try:
            logger.info(f"Executing delayed preview generation for case {token[:8]}...")
            success = auto_generate_case_preview(token, case_id)
            if success:
                logger.info(f"Delayed preview generation successful for case {token[:8]}...")
            else:
                logger.warning(f"Delayed preview generation failed for case {token[:8]}...")
        except Exception as e:
            logger.error(f"Error in delayed preview generation for case {token[:8]}...: {str(e)}")

    # Start the delayed task in a background thread
    thread = threading.Thread(target=delayed_preview, daemon=True)
    thread.start()
    logger.info(f"Scheduled delayed preview generation for case {token[:8]}... in {delay_seconds} seconds")


@app.route('/webhooks/case-created', methods=['POST'])
def case_created_webhook():
    """Webhook to handle case creation and trigger initial preview generation."""
    # Validate webhook secret
    webhook_secret = request.headers.get('X-Webhook-Secret')
    if not webhook_secret or webhook_secret != DOC_EXTRACT_WEBHOOK_SECRET:
        logger.warning("Invalid or missing webhook secret for case-created")
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Parse JSON body
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        token = data.get('token')
        case_id = data.get('case_id')

        if not token:
            return jsonify({'error': 'Missing required field: token'}), 400

        logger.info(f"Case created - Token: {token[:8]}..., Case ID: {case_id}")

        # If case_id not provided, look it up by token
        if not case_id:
            case = read_case_by_token(token)
            if not case:
                return jsonify({'error': 'Case not found for token'}), 404
            case_id = case.get('id')

        # Schedule both immediate and delayed preview generation
        # Immediate: Generate basic preview now
        immediate_success = auto_generate_case_preview(token, case_id)

        # Delayed: Give time for documents to be uploaded and processed, then regenerate
        schedule_delayed_preview_generation(token, case_id, delay_seconds=60)

        # Also schedule a longer delay for cases where document processing might take longer
        schedule_delayed_preview_generation(token, case_id, delay_seconds=300)  # 5 minutes

        return jsonify({
            'message': 'Case creation handled and preview generation scheduled',
            'token': token,
            'case_id': case_id,
            'immediate_preview': immediate_success
        }), 200

    except Exception as e:
        error_msg = f"Unexpected error handling case creation: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


def create_case_in_supabase(case_data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """Create a new case in Supabase database."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        headers = supabase_headers()
        headers['Prefer'] = 'return=representation'

        # Prepare case data
        case_payload = {
            'token': case_data.get('token'),
            'hoa_name': case_data.get('hoaName', ''),
            'violation_type': case_data.get('violationType', ''),
            'case_description': case_data.get('caseDescription', ''),
            'owner_name': case_data.get('ownerName', ''),
            'owner_email': case_data.get('ownerEmail', ''),
            'property_address': case_data.get('propertyAddress', ''),
            'payload': json.dumps(case_data),  # Store full payload as JSON
            'status': 'active'
        }

        # Remove None/empty values
        case_payload = {k: v for k, v in case_payload.items() if v}

        response = requests.post(url, headers=headers, json=case_payload, timeout=TIMEOUT)
        response.raise_for_status()

        result = response.json()
        if result and len(result) > 0:
            case_id = result[0].get('id')
            token = result[0].get('token')
            logger.info(f"Created case successfully - ID: {case_id}, Token: {token[:8]}...")
            return True, case_id, token
        else:
            logger.error("No case data returned from Supabase")
            return False, None, None

    except Exception as e:
        logger.error(f"Failed to create case in Supabase: {str(e)}")
        return False, None, None


@app.route('/api/save-case', methods=['POST', 'OPTIONS'])
def save_case():
    """Save a new case and trigger preview generation."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        # Parse JSON body
        case_data = request.get_json()
        if not case_data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        # Validate required fields
        required_fields = ['token', 'hoaName', 'violationType']
        missing_fields = [field for field in required_fields if not case_data.get(field)]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        token = case_data['token']
        logger.info(f"Saving new case - Token: {token[:8]}...")

        # Create case in database
        success, case_id, returned_token = create_case_in_supabase(case_data)

        if not success:
            return jsonify({
                'error': 'Failed to create case in database'
            }), 500

        # Trigger case creation webhook to start preview generation
        try:
            if case_id:
                # Call our own case-created webhook to trigger preview generation
                webhook_data = {
                    'token': token,
                    'case_id': case_id
                }

                # Make internal webhook call
                webhook_headers = {
                    'X-Webhook-Secret': DOC_EXTRACT_WEBHOOK_SECRET,
                    'Content-Type': 'application/json'
                }

                # Use localhost for internal calls
                webhook_url = f"http://localhost:{os.environ.get('PORT', 5000)}/webhooks/case-created"

                webhook_response = requests.post(
                    webhook_url,
                    headers=webhook_headers,
                    json=webhook_data,
                    timeout=5
                )

                if webhook_response.status_code == 200:
                    logger.info(f"Successfully triggered preview generation for case {token[:8]}...")
                else:
                    logger.warning(f"Failed to trigger preview generation: {webhook_response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to trigger preview generation webhook: {str(e)}")
            # Don't fail the main request if preview generation fails

        response_data = {
            'success': True,
            'message': 'Case saved successfully',
            'token': token,
            'case_id': case_id
        }

        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

        return response, 200

    except Exception as e:
        error_msg = f"Unexpected error saving case: {str(e)}"
        logger.error(error_msg)

        response = jsonify({'error': error_msg})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

        return response, 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
