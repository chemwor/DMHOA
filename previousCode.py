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
        logger.info(f"Raw payload data - filename: {repr(filename)}, mime_type: {repr(mime_type)}, path: {path}")

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
            logger.info(f"Processing as image using OCR: {filename}")
            extracted_text, page_count, char_count, extraction_error = extract_image_text(file_bytes, filename)
        else:
            error_msg = f"Unsupported file type: filename='{filename}', mime_type='{mime_type}', path='{path}'. Supported formats: PDF, JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP"
            logger.warning(error_msg)
            update_document(document_id, token, {
                'status': 'failed',
                'error': error_msg[:2000]
            })
            return jsonify({
                'error': error_msg,
                'document_id': document_id
            }, 400)

        if extraction_error:
            # Handle extraction failure
            update_document(document_id, token, {
                'status': 'failed',
                'error': extraction_error[:2000],
                'page_count': page_count
            })
            return jsonify({
                'error': extraction_error,
                'document_id': document_id
            }), 500

        if char_count == 0:
            # Handle no text found case
            error_msg = "No text found in document - document may be blank or contain no readable text"
            update_document(document_id, token, {
                'status': 'failed',
                'error': error_msg,
                'page_count': page_count,
                'char_count': 0
            })
            return jsonify({
                'error': error_msg,
                'document_id': document_id
            }), 500

        # Update document with extracted text
        success = update_document(document_id, token, {
            'status': 'ready',
            'extracted_text': extracted_text,
            'page_count': page_count,
            'char_count': char_count,
            'error': None
        })

        if not success:
            return jsonify({
                'error': 'Failed to update document with extracted text',
                'document_id': document_id
            }), 500

        file_type = "PDF" if is_pdf_file(filename, mime_type) else "image"
        logger.info(f"Successfully processed {file_type} document {document_id} - {page_count} pages, {char_count} characters")

        return jsonify({
            'message': f'{file_type} document processed successfully',
            'document_id': document_id,
            'status': 'ready',
            'page_count': page_count,
            'char_count': char_count,
            'file_type': file_type.lower()
        }), 200

    except Exception as e:
        error_msg = str(e)[:2000]  # Truncate error message
        logger.error(f"Unexpected error processing document: {error_msg}")

        # Try to update document status to failed if we have the required data
        if document_id and token:
            update_document(document_id, token, {
                'status': 'failed',
                'error': error_msg
            })
            return jsonify({
                'error': error_msg,
                'document_id': document_id
            }), 500
        else:
            return jsonify({'error': error_msg}), 500

@app.route("/webhooks/send-receipt-email", methods=["POST"])
def send_receipt_email():
    secret = request.headers.get("X-Webhook-Secret")
    logger.info("Received send-receipt-email webhook")

    if not secret or secret != SMTP_SENDER_WEBHOOK_SECRET:
        logger.warning("Unauthorized send-receipt-email attempt: missing or invalid webhook secret")
        return jsonify({"error": "Unauthorized"}), 401

    # Check which SMTP environment variables are missing so we can diagnose quickly
    required = {
        'SMTP_HOST': SMTP_HOST,
        'SMTP_PORT': SMTP_PORT,
        'SMTP_USER': SMTP_USER,
        'SMTP_PASS': SMTP_PASS,
        'SMTP_SENDER_WEBHOOK_SECRET': SMTP_SENDER_WEBHOOK_SECRET,
    }
    missing = [name for name, val in required.items() if not val]
    if missing:
        # Log the missing variable names (not their values) for diagnostics
        logger.error(f"SMTP not configured - missing environment variables: {missing}")
        return jsonify({"error": "SMTP not configured", "missing": missing}), 500

    data = request.get_json() or {}
    token = data.get("token")
    to_email = data.get("email")
    case_url = data.get("case_url")
    amount_total = data.get("amount_total")
    currency = (data.get("currency") or "usd").upper()
    customer_name = data.get("customer_name")
    stripe_session_id = data.get("stripe_session_id")

    logger.info(f"send-receipt-email payload: token_present={bool(token)}, to_email={to_email}, case_url_present={bool(case_url)}, customer_name={customer_name}")

    if not token or not to_email or not case_url:
        logger.warning("send-receipt-email missing required fields")
        return jsonify({"error": "Missing token/email/case_url"}), 400

    # Personalized greeting
    greeting = f"Hi {customer_name}," if customer_name else "Hi,"

    # Format payment amount
    dollars = f"${(amount_total or 0)/100:.2f}" if isinstance(amount_total, int) else ""
    payment_info = f" for {dollars} {currency}" if dollars else ""

    # Enhanced email content
    subject = "Payment Confirmed - Your Dispute My HOA Case is Ready"

    text = f"""{greeting}

Thank you for your payment{payment_info}! Your payment has been successfully processed and your Dispute My HOA case is now unlocked and ready for access.

🔓 ACCESS YOUR CASE:
{case_url}

📧 IMPORTANT: Please save this email for your records. You can use the link above to access your case anytime in the future.

💡 WHAT'S NEXT:
• Review your case documents and analysis
• Use the AI-powered insights to understand your dispute
• Access legal templates and guidance specific to your situation
• Get step-by-step instructions for resolving your HOA dispute

📞 NEED HELP?
If you have any questions or need assistance accessing your case, simply reply to this email and we'll get back to you promptly.

Your case token for reference: {token[:8]}...
{f'Transaction ID: {stripe_session_id}' if stripe_session_id else ''}

Best regards,
The Dispute My HOA Team
https://disputemyhoa.com

---
This email confirms your payment and provides access to your case. Keep this email safe for future reference."""

    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(text, "plain"))

    try:
        logger.info(f"Connecting to SMTP {SMTP_HOST}:{SMTP_PORT} to send to {to_email}")
        logger.info(f"SMTP_USER after cleaning: '{SMTP_USER}' (length: {len(SMTP_USER)})")
        logger.info(f"SMTP_PASS after cleaning: length={len(SMTP_PASS)}, starts_with='{SMTP_PASS[:4]}...'")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())

        logger.info(f"Receipt email sent to {to_email} (token={token[:8]}..." )
        return jsonify({"ok": True}), 200

    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Gmail authentication failed. For Gmail, you need: 1) Enable 2-Step Verification, 2) Create App Password. Error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"ok": False, "error": error_msg}), 500
    except Exception as e:
        # Log full exception with traceback to help diagnose mail failures
        logger.exception("Failed to send receipt email")
        return jsonify({"ok": False, "error": str(e)}), 500


def preview_env(name: str, value: str) -> Dict[str, Any]:
    """Helper: safely preview env values without leaking secrets"""
    if not value:
        return {'name': name, 'present': False, 'preview': None, 'length': 0}

    lower = name.lower()
    is_secret = (
        'secret' in lower or
        'service_role' in lower or
        'key' in lower
    )

    preview = f"{value[:6]}…({len(value)})" if is_secret else f"{value[:24]}{'…' if len(value) > 24 else ''}"

    return {'name': name, 'present': True, 'preview': preview, 'length': len(value)}

def extract_structured_result(responses_json: Any) -> Optional[Dict[str, Any]]:
    """Safer JSON extraction from OpenAI Responses API payload"""
    output = responses_json.get('output') if responses_json else None

    if isinstance(output, list) and len(output) > 0:
        for item in output:
            content = item.get('content') if item else None

            if isinstance(content, list):
                # Look for output_json type first
                cj = next((c for c in content
                          if c and c.get('type') == 'output_json' and c.get('json')), None)
                if not cj:
                    # Fallback to any content with json
                    cj = next((c for c in content
                              if c and c.get('json') and isinstance(c['json'], dict)), None)

                if cj and cj.get('json') and isinstance(cj['json'], dict):
                    return cj['json']

                # Look for text content to parse as JSON
                ct = next((c for c in content
                          if c and c.get('type') == 'output_text' and isinstance(c.get('text'), str)), None)
                if not ct:
                    ct = next((c for c in content
                              if c and isinstance(c.get('text'), str)), None)

                if ct and ct.get('text'):
                    try:
                        parsed = json.loads(ct['text'])
                        if parsed and isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass  # ignore parse errors

    return None


def get_draft_titles(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Keeps DB keys stable (drafts.clarification/extension/compliance),
    but changes what those "slots" mean based on the user's selection.
    """
    outcome = str(payload.get('outcome', '')).lower()

    titles = {
        'clarification': 'Request Clarification / Rule Citation',
        'extension': 'Request Extension / Pause Enforcement',
        'compliance': 'Confirm Compliance Plan'
    }

    if outcome == 'clarification':
        titles = {
            'clarification': 'Request Clarification / Rule Citation',
            'extension': 'Request Extension While Clarifying',
            'compliance': 'Confirm Compliance Plan (If Needed)'
        }
    elif outcome == 'extension':
        titles = {
            'clarification': 'Request Clarification + Confirm Requirements',
            'extension': 'Request Extension / New Deadline',
            'compliance': 'Confirm Compliance Plan + Timeline'
        }
    elif outcome == 'alternative':
        titles = {
            'clarification': 'Request Approved Options / Standards',
            'extension': 'Request Temporary Variance / Extra Time',
            'compliance': 'Propose Alternative Remedy Plan'
        }
    elif outcome == 'comply':
        titles = {
            'clarification': 'Confirm Requirements Before Starting Work',
            'extension': 'Request Extra Time to Complete Work',
            'compliance': 'Confirm Compliance Completion'
        }
    elif outcome == 'dispute':
        titles = {
            'clarification': 'Formal Dispute / Appeal Letter',
            'extension': 'Request Hearing Extension / Reschedule',
            'compliance': 'Evidence Submission Cover Letter'
        }
    elif outcome == 'not-sure':
        titles = {
            'clarification': 'Request Clarification / Rule Citation',
            'extension': 'Request Extension to Evaluate Options',
            'compliance': 'Provisional Compliance Plan (If Required)'
        }

    return titles


def safe_iso(s: Any) -> Optional[str]:
    """Convert value to ISO string safely"""
    v = str(s or '').strip()
    if not v:
        return None
    try:
        d = datetime.fromisoformat(v.replace('Z', '+00:00'))
        return d.isoformat()
    except Exception:
        try:
            d = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return d.isoformat()
        except Exception:
            return None


def newest_updated_at(docs: List[Dict[str, Any]]) -> Optional[str]:
    """Find the newest updated_at timestamp from documents"""
    newest = None
    for d in docs:
        iso = safe_iso(d.get('updated_at'))
        if not iso:
            continue
        if not newest or datetime.fromisoformat(iso) > datetime.fromisoformat(newest):
            newest = iso
    return newest


@app.route('/api/create-checkout-session', methods=['POST', 'OPTIONS'])
def create_checkout_session():
    """Create Stripe checkout session endpoint (converted from Supabase edge function)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    logger.info('[create-checkout-session] request', extra={
        'method': request.method,
        'url': request.url,
        'has_auth_header': bool(request.headers.get('authorization')),
        'has_apikey_header': bool(request.headers.get('apikey')),
        'origin': request.headers.get('origin')
    })

    try:
        # Environment variables validation
        env_report = [
            preview_env('STRIPE_SECRET_KEY', STRIPE_SECRET_KEY or ''),
            preview_env('STRIPE_PRICE_ID', STRIPE_PRICE_ID or ''),
            preview_env('SITE_URL', SITE_URL),
            preview_env('SUPABASE_URL', SUPABASE_URL or ''),
            preview_env('SUPABASE_SERVICE_ROLE_KEY', SUPABASE_SERVICE_ROLE_KEY or ''),
        ]
        logger.info('[create-checkout-session] env report', extra={'env_report': env_report})

        missing = [v['name'] for v in env_report if not v['present']]
        if missing:
            logger.error('[create-checkout-session] missing env vars', extra={'missing': missing})
            response = jsonify({'error': 'Missing required environment variables', 'missing': missing})
            return add_cors_headers(response), 500

        logger.info('[create-checkout-session] initializing clients')

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = body.get('token')
        email = body.get('email')
        payload = body.get('payload')

        logger.info('[create-checkout-session] parsed body', extra={
            'has_token': bool(token),
            'token_preview': f"{token[:12]}…" if isinstance(token, str) else None,
            'has_email': bool(email),
            'email_domain': email.split('@')[1] if isinstance(email, str) and '@' in email else None,
            'has_payload': bool(payload)
        })

        if not token or not email:
            response = jsonify({'error': 'Token and email are required'})
            return add_cors_headers(response), 400

        # Validate email format
        email_regex = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
        if not email_regex.match(email):
            response = jsonify({'error': 'Invalid email format'})
            return add_cors_headers(response), 400

        # 1) Fetch case (it may not exist yet — that's OK)
        logger.info('[create-checkout-session] fetching case', extra={'token_preview': f"{token[:12]}…"})

        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {
            'token': f'eq.{token}',
            'select': 'id,token,status,stripe_checkout_session_id'
        }
        headers = supabase_headers()

        try:
            response_data = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            response_data.raise_for_status()
            cases = response_data.json()
            existing_case = cases[0] if cases else None
        except Exception as e:
            logger.error('[create-checkout-session] database fetch error', extra={
                'message': str(e)
            })
            response = jsonify({'error': 'Database error'})
            return add_cors_headers(response), 500

        # 2) Create case if missing
        if not existing_case:
            logger.warning('[create-checkout-session] case not found, creating new case',
                         extra={'token_preview': f"{token[:12]}…"})

            insert_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            insert_data = {
                'token': token,
                'email': email,
                'status': 'pending_payment',
                'unlocked': False,
                'payload': payload,
                'updated_at': datetime.utcnow().isoformat()
            }
            insert_headers = supabase_headers()
            insert_headers['Prefer'] = 'return=representation'

            try:
                insert_response = requests.post(insert_url, headers=insert_headers,
                                              json=insert_data, timeout=TIMEOUT)
                insert_response.raise_for_status()
            except Exception as e:
                logger.error('[create-checkout-session] failed to create case', extra={
                    'message': str(e)
                })
                response = jsonify({'error': 'Failed to create case'})
                return add_cors_headers(response), 500
        else:
            logger.info('[create-checkout-session] case found', extra={
                'status': existing_case.get('status'),
                'has_existing_session': bool(existing_case.get('stripe_checkout_session_id'))
            })

            # Update existing case
            logger.info('[create-checkout-session] updating case status -> pending_payment')
            update_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            update_params = {'token': f'eq.{token}'}
            update_data = {
                'email': email,
                'payload': payload,
                'status': 'pending_payment',
                'updated_at': datetime.utcnow().isoformat()
            }
            update_headers = supabase_headers()

            try:
                update_response = requests.patch(update_url, params=update_params,
                                               headers=update_headers, json=update_data, timeout=TIMEOUT)
                update_response.raise_for_status()
            except Exception as e:
                logger.error('[create-checkout-session] database update error', extra={
                    'message': str(e)
                })
                response = jsonify({'error': 'Failed to update case'})
                return add_cors_headers(response), 500

        # 3) Create Stripe Checkout Session
        logger.info('[create-checkout-session] creating stripe checkout session', extra={
            'price_id': STRIPE_PRICE_ID,
            'site_url': SITE_URL,
            'expires_in_minutes': 30
        })

        try:
            session = stripe.checkout.Session.create(
                mode='payment',
                line_items=[{'price': STRIPE_PRICE_ID, 'quantity': 1}],
                success_url=f"{SITE_URL}/case.html?case={token}&session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{SITE_URL}/case-preview.html?case={token}",
                client_reference_id=token,
                customer_email=email,
                metadata={'token': token, 'source': 'dispute-my-hoa'},
                expires_at=int(datetime.utcnow().timestamp()) + (30 * 60)  # 30 minutes
            )
        except Exception as e:
            logger.error('[create-checkout-session] stripe session creation failed', extra={
                'message': str(e)
            })
            response = jsonify({'error': f'Stripe error: {str(e)}'})
            return add_cors_headers(response), 500

        logger.info('[create-checkout-session] stripe session created', extra={
            'session_id': session.id,
            'has_url': bool(session.url),
            'amount_total': session.amount_total,
            'currency': session.currency
        })

        # 4) Save session id on the case
        save_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        save_params = {'token': f'eq.{token}'}
        save_data = {
            'stripe_checkout_session_id': session.id,
            'updated_at': datetime.utcnow().isoformat()
        }
        save_headers = supabase_headers()

        try:
            save_response = requests.patch(save_url, params=save_params,
                                         headers=save_headers, json=save_data, timeout=TIMEOUT)
            save_response.raise_for_status()
        except Exception as e:
            logger.warning('[create-checkout-session] failed saving stripe session id (non-fatal)', extra={
                'message': str(e)
            })

        # 5) Log event (non-fatal)
        event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
        event_data = {
            'token': token,
            'type': 'checkout_session_created',
            'data': {
                'session_id': session.id,
                'email_domain': email.split('@')[1] if '@' in email else None,
                'amount': session.amount_total,
                'currency': session.currency
            }
        }
        event_headers = supabase_headers()

        try:
            event_response = requests.post(event_url, headers=event_headers,
                                         json=event_data, timeout=TIMEOUT)
            event_response.raise_for_status()
        except Exception as e:
            logger.warning('[create-checkout-session] failed to insert dmhoa_events (non-fatal)', extra={
                'message': str(e)
            })

        response = jsonify({'url': session.url})
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error('[create-checkout-session] error', extra={
            'message': str(e),
            'name': type(e).__name__
        })
        response = jsonify({'error': str(e) or 'Internal server error'})
        return add_cors_headers(response), 500


@app.route('/api/store-message', methods=['POST', 'OPTIONS'])
def store_message():
    """Store chat message endpoint (converted from Supabase edge function)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        origin = request.headers.get('Origin')
        allowed_origins = [
            'https://disputemyhoa.com',
            'https://dmhoadev.netlify.app',
            'http://localhost:5173',
            'http://localhost:3000',
            'http://127.0.0.1:5173'
        ]
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
        else:
            response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        origin = request.headers.get('Origin')
        allowed_origins = [
            'https://disputemyhoa.com',
            'https://dmhoadev.netlify.app',
            'http://localhost:5173',
            'http://localhost:3000',
            'http://127.0.0.1:5173'
        ]
        if origin in allowed_origins:
            response.headers.add('Access-Control-Allow-Origin', origin)
        else:
            response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    try:
        # Validate environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing env vars'})
            return add_cors_headers(response), 500

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = (body.get('token') or '').strip()
        role = (body.get('role') or '').strip()
        content = (body.get('content') or '').strip()

        if not token or not role or not content:
            response = jsonify({'error': 'token, role, content are required'})
            return add_cors_headers(response), 400

        if role not in ['user', 'assistant', 'system']:
            response = jsonify({'error': 'invalid role'})
            return add_cors_headers(response), 400

        # Check if case is unlocked
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'unlocked'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None
        except Exception as e:
            logger.error('Failed to fetch case for unlock check', extra={'error': str(e)})
            response = jsonify({'error': 'DB fetch failed', 'details': str(e)})
            return add_cors_headers(response), 500

        if not case_row or not case_row.get('unlocked'):
            response = jsonify({'error': 'Case is not unlocked'})
            return add_cors_headers(response), 402

        # Insert message into dmhoa_messages
        message_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
        message_data = {
            'token': token,
            'role': role,
            'content': content
        }
        message_headers = supabase_headers()
        message_headers['Prefer'] = 'return=representation'

        try:
            message_response = requests.post(message_url, headers=message_headers,
                                           json=message_data, timeout=TIMEOUT)
            message_response.raise_for_status()
            inserted_message = message_response.json()
        except Exception as e:
            logger.error('Failed to insert message', extra={'error': str(e)})
            response = jsonify({'error': 'DB insert failed', 'details': str(e)})
            return add_cors_headers(response), 500

        # Return the inserted message data
        message_data = inserted_message[0] if inserted_message else {}
        response = jsonify({'ok': True, 'message': message_data})
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error('store-message error', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


@app.route('/api/doc-extract-start', methods=['POST', 'OPTIONS'])
def doc_extract_start():
    """Document extraction trigger endpoint (converted from Supabase edge function)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type, x-doc-secret')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type, x-doc-secret')
        return response

    def safe_trim(v) -> str:
        return str(v or '').strip()

    logger.info(f"[{datetime.utcnow().isoformat()}] {request.method} {request.url}")

    if request.method != 'POST':
        response = jsonify({'error': 'Method not allowed'})
        return add_cors_headers(response), 405

    try:
        # Environment variables check
        WEBHOOK_URL = os.environ.get('DOC_EXTRACT_WEBHOOK_URL', f"{request.url_root.rstrip('/')}/webhooks/doc-extract")
        DOC_BUCKET = os.environ.get('DOC_EXTRACT_BUCKET', 'dmhoa-docs')

        logger.info('Env check', extra={
            'hasUrl': bool(SUPABASE_URL),
            'hasServiceRole': bool(SUPABASE_SERVICE_ROLE_KEY),
            'hasWebhookUrl': bool(WEBHOOK_URL),
            'hasWebhookSecret': bool(DOC_EXTRACT_WEBHOOK_SECRET),
            'bucket': DOC_BUCKET
        })

        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing SUPABASE env vars'})
            return add_cors_headers(response), 500

        if not DOC_EXTRACT_WEBHOOK_SECRET:
            response = jsonify({'error': 'Missing webhook env vars'})
            return add_cors_headers(response), 500

        # Validate secret header (optional - commented out for now like in original)
        # incoming_secret = request.headers.get('x-doc-secret', '').strip()
        # if not incoming_secret or incoming_secret != 'dmhoa_9baf6a13e2f847d0b52f':
        #     logger.error('Unauthorized - secret mismatch')
        #     response = jsonify({'error': 'Unauthorized'})
        #     return add_cors_headers(response), 401

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = safe_trim(body.get('token'))
        storage_path = safe_trim(body.get('storage_path'))
        filename = safe_trim(body.get('filename')) or None
        mime_type = safe_trim(body.get('mime_type')) or None

        if not token or not storage_path:
            response = jsonify({'error': 'token and storage_path are required'})
            return add_cors_headers(response), 400

        # Ensure case exists (fail fast)
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'id,token,payload,created_at'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None
        except Exception as e:
            logger.error('Case lookup error', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading case', 'details': str(e)})
            return add_cors_headers(response), 500

        if not case_row:
            response = jsonify({'error': 'Case not found', 'token': token})
            return add_cors_headers(response), 404

        # 1) Create (or reuse) a dmhoa_documents row for this file
        document_id = None

        # Check for existing document
        doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        doc_params = {
            'token': f'eq.{token}',
            'path': f'eq.{storage_path}',
            'select': 'id,status'
        }
        doc_headers = supabase_headers()

        try:
            doc_response = requests.get(doc_url, params=doc_params, headers=doc_headers, timeout=TIMEOUT)
            doc_response.raise_for_status()
            docs = doc_response.json()
            existing_doc = docs[0] if docs else None
        except Exception as e:
            logger.error('Doc lookup error', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading document', 'details': str(e)})
            return add_cors_headers(response), 500

        if existing_doc and existing_doc.get('id'):
            document_id = existing_doc['id']
            logger.info('Reusing existing dmhoa_documents row', extra={
                'document_id': document_id,
                'status': existing_doc.get('status')
            })
        else:
            # Insert new document
            insert_doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
            insert_doc_data = {
                'token': token,
                'bucket': DOC_BUCKET,
                'path': storage_path,
                'filename': filename,
                'mime_type': mime_type,
                'status': 'pending'
            }
            insert_doc_headers = supabase_headers()
            insert_doc_headers['Prefer'] = 'return=representation'

            try:
                insert_doc_response = requests.post(insert_doc_url, headers=insert_doc_headers,
                                                  json=insert_doc_data, timeout=TIMEOUT)
                insert_doc_response.raise_for_status()
                inserted_docs = insert_doc_response.json()
                document_id = inserted_docs[0]['id'] if inserted_docs else None
                logger.info('Created dmhoa_documents row', extra={'document_id': document_id})
            except Exception as e:
                logger.error('Doc insert error', extra={'error': str(e)})
                response = jsonify({'error': 'Failed to create dmhoa_documents row', 'details': str(e)})
                return add_cors_headers(response), 500

        if not document_id:
            response = jsonify({'error': 'Could not determine document_id'})
            return add_cors_headers(response), 500

        # 2) Update case payload summary status (optional, but useful for UI)
        current_payload = {}
        try:
            payload_data = case_row.get('payload')
            if isinstance(payload_data, str):
                current_payload = json.loads(payload_data)
            elif isinstance(payload_data, dict):
                current_payload = payload_data
        except Exception:
            current_payload = {}

        next_payload = {
            **current_payload,
            'extract_status': 'triggered',
            'notice_storage_path': storage_path,
            'notice_filename': filename,
            'notice_mime_type': mime_type,
            'extract_triggered_at': datetime.utcnow().isoformat(),
            'document_id': document_id
        }

        # Update case payload
        update_case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        update_case_params = {'token': f'eq.{token}'}
        update_case_data = {'payload': next_payload}
        update_case_headers = supabase_headers()

        try:
            update_case_response = requests.patch(update_case_url, params=update_case_params,
                                                headers=update_case_headers, json=update_case_data, timeout=TIMEOUT)
            update_case_response.raise_for_status()
        except Exception as e:
            logger.error('Case payload update error', extra={'error': str(e)})
            # Not fatal—doc record exists and webhook can still run

        # 3) Mark document processing BEFORE webhook
        mark_proc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        mark_proc_params = {'id': f'eq.{document_id}'}
        mark_proc_data = {'status': 'processing', 'error': None}
        mark_proc_headers = supabase_headers()

        try:
            mark_proc_response = requests.patch(mark_proc_url, params=mark_proc_params,
                                              headers=mark_proc_headers, json=mark_proc_data, timeout=TIMEOUT)
            mark_proc_response.raise_for_status()
        except Exception as e:
            logger.error('Failed to mark document processing', extra={'error': str(e)})
            # Not fatal, but you'll want to know

        # 4) Call backend webhook (server-to-server)
        logger.info('Calling webhook', extra={'webhook_url': WEBHOOK_URL})

        webhook_payload = {
            'token': token,
            'document_id': document_id,
            'bucket': DOC_BUCKET,
            'path': storage_path,
            'filename': filename,
            'mime_type': mime_type,
            'supabase_url': SUPABASE_URL  # optional
        }

        webhook_headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Secret': WEBHOOK_SECRET,
            'X-Doc-Extract-Secret': WEBHOOK_SECRET  # Alternative header name
        }

        try:
            webhook_response = requests.post(WEBHOOK_URL, headers=webhook_headers,
                                           json=webhook_payload, timeout=(10, 120))  # Longer timeout for processing

            logger.info('Webhook response', extra={'status': webhook_response.status_code, 'ok': webhook_response.ok})

            if not webhook_response.ok:
                error_text = webhook_response.text
                logger.error('Webhook failed', extra={'error': error_text})

                # Update document status to failed
                fail_doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
                fail_doc_params = {'id': f'eq.{document_id}'}
                fail_doc_data = {
                    'status': 'failed',
                    'error': f'Webhook {webhook_response.status_code}: {error_text}'[:1500]
                }
                fail_doc_headers = supabase_headers()

                try:
                    requests.patch(fail_doc_url, params=fail_doc_params,
                                 headers=fail_doc_headers, json=fail_doc_data, timeout=TIMEOUT)
                except Exception:
                    pass  # Best effort

                # Also reflect summary status on the case payload (optional)
                fail_payload = {
                    **next_payload,
                    'extract_status': 'failed',
                    'extract_error': f'Webhook {webhook_response.status_code}: {error_text}'[:1500],
                    'extract_failed_at': datetime.utcnow().isoformat()
                }

                try:
                    requests.patch(update_case_url, params=update_case_params,
                                 headers=update_case_headers, json={'payload': fail_payload}, timeout=TIMEOUT)
                except Exception:
                    pass  # Best effort

                response = jsonify({
                    'error': 'Webhook call failed',
                    'status': webhook_response.status_code,
                    'details': error_text
                })
                return add_cors_headers(response), 502

            # If webhook returns JSON, capture it (optional)
            try:
                webhook_json = webhook_response.json()
            except Exception:
                webhook_json = {}

            # Mark queued/accepted (document is now in backend pipeline)
            queue_doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
            queue_doc_params = {'id': f'eq.{document_id}'}
            queue_doc_data = {'status': 'processing'}
            queue_doc_headers = supabase_headers()

            try:
                requests.patch(queue_doc_url, params=queue_doc_params,
                             headers=queue_doc_headers, json=queue_doc_data, timeout=TIMEOUT)
            except Exception:
                pass  # Best effort

            ok_payload = {
                **next_payload,
                'extract_status': 'queued',
                'webhook_response': webhook_json,
                'extract_queued_at': datetime.utcnow().isoformat()
            }

            try:
                requests.patch(update_case_url, params=update_case_params,
                             headers=update_case_headers, json={'payload': ok_payload}, timeout=TIMEOUT)
            except Exception:
                pass  # Best effort

            response = jsonify({
                'ok': True,
                'token': token,
                'document_id': document_id,
                'bucket': DOC_BUCKET,
                'path': storage_path,
                'webhook': webhook_json
            })
            return add_cors_headers(response), 200

        except Exception as e:
            logger.error('Webhook request failed', extra={'error': str(e)})
            response = jsonify({'error': f'Webhook request failed: {str(e)}'})
            return add_cors_headers(response), 502

    except Exception as e:
        logger.error('Unexpected error in doc-extract-start', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


@app.route('/api/case-analysis', methods=['POST', 'OPTIONS'])
def case_analysis():
    """Generate HOA case analysis using OpenAI (converted from Deno/TypeScript code)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    try:
        # Validate environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
            response = jsonify({'error': 'Missing env vars'})
            return add_cors_headers(response), 500

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = body.get('token')

        if not token:
            response = jsonify({'error': 'token is required'})
            return add_cors_headers(response), 400

        # 1) Ensure case exists + is unlocked/paid
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'token,unlocked,status,payload'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None
        except Exception as e:
            logger.error('Database error reading case', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading case', 'details': str(e)})
            return add_cors_headers(response), 500

        if not case_row:
            response = jsonify({'error': 'Case not found'})
            return add_cors_headers(response), 404

        if not case_row.get('unlocked'):
            response = jsonify({'error': 'Case is not unlocked'})
            return add_cors_headers(response), 402

        # 1b) Load extracted documents for this case
        docs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        docs_params = {
            'token': f'eq.{token}',
            'select': 'id,filename,path,status,extracted_text,page_count,char_count,updated_at,error',
            'order': 'updated_at.desc',
            'limit': '10'
        }
        docs_headers = supabase_headers()

        try:
            docs_response = requests.get(docs_url, params=docs_params, headers=docs_headers, timeout=TIMEOUT)
            docs_response.raise_for_status()
            docs = docs_response.json()
        except Exception as e:
            logger.error('Docs lookup error', extra={'error': str(e)})
            docs = []

        docs_newest = newest_updated_at(docs)

        # Consider usable if it has text, even if status still says "processing"
        usable_docs = [
            d for d in docs
            if isinstance(d.get('extracted_text'), str) and d['extracted_text'].strip()
        ]

        logger.info('DOCS DEBUG', extra={
            'count': len(docs),
            'docs_newest': docs_newest,
            'statuses': [{'id': d['id'], 'status': d['status'], 'hasText': bool(d.get('extracted_text', '').strip()),
                         'charCount': d.get('char_count'), 'updated_at': d.get('updated_at'), 'err': d.get('error')}
                        for d in docs],
            'usable_count': len(usable_docs)
        })

        if usable_docs:
            docs_block = '\n'.join([
                f"DOCUMENT {i + 1}: {d.get('filename') or d.get('path') or d['id']}\n"
                f"---\n{(d.get('extracted_text', '') or '')[:12000]}\n---\n"
                for i, d in enumerate(usable_docs[:5])
            ])
        else:
            statuses = ', '.join([d.get('status', 'unknown') for d in docs]) or 'none'
            errors = ' | '.join([d.get('error', '') for d in docs if d.get('error')])[:100] or 'none'
            docs_block = f"""No document text available yet.
Docs found: {len(docs)}
Statuses: {statuses}
Errors: {errors}"""

        logger.info('DOCS BLOCK LENGTH', extra={'length': len(docs_block)})

        # 2) If outputs already exist and are ready, return cached ONLY if docs haven't changed since outputs updated_at
        outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        outputs_params = {
            'case_token': f'eq.{token}',
            'select': 'case_token,status,outputs,error,updated_at'
        }
        outputs_headers = supabase_headers()

        try:
            outputs_response = requests.get(outputs_url, params=outputs_params, headers=outputs_headers, timeout=TIMEOUT)
            outputs_response.raise_for_status()
            existing_outputs = outputs_response.json()
            existing_out = existing_outputs[0] if existing_outputs else None
        except Exception as e:
            logger.error('Database error reading outputs', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading outputs', 'details': str(e)})
            return add_cors_headers(response), 500

        out_updated = safe_iso(existing_out.get('updated_at') if existing_out else None)
        docs_are_newer = (
            docs_newest and out_updated and
            datetime.fromisoformat(docs_newest) > datetime.fromisoformat(out_updated)
        )

        if (existing_out and existing_out.get('status') == 'ready' and
            existing_out.get('outputs') and not docs_are_newer):
            response = jsonify({
                'ok': True,
                'status': 'ready',
                'cached': True,
                'outputs': existing_out['outputs']
            })
            return add_cors_headers(response), 200

        # 3) Mark outputs as pending (upsert)
        pending_data = {
            'case_token': token,
            'status': 'pending',
            'error': None,
            'model': 'gpt-4o-mini',
            'prompt_version': 'v3_docs_cache_invalidation',
            'updated_at': datetime.utcnow().isoformat()
        }

        upsert_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        upsert_headers = supabase_headers()
        upsert_headers['Prefer'] = 'resolution=merge-duplicates'

        try:
            upsert_response = requests.post(upsert_url, headers=upsert_headers, json=pending_data, timeout=TIMEOUT)
            upsert_response.raise_for_status()
        except Exception as e:
            logger.error('Failed to mark outputs pending', extra={'error': str(e)})
            response = jsonify({'error': 'Failed to mark outputs pending', 'details': str(e)})
            return add_cors_headers(response), 500

        payload = case_row.get('payload') or {}
        draft_titles = get_draft_titles(payload)

        # 4) OpenAI Responses API call (strict JSON schema)
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

        messages = [
            {
                "role": "system",
                "content": """
You generate HOA dispute assistance for a homeowner.
This is educational drafting help, not legal advice.

OUTPUT RULES (CRITICAL):
- ONLY "summary_html" may contain HTML.
- summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.
- ALL drafts (clarification/extension/compliance) MUST be PLAIN TEXT ONLY:
  - NO HTML tags
  - Use newlines with \\n
  - Bullets: use "- item" lines
- Return STRICT JSON that matches the schema exactly.

DRAFT QUALITY REQUIREMENTS:
- Each draft must be a complete, ready-to-send letter.
- MUST directly quote or reference concrete facts from the extracted documents when available
  (deadlines, email addresses, paragraph citations, dollar amounts, dates, etc.).
- Each must include:
  - Subject line
  - Short opening
  - 3–6 bullet-point requests (specific asks)
  - Proposed timeline (e.g., "Please respond within 10 business days" if no deadline is provided)
  - Request fines/penalties be paused/waived while pending (when relevant)
  - Closing requesting confirmation in writing

DEPTH REQUIREMENTS:
- action_plan >= 6 steps with timing hints (Today / 48 hours / Before deadline).
- risks >= 3 concrete risks tied to HOA enforcement.
- questions_to_ask >= 6 questions.
- lowest_cost_path >= 4 items.

STYLE:
- Calm, professional, firm, factual.
"""
            },
            {
                "role": "user",
                "content": (
                    f"Case payload JSON:\n{json.dumps(payload)}\n\n"
                    f"Document fingerprint (debug):\n{json.dumps(doc_fingerprint)}\n\n"
                    f"Extracted documents:\n{docs_block}\n\n"
                    f"Draft types for this case (MUST follow exactly):\n"
                    f"- drafts.clarification MUST be: \"{draft_titles['clarification']}\"\n"
                    f"- drafts.extension MUST be: \"{draft_titles['extension']}\"\n"
                    f"- drafts.compliance MUST be: \"{draft_titles['compliance']}\"\n\n"
                    f"Also include draft_titles using these exact same strings.\n\n"
                    f"summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.\n"
                    f"Drafts must be PLAIN TEXT ONLY with \\n, and must NOT include any HTML tags.\n\n"
                    f"Make this feel like a $30 deliverable: concrete, specific, complete.\n"
                )
            }
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

        # Make OpenAI API call
        try:
            openai_response = requests.post(
                'https://api.openai.com/v1/responses',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json=openai_payload,
                timeout=(10, 120)  # 10s connect, 120s read
            )

            if not openai_response.ok:
                error_text = openai_response.text
                logger.error('OpenAI call failed', extra={'status': openai_response.status_code, 'error': error_text})

                # Update outputs table with error
                error_data = {
                    'case_token': token,
                    'status': 'error',
                    'error': error_text or 'OpenAI call failed',
                    'updated_at': datetime.utcnow().isoformat()
                }
                try:
                    requests.post(upsert_url, headers=upsert_headers, json=error_data, timeout=TIMEOUT)
                except Exception:
                    pass  # Best effort

                response = jsonify({'error': 'OpenAI call failed', 'details': error_text})
                return add_cors_headers(response), 500

            openai_json = openai_response.json()
            structured = extract_structured_result(openai_json)

            if structured:
                outputs_to_store = {
                    **structured,
                    'draft_titles': structured.get('draft_titles', draft_titles),
                    'doc_fingerprint': doc_fingerprint  # helpful for debugging what it saw
                }
            else:
                outputs_to_store = {
                    'raw': openai_json,
                    'draft_titles': draft_titles,
                    'doc_fingerprint': doc_fingerprint
                }

            # Save successful outputs
            success_data = {
                'case_token': token,
                'status': 'ready',
                'outputs': outputs_to_store,
                'error': None,
                'model': 'gpt-4o-mini',
                'prompt_version': 'v3_docs_cache_invalidation',
                'updated_at': datetime.utcnow().isoformat()
            }

            try:
                requests.post(upsert_url, headers=upsert_headers, json=success_data, timeout=TIMEOUT)
            except Exception as e:
                logger.error('Failed saving outputs', extra={'error': str(e)})
                response = jsonify({'error': 'Failed saving outputs', 'details': str(e)})
                return add_cors_headers(response), 500

            # Update case updated_at timestamp
            try:
                case_update_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                case_update_params = {'token': f'eq.{token}'}
                case_update_data = {'updated_at': datetime.utcnow().isoformat()}
                case_update_headers = supabase_headers()
                requests.patch(case_update_url, params=case_update_params,
                             headers=case_update_headers, json=case_update_data, timeout=TIMEOUT)
            except Exception:
                pass  # Best effort

            response = jsonify({
                'ok': True,
                'status': 'ready',
                'cached': False,
                'outputs': outputs_to_store
            })
            return add_cors_headers(response), 200

        except Exception as e:
            logger.error('OpenAI API error', extra={'error': str(e)})
            response = jsonify({'error': 'OpenAI API error', 'details': str(e)})
            return add_cors_headers(response), 500

    except Exception as e:
        logger.error('case-analysis error', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


@app.route('/api/case-data', methods=['GET', 'OPTIONS'])
def get_case_data():
    """Get case data by token (converted from Deno/TypeScript code)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    try:
        # Validate environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing env vars'})
            return add_cors_headers(response), 500

        # Get token from query parameters
        token = request.args.get('token', '').strip()

        if not token:
            response = jsonify({'error': 'token is required'})
            return add_cors_headers(response), 400

        # Fetch case data from Supabase
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'token,unlocked,status,created_at,payload,amount_total,currency'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()

            if not cases:
                response = jsonify({'error': 'Case not found'})
                return add_cors_headers(response), 404

            case_data = cases[0]

        except requests.exceptions.RequestException as e:
            logger.error('Database error reading case', extra={'error': str(e)})
            response = jsonify({'error': 'Database error'})
            return add_cors_headers(response), 500

        # Return case data
        response = jsonify(case_data)
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error('get-case-data error', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


@app.route('/api/read-messages', methods=['GET', 'OPTIONS'])
def read_messages():
    """Read chat messages for a case (converted from Deno/TypeScript code)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', 'https://disputemyhoa.com')  # or "*" while testing
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        response.headers.add('Access-Control-Max-Age', '86400')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', 'https://disputemyhoa.com')  # or "*" while testing
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        response.headers.add('Access-Control-Max-Age', '86400')
        return response

    try:
        # Validate environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing env vars'})
            return add_cors_headers(response), 500

        # Get parameters from query string
        token = request.args.get('token', '').strip()
        limit_param = request.args.get('limit', '50')

        # Parse and validate limit (min 1, max 200, default 50)
        try:
            limit = max(1, min(int(limit_param) or 50, 200))
        except (ValueError, TypeError):
            limit = 50

        if not token:
            response = jsonify({'error': 'token is required'})
            return add_cors_headers(response), 400

        # Check if case exists and is unlocked
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'unlocked'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None
        except Exception as e:
            logger.error('Database error reading case', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading case'})
            return add_cors_headers(response), 500

        if not case_row:
            response = jsonify({'error': 'Case not found'})
            return add_cors_headers(response), 404

        if not case_row.get('unlocked'):
            # Return empty messages array if case is not unlocked
            response = jsonify({'ok': True, 'messages': []})
            return add_cors_headers(response), 200

        # Fetch messages for the case
        messages_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
        messages_params = {
            'token': f'eq.{token}',
            'select': 'id,token,role,content,created_at',
            'order': 'created_at.asc',
            'limit': str(limit)
        }
        messages_headers = supabase_headers()

        try:
            messages_response = requests.get(messages_url, params=messages_params, headers=messages_headers, timeout=TIMEOUT)
            messages_response.raise_for_status()
            messages = messages_response.json()
        except Exception as e:
            logger.error('Database error reading messages', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading messages'})
            return add_cors_headers(response), 500

        # Return messages
        response = jsonify({'ok': True, 'messages': messages or []})
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error('read-messages error', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


@app.route('/api/read-outputs', methods=['GET', 'POST', 'OPTIONS'])
def read_outputs():
    """Read case analysis outputs by token (converted from Deno/TypeScript code)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    try:
        # Validate environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing required environment variables'})
            return add_cors_headers(response), 500

        # Extract token from GET params or POST body
        token = ""
        if request.method == 'GET':
            token = request.args.get('token', '').strip()
        elif request.method == 'POST':
            try:
                body = request.get_json() or {}
                token = (body.get('token') or '').strip()
            except Exception:
                token = ""
        else:
            response = jsonify({'error': 'Method not allowed'})
            return add_cors_headers(response), 405

        if not token:
            response = jsonify({'error': 'Token is required'})
            return add_cors_headers(response), 400

        # Fetch case outputs from Supabase
        outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        outputs_params = {
            'case_token': f'eq.{token}',
            'select': 'case_token,status,outputs,error,created_at,updated_at,model,prompt_version'
        }
        outputs_headers = supabase_headers()

        try:
            outputs_response = requests.get(outputs_url, params=outputs_params, headers=outputs_headers, timeout=TIMEOUT)
            outputs_response.raise_for_status()
            outputs_data = outputs_response.json()
            data = outputs_data[0] if outputs_data else None
        except Exception as e:
            logger.error('[read-outputs] DB error', extra={'error': str(e)})
            response = jsonify({'error': 'Database error', 'details': str(e)})
            return add_cors_headers(response), 500

        if not data:
            # Return 200 with missing status so client can poll without treating as error
            response = jsonify({
                'case_token': token,
                'status': 'missing',
                'outputs': None,
                'error': None
            })
            return add_cors_headers(response), 200

        # Return the outputs data
        response = jsonify({
            'case_token': data.get('case_token'),
            'status': data.get('status'),
            'outputs': data.get('outputs'),
            'error': data.get('error'),
            'model': data.get('model'),
            'prompt_version': data.get('prompt_version'),
            'created_at': data.get('created_at'),
            'updated_at': data.get('updated_at')
        })
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error('[read-outputs] Error', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'Internal server error'})
        return add_cors_headers(response), 500


def trigger_document_extraction_async(token: str, payload: dict):
    """
    Async function to trigger document extraction without blocking the save operation
    """
    try:
        logger.info(f"Checking if document extraction is needed for token: {token}")

        # Check if there are uploaded documents that need processing
        needs_extraction = (
            (payload.get('pastedText') or (payload.get('additional_docs') and len(payload.get('additional_docs', [])) > 0)) and
            payload.get('extract_status') == 'pending'
        )

        if not needs_extraction:
            logger.info("No document extraction needed")
            return

        logger.info("Document extraction needed, preparing to trigger...")

        # Get environment variables for doc-extract-start
        if not DOC_EXTRACT_WEBHOOK_SECRET:
            logger.warning("DOC_EXTRACT_WEBHOOK_SECRET not configured, skipping document extraction")

            # Update case with error status
            try:
                url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                params = {'token': f'eq.{token}'}
                headers = supabase_headers()

                updated_payload = {
                    **payload,
                    'extract_status': 'not_configured',
                    'extract_error': 'DOC_EXTRACT_WEBHOOK_SECRET environment variable not set'
                }

                requests.patch(url, params=params, headers=headers,
                             json={'payload': updated_payload}, timeout=TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to update case with config error: {str(e)}")
            return

        # Determine what to extract
        storage_path = None
        filename = None
        mime_type = None

        if payload.get('pastedText'):
            storage_path = f"virtual/{token}/pasted_text.txt"
            filename = "pasted_text.txt"
            mime_type = "text/plain"
            logger.info("Processing pasted text as virtual document")
        elif payload.get('additional_docs') and len(payload.get('additional_docs', [])) > 0:
            first_doc = payload['additional_docs'][0]
            storage_path = first_doc.get('storage_path') or first_doc.get('path')
            filename = first_doc.get('filename') or first_doc.get('name')
            mime_type = first_doc.get('mime_type') or first_doc.get('type')
            logger.info(f"Processing uploaded document: {filename}")

        if not storage_path:
            logger.warning("No storage path found for document extraction")
            return

        # Call doc-extract-start function
        logger.info("Calling doc-extract-start function...")
        doc_extract_url = f"{request.url_root.rstrip('/')}/api/doc-extract-start"

        extract_response = requests.post(
            doc_extract_url,
            headers={
                'Content-Type': 'application/json',
                'x-doc-secret': DOC_EXTRACT_WEBHOOK_SECRET,
                'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}'
            },
            json={
                'token': token,
                'storage_path': storage_path,
                'filename': filename,
                'mime_type': mime_type
            },
            timeout=TIMEOUT
        )

        logger.info(f"Extract response status: {extract_response.status_code}")

        if extract_response.ok:
            logger.info("Document extraction triggered successfully")

            try:
                url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                params = {'token': f'eq.{token}'}
                headers = supabase_headers()

                updated_payload = {
                    **payload,
                    'extract_status': 'triggered',
                    'extract_triggered_at': datetime.utcnow().isoformat()
                }

                requests.patch(url, params=params, headers=headers,
                             json={'payload': updated_payload}, timeout=TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to update case status after triggering: {str(e)}")
            return

        if extract_response.status_code == 404:
            logger.warning("doc-extract-start function not deployed yet, skipping")
            try:
                url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                params = {'token': f'eq.{token}'}
                headers = supabase_headers()

                updated_payload = {
                    **payload,
                    'extract_status': 'not_deployed',
                    'extract_error': 'Document extraction function not yet deployed'
                }

                requests.patch(url, params=params, headers=headers,
                             json={'payload': updated_payload}, timeout=TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to update case with not deployed error: {str(e)}")
            return

        if extract_response.status_code == 401:
            logger.error("Unauthorized - check DOC_EXTRACT_WEBHOOK_SECRET")
            error_text = extract_response.text if hasattr(extract_response, 'text') else ""
            try:
                url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                params = {'token': f'eq.{token}'}
                headers = supabase_headers()

                updated_payload = {
                    **payload,
                    'extract_status': 'auth_failed',
                    'extract_error': 'Authentication failed - check webhook secret configuration',
                    'extract_error_detail': error_text[:500] if error_text else ""
                }

                requests.patch(url, params=params, headers=headers,
                             json={'payload': updated_payload}, timeout=TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to update case with auth error: {str(e)}")
            return

        # Other failure
        error_text = extract_response.text if hasattr(extract_response, 'text') else ""
        logger.error(f"Failed to trigger document extraction: {extract_response.status_code}, {error_text}")

        try:
            url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            params = {'token': f'eq.{token}'}
            headers = supabase_headers()

            updated_payload = {
                **payload,
                'extract_status': 'failed',
                'extract_error': f"HTTP {extract_response.status_code}: {error_text}"[:500]
            }

            requests.patch(url, params=params, headers=headers,
                         json={'payload': updated_payload}, timeout=TIMEOUT)
        except Exception as e:
            logger.error(f"Failed to update case with failure: {str(e)}")

    except Exception as error:
        logger.error(f"Error in trigger_document_extraction_async: {str(error)}")


@app.route('/api/save-case', methods=['POST', 'OPTIONS'])
def save_case():
    """Save case endpoint (converted from Deno/TypeScript save-case function)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    try:
        # Environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing required environment variables'})
            return add_cors_headers(response), 500

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = body.get('token')
        payload = body.get('payload')

        if not token or not payload:
            response = jsonify({'error': 'Token and payload are required'})
            return add_cors_headers(response), 400

        # Validate token format
        if not str(token).startswith('case_'):
            response = jsonify({'error': 'Invalid token format'})
            return add_cors_headers(response), 400

        # Check if case already exists
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'id,payload,created_at'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            existing_case = cases[0] if cases else None
        except Exception as e:
            logger.error(f"Database error reading case: {str(e)}")
            response = jsonify({'error': 'Failed to check existing case'})
            return add_cors_headers(response), 500

        result = None

        if existing_case:
            # Case exists - update with merged payload
            existing_payload = existing_case.get('payload') or {}
            if isinstance(existing_payload, str):
                try:
                    existing_payload = json.loads(existing_payload)
                except:
                    existing_payload = {}

            merged_payload = {**existing_payload, **payload}

            update_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            update_params = {'token': f'eq.{token}'}
            update_data = {
                'payload': merged_payload,
                'updated_at': datetime.utcnow().isoformat()
            }
            update_headers = supabase_headers()
            update_headers['Prefer'] = 'return=representation'

            try:
                update_response = requests.patch(update_url, params=update_params,
                                               headers=update_headers, json=update_data, timeout=TIMEOUT)
                update_response.raise_for_status()
                result = update_response.json()
                logger.info(f"Case updated: {token}")
            except Exception as e:
                logger.error(f"Database update error: {str(e)}")
                response = jsonify({'error': 'Failed to update case data'})
                return add_cors_headers(response), 500

        else:
            # Case doesn't exist - create new
            insert_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            insert_data = {
                'token': token,
                'payload': payload,
                'status': 'new',
                'unlocked': False,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            insert_headers = supabase_headers()
            insert_headers['Prefer'] = 'return=representation'

            try:
                insert_response = requests.post(insert_url, headers=insert_headers,
                                              json=insert_data, timeout=TIMEOUT)
                insert_response.raise_for_status()
                result = insert_response.json()
                logger.info(f"Case created: {token}")
            except Exception as e:
                logger.error(f"Database insert error: {str(e)}")
                response = jsonify({'error': 'Failed to create case data'})
                return add_cors_headers(response), 500

        # Log the save event for audit (non-critical)
        try:
            event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
            event_data = {
                'token': token,
                'type': 'case_updated' if existing_case else 'case_created',
                'data': {
                    'payload_keys': list((payload or {}).keys()),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            event_headers = supabase_headers()

            requests.post(event_url, headers=event_headers, json=event_data, timeout=TIMEOUT)
        except Exception as e:
            logger.warning(f"Failed to log event (non-critical): {str(e)}")

        # Start document extraction in background thread with delay
        def delayed_extraction():
            time.sleep(2)  # 2 second delay to ensure commit propagation
            trigger_document_extraction_async(token, payload)

        extraction_thread = threading.Thread(target=delayed_extraction)
        extraction_thread.daemon = True
        extraction_thread.start()

        case_id = result[0].get('id') if result and len(result) > 0 else None
        response = jsonify({'success': True, 'case_id': case_id})
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error(f"Save case error: {str(e)}")
        response = jsonify({'error': str(e) or 'Internal server error'})
        return add_cors_headers(response), 500


def clamp_text(s: str, max_length: int = 1200) -> str:
    """Clamp text to maximum length"""
    text = str(s or "").strip()
    if not text:
        return ""
    return text[:max_length] if len(text) > max_length else text


@app.route('/api/send-message', methods=['POST', 'OPTIONS'])
def send_message():
    """Send chat message endpoint (converted from Deno/TypeScript send-message function)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type')
        return response

    if request.method != 'POST':
        response = jsonify({'error': 'Method not allowed'})
        return add_cors_headers(response), 405

    try:
        # Validate environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
            response = jsonify({'error': 'Missing env vars'})
            return add_cors_headers(response), 500

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = str(body.get('token', '')).strip()
        user_content = clamp_text(body.get('content', ''), 1000)

        if not token or not user_content:
            response = jsonify({'error': 'token and content are required'})
            return add_cors_headers(response), 400

        # 1) Ensure case exists + is unlocked
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'token,unlocked,payload'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None
        except Exception as e:
            logger.error('Database error reading case', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading case'})
            return add_cors_headers(response), 500

        if not case_row:
            response = jsonify({'error': 'Case not found'})
            return add_cors_headers(response), 404

        if not case_row.get('unlocked'):
            response = jsonify({'error': 'Case is not unlocked'})
            return add_cors_headers(response), 402

        # 2) Save user message
        user_message_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
        user_message_data = {
            'token': token,
            'role': 'user',
            'content': user_content
        }
        user_message_headers = supabase_headers()
        user_message_headers['Prefer'] = 'return=representation'

        try:
            user_message_response = requests.post(user_message_url, headers=user_message_headers,
                                                json=user_message_data, timeout=TIMEOUT)
            user_message_response.raise_for_status()
            user_insert = user_message_response.json()
            user_message = user_insert[0] if user_insert else None
        except Exception as e:
            logger.error('Failed to save user message', extra={'error': str(e)})
            response = jsonify({'error': 'Failed to save user message'})
            return add_cors_headers(response), 500

        # 3) Load recent history (keep it lightweight)
        history_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
        history_params = {
            'token': f'eq.{token}',
            'select': 'role,content,created_at',
            'order': 'created_at.asc',
            'limit': '20'
        }
        history_headers = supabase_headers()

        try:
            history_response = requests.get(history_url, params=history_params, headers=history_headers, timeout=TIMEOUT)
            history_response.raise_for_status()
            history = history_response.json() or []
        except Exception as e:
            logger.error('Failed to load chat history', extra={'error': str(e)})
            response = jsonify({'error': 'Failed to load chat history'})
            return add_cors_headers(response), 500

        payload = case_row.get('payload') or {}

        system_prompt = """
You are "Dispute My HOA" chat support.
This is educational drafting assistance, not legal advice.

You help the homeowner with:
- understanding the notice
- drafting a practical response
- next steps / timelines
- reducing escalation risk

Rules:
- Be calm, practical, specific.
- Ask 1-2 clarifying questions only if truly needed.
- If the user asks for a new letter, produce a ready-to-send draft (plain text).
- Do NOT claim to be a lawyer.
"""

        # 4) Call OpenAI (Responses API)
        history_text = '\n'.join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])

        input_messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': (
                    f"Case payload JSON:\n{json.dumps(payload)}\n\n"
                    f"Recent chat history:\n{history_text}\n\n"
                    f"User message:\n{user_content}\n\nRespond as the assistant."
                )
            }
        ]

        openai_payload = {
            'model': 'gpt-4o-mini',
            'input': input_messages
        }

        try:
            openai_response = requests.post(
                'https://api.openai.com/v1/responses',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json=openai_payload,
                timeout=(10, 60)
            )

            if not openai_response.ok:
                error_text = openai_response.text
                logger.error('OpenAI call failed', extra={'status': openai_response.status_code, 'error': error_text})
                response = jsonify({'error': 'OpenAI call failed', 'details': error_text})
                return add_cors_headers(response), 500

            openai_json = openai_response.json()

            # Extract assistant text from Responses API output safely
            assistant_text = ""
            output = openai_json.get('output')
            if isinstance(output, list):
                for item in output:
                    content = item.get('content') if item else None
                    if isinstance(content, list):
                        for c in content:
                            if (c and c.get('type') == 'output_text' and
                                isinstance(c.get('text'), str)):
                                assistant_text = c['text']
                                break
                        if assistant_text:
                            break

            assistant_text = clamp_text(
                assistant_text or "I can help—what do you want to do next?",
                2000
            )

        except Exception as e:
            logger.error('OpenAI API error', extra={'error': str(e)})
            response = jsonify({'error': 'OpenAI API error', 'details': str(e)})
            return add_cors_headers(response), 500

        # 5) Save assistant message
        assistant_message_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
        assistant_message_data = {
            'token': token,
            'role': 'assistant',
            'content': assistant_text
        }
        assistant_message_headers = supabase_headers()
        assistant_message_headers['Prefer'] = 'return=representation'

        try:
            assistant_message_response = requests.post(assistant_message_url, headers=assistant_message_headers,
                                                     json=assistant_message_data, timeout=TIMEOUT)
            assistant_message_response.raise_for_status()
            assistant_insert = assistant_message_response.json()
            assistant_message = assistant_insert[0] if assistant_insert else None
        except Exception as e:
            logger.error('Failed to save assistant message', extra={'error': str(e)})
            response = jsonify({'error': 'Failed to save assistant message'})
            return add_cors_headers(response), 500

        response = jsonify({
            'ok': True,
            'token': token,
            'user_message': user_message,
            'assistant_message': assistant_message
        })
        return add_cors_headers(response), 200

    except Exception as e:
        logger.error('send-message error', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


@app.route('/api/start-extraction', methods=['POST', 'OPTIONS'])
def start_extraction():
    """Document extraction start endpoint (converted from Deno/TypeScript start-extraction function)"""

    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'ok': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type, x-doc-secret')
        return response

    # Add CORS headers to actual response
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'authorization, x-client-info, apikey, content-type, x-doc-secret')
        return response

    def safe_trim(v) -> str:
        return str(v or '').strip()

    logger.info(f"[{datetime.utcnow().isoformat()}] {request.method} {request.url}")

    if request.method != 'POST':
        response = jsonify({'error': 'Method not allowed'})
        return add_cors_headers(response), 405

    try:
        # Environment variables check
        WEBHOOK_URL = os.environ.get('DOC_EXTRACT_WEBHOOK_URL', f"{request.url_root.rstrip('/')}/webhooks/doc-extract")
        WEBHOOK_SECRET = os.environ.get('DOC_EXTRACT_WEBHOOK_SECRET')
        DOC_BUCKET = os.environ.get('DOC_EXTRACT_BUCKET', 'dmhoa-docs')

        logger.info('Env check', extra={
            'hasUrl': bool(SUPABASE_URL),
            'hasServiceRole': bool(SUPABASE_SERVICE_ROLE_KEY),
            'hasWebhookUrl': bool(WEBHOOK_URL),
            'hasWebhookSecret': bool(DOC_EXTRACT_WEBHOOK_SECRET),
            'bucket': DOC_BUCKET
        })

        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            response = jsonify({'error': 'Missing SUPABASE env vars'})
            return add_cors_headers(response), 500

        if not WEBHOOK_URL or not WEBHOOK_SECRET:
            response = jsonify({'error': 'Missing webhook env vars'})
            return add_cors_headers(response), 500

        # Validate secret header
        incoming_secret = request.headers.get('x-doc-secret', '').strip()
        if not incoming_secret or incoming_secret != WEBHOOK_SECRET:
            logger.error('Unauthorized - secret mismatch')
            response = jsonify({'error': 'Unauthorized'})
            return add_cors_headers(response), 401

        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = safe_trim(body.get('token'))
        storage_path = safe_trim(body.get('storage_path'))
        filename = safe_trim(body.get('filename')) or None
        mime_type = safe_trim(body.get('mime_type')) or None

        if not token or not storage_path:
            response = jsonify({'error': 'token and storage_path are required'})
            return add_cors_headers(response), 400

        # Ensure case exists (fail fast)
        case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        case_params = {
            'token': f'eq.{token}',
            'select': 'id,token,payload,created_at'
        }
        case_headers = supabase_headers()

        try:
            case_response = requests.get(case_url, params=case_params, headers=case_headers, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None
        except Exception as e:
            logger.error('Case lookup error', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading case', 'details': str(e)})
            return add_cors_headers(response), 500

        if not case_row:
            response = jsonify({'error': 'Case not found', 'token': token})
            return add_cors_headers(response), 404

        # 1) Create (or reuse) a dmhoa_documents row for this file
        document_id = None

        # Check for existing document
        doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        doc_params = {
            'token': f'eq.{token}',
            'path': f'eq.{storage_path}',
            'select': 'id,status'
        }
        doc_headers = supabase_headers()

        try:
            doc_response = requests.get(doc_url, params=doc_params, headers=doc_headers, timeout=TIMEOUT)
            doc_response.raise_for_status()
            docs = doc_response.json()
            existing_doc = docs[0] if docs else None
        except Exception as e:
            logger.error('Doc lookup error', extra={'error': str(e)})
            response = jsonify({'error': 'Database error reading document', 'details': str(e)})
            return add_cors_headers(response), 500

        if existing_doc and existing_doc.get('id'):
            document_id = existing_doc['id']
            logger.info('Reusing existing dmhoa_documents row', extra={
                'document_id': document_id,
                'status': existing_doc.get('status')
            })
        else:
            # Insert new document
            insert_doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
            insert_doc_data = {
                'token': token,
                'bucket': DOC_BUCKET,
                'path': storage_path,
                'filename': filename,
                'mime_type': mime_type,
                'status': 'pending'
            }
            insert_doc_headers = supabase_headers()
            insert_doc_headers['Prefer'] = 'return=representation'

            try:
                insert_doc_response = requests.post(insert_doc_url, headers=insert_doc_headers,
                                                  json=insert_doc_data, timeout=TIMEOUT)
                insert_doc_response.raise_for_status()
                inserted_docs = insert_doc_response.json()
                document_id = inserted_docs[0]['id'] if inserted_docs else None
                logger.info('Created dmhoa_documents row', extra={'document_id': document_id})
            except Exception as e:
                logger.error('Doc insert error', extra={'error': str(e)})
                response = jsonify({'error': 'Failed to create dmhoa_documents row', 'details': str(e)})
                return add_cors_headers(response), 500

        if not document_id:
            response = jsonify({'error': 'Could not determine document_id'})
            return add_cors_headers(response), 500

        # 2) Update case payload summary status (optional, but useful for UI)
        current_payload = {}
        try:
            payload_data = case_row.get('payload')
            if isinstance(payload_data, str):
                current_payload = json.loads(payload_data)
            elif isinstance(payload_data, dict):
                current_payload = payload_data
        except Exception:
            current_payload = {}

        next_payload = {
            **current_payload,
            'extract_status': 'triggered',
            'notice_storage_path': storage_path,
            'notice_filename': filename,
            'notice_mime_type': mime_type,
            'extract_triggered_at': datetime.utcnow().isoformat(),
            'document_id': document_id
        }

        # Update case payload
        update_case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        update_case_params = {'token': f'eq.{token}'}
        update_case_data = {'payload': next_payload}
        update_case_headers = supabase_headers()

        try:
            update_case_response = requests.patch(update_case_url, params=update_case_params,
                                                headers=update_case_headers, json=update_case_data, timeout=TIMEOUT)
            update_case_response.raise_for_status()
        except Exception as e:
            logger.error('Case payload update error', extra={'error': str(e)})
            # Not fatal—doc record exists and webhook can still run

        # 3) Mark document processing BEFORE webhook
        mark_proc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
        mark_proc_params = {'id': f'eq.{document_id}'}
        mark_proc_data = {'status': 'processing', 'error': None}
        mark_proc_headers = supabase_headers()

        try:
            mark_proc_response = requests.patch(mark_proc_url, params=mark_proc_params,
                                              headers=mark_proc_headers, json=mark_proc_data, timeout=TIMEOUT)
            mark_proc_response.raise_for_status()
        except Exception as e:
            logger.error('Failed to mark document processing', extra={'error': str(e)})
            # Not fatal, but you'll want to know

        # 4) Call backend webhook (server-to-server)
        logger.info('Calling webhook', extra={'webhook_url': WEBHOOK_URL})

        webhook_payload = {
            'token': token,
            'document_id': document_id,
            'bucket': DOC_BUCKET,
            'path': storage_path,
            'filename': filename,
            'mime_type': mime_type,
            'supabase_url': SUPABASE_URL  # optional
        }

        webhook_headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Secret': WEBHOOK_SECRET,
            'X-Doc-Extract-Secret': WEBHOOK_SECRET  # Alternative header name
        }

        try:
            webhook_response = requests.post(WEBHOOK_URL, headers=webhook_headers,
                                           json=webhook_payload, timeout=(10, 120))  # Longer timeout for processing

            logger.info('Webhook response', extra={'status': webhook_response.status_code, 'ok': webhook_response.ok})

            if not webhook_response.ok:
                error_text = webhook_response.text
                logger.error('Webhook failed', extra={'error': error_text})

                # Update document status to failed
                fail_doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
                fail_doc_params = {'id': f'eq.{document_id}'}
                fail_doc_data = {
                    'status': 'failed',
                    'error': f'Webhook {webhook_response.status_code}: {error_text}'[:1500]
                }
                fail_doc_headers = supabase_headers()

                try:
                    requests.patch(fail_doc_url, params=fail_doc_params,
                                 headers=fail_doc_headers, json=fail_doc_data, timeout=TIMEOUT)
                except Exception:
                    pass  # Best effort

                # Also reflect summary status on the case payload (optional)
                fail_payload = {
                    **next_payload,
                    'extract_status': 'failed',
                    'extract_error': f'Webhook {webhook_response.status_code}: {error_text}'[:1500],
                    'extract_failed_at': datetime.utcnow().isoformat()
                }

                try:
                    requests.patch(update_case_url, params=update_case_params,
                                 headers=update_case_headers, json={'payload': fail_payload}, timeout=TIMEOUT)
                except Exception:
                    pass  # Best effort

                response = jsonify({
                    'error': 'Webhook call failed',
                    'status': webhook_response.status_code,
                    'details': error_text
                })
                return add_cors_headers(response), 502

            # If webhook returns JSON, capture it (optional)
            try:
                webhook_json = webhook_response.json()
            except Exception:
                webhook_json = {}

            # Mark queued/accepted (document is now in backend pipeline)
            queue_doc_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
            queue_doc_params = {'id': f'eq.{document_id}'}
            queue_doc_data = {'status': 'processing'}
            queue_doc_headers = supabase_headers()

            try:
                requests.patch(queue_doc_url, params=queue_doc_params,
                             headers=queue_doc_headers, json=queue_doc_data, timeout=TIMEOUT)
            except Exception:
                pass  # Best effort

            ok_payload = {
                **next_payload,
                'extract_status': 'queued',
                'webhook_response': webhook_json,
                'extract_queued_at': datetime.utcnow().isoformat()
            }

            try:
                requests.patch(update_case_url, params=update_case_params,
                             headers=update_case_headers, json={'payload': ok_payload}, timeout=TIMEOUT)
            except Exception:
                pass  # Best effort

            response = jsonify({
                'ok': True,
                'token': token,
                'document_id': document_id,
                'bucket': DOC_BUCKET,
                'path': storage_path,
                'webhook': webhook_json
            })
            return add_cors_headers(response), 200

        except Exception as e:
            logger.error('Webhook request failed', extra={'error': str(e)})
            response = jsonify({'error': f'Webhook request failed: {str(e)}'})
            return add_cors_headers(response), 502

    except Exception as e:
        logger.error('Unexpected error in start-extraction', extra={'error': str(e)})
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


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

            # Get token from client_reference_id or metadata
            token = session.get('client_reference_id') or session.get('metadata', {}).get('token')
            if not token:
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
                logger.warning(f"Failed to log payment_completed event (non-fatal): {str(e)}")

            logger.info(f"Payment completion processed successfully for token: {token}")

        return jsonify({'received': True}), 200

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({'error': 'Webhook error'}), 500