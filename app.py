import os
import io
import logging
from typing import Dict, Any, Optional, Tuple

import requests
from flask import Flask, request, jsonify
from pypdf import PdfReader

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

# SMTP Configuration - Optional, only needed for email functionality
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
# Clean SMTP credentials to handle non-ASCII characters like non-breaking spaces
SMTP_USER = (os.environ.get("SMTP_USER") or "").strip().replace('\xa0', ' ')
SMTP_PASS = (os.environ.get("SMTP_PASS") or "").strip().replace('\xa0', ' ')
SMTP_FROM = os.environ.get("SMTP_FROM", "support@disputemyhoa.com")

SMTP_SENDER_WEBHOOK_SECRET = os.environ.get("SMTP_SENDER_WEBHOOK_SECRET")

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

        # Use Tesseract OCR to extract text
        # Using config options for better accuracy
        custom_config = r'--oem 3 --psm 6 -l eng'
        extracted_text = pytesseract.image_to_string(image, config=custom_config)

        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        char_count = len(extracted_text)

        if char_count == 0:
            return "", 1, 0, "No text found in image - image may be blank or contain no readable text"

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
            }), 400

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


if __name__ == '__main__':
    # Validate required environment variables
    required_env_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'DOC_EXTRACT_WEBHOOK_SECRET']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)

    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
