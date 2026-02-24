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

# Statute lookup for state-specific HOA law context
from statute_lookup import (
    get_statute_context,
    extract_state_from_payload,
    extract_violation_type_from_payload
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# NEW: Updated CORS configuration with comprehensive headers and methods
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "apikey", "x-client-info", "x-supabase-api-version",
                    "X-Webhook-Secret"],
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

# Anthropic Configuration (for Claude API)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

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

# Klaviyo Configuration
KLAVIYO_API_KEY = os.environ.get("KLAVIYO_API_KEY")
KLAVIYO_QUICK_PREVIEW_LIST_ID = os.environ.get("KLAVIYO_QUICK_PREVIEW_LIST_ID")
KLAVIYO_FULL_PREVIEW_LIST_ID = os.environ.get("KLAVIYO_FULL_PREVIEW_LIST_ID")
KLAVIYO_POST_PURCHASE_LIST_ID = os.environ.get("KLAVIYO_POST_PURCHASE_LIST_ID")

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


def klaviyo_headers() -> Dict[str, str]:
    """Return headers for Klaviyo API requests."""
    return {
        'Authorization': f'Klaviyo-API-Key {KLAVIYO_API_KEY}',
        'Content-Type': 'application/json',
        'revision': '2024-02-15'
    }


def klaviyo_get_profile_id_by_email(email: str) -> Optional[str]:
    """Get a Klaviyo profile ID by email address."""
    if not KLAVIYO_API_KEY:
        return None

    try:
        url = "https://a.klaviyo.com/api/profiles/"
        params = {
            'filter': f'equals(email,"{email}")'
        }
        response = requests.get(url, headers=klaviyo_headers(), params=params, timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            profiles = data.get('data', [])
            if profiles:
                return profiles[0].get('id')
        return None

    except Exception as e:
        logger.error(f"Klaviyo get profile error: {str(e)}")
        return None


def klaviyo_create_or_get_profile(email: str) -> Optional[str]:
    """Create a Klaviyo profile or get existing profile ID by email."""
    if not KLAVIYO_API_KEY:
        return None

    try:
        # First try to get existing profile
        profile_id = klaviyo_get_profile_id_by_email(email)
        if profile_id:
            return profile_id

        # Create new profile
        url = "https://a.klaviyo.com/api/profiles/"
        payload = {
            "data": {
                "type": "profile",
                "attributes": {
                    "email": email
                }
            }
        }
        response = requests.post(url, headers=klaviyo_headers(), json=payload, timeout=TIMEOUT)

        if response.status_code in [200, 201]:
            data = response.json()
            profile_id = data.get('data', {}).get('id')
            logger.info(f"Created Klaviyo profile for {email}: {profile_id}")
            return profile_id
        else:
            logger.warning(f"Klaviyo create profile failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Klaviyo create/get profile error: {str(e)}")
        return None


def klaviyo_add_profile_to_list(email: str, list_id: str) -> bool:
    """Add a profile to a Klaviyo list. Creates the profile if it doesn't exist."""
    if not KLAVIYO_API_KEY or not list_id:
        logger.warning("Klaviyo API key or list ID not configured")
        return False

    try:
        # First create or get the profile ID
        profile_id = klaviyo_create_or_get_profile(email)
        if not profile_id:
            logger.warning(f"Could not create or get Klaviyo profile for {email}")
            return False

        # Add profile to list using the profile ID
        url = f"https://a.klaviyo.com/api/lists/{list_id}/relationships/profiles/"
        payload = {
            "data": [
                {
                    "type": "profile",
                    "id": profile_id
                }
            ]
        }
        response = requests.post(url, headers=klaviyo_headers(), json=payload, timeout=TIMEOUT)

        if response.status_code in [200, 201, 202, 204]:
            logger.info(f"Successfully added {email} to Klaviyo list {list_id}")
            return True
        else:
            logger.warning(f"Klaviyo add to list failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Klaviyo add to list error: {str(e)}")
        return False


def klaviyo_remove_profile_from_list(email: str, list_id: str) -> bool:
    """Remove a profile from a Klaviyo list by email."""
    if not KLAVIYO_API_KEY or not list_id:
        return False

    try:
        # First, get the profile ID by email
        profile_id = klaviyo_get_profile_id_by_email(email)
        if not profile_id:
            logger.info(f"No Klaviyo profile found for {email}, nothing to remove")
            return True  # Nothing to remove is still a success

        url = f"https://a.klaviyo.com/api/lists/{list_id}/relationships/profiles/"
        payload = {
            "data": [
                {
                    "type": "profile",
                    "id": profile_id
                }
            ]
        }
        response = requests.delete(url, headers=klaviyo_headers(), json=payload, timeout=TIMEOUT)

        if response.status_code in [200, 204]:
            logger.info(f"Successfully removed {email} from Klaviyo list {list_id}")
            return True
        else:
            logger.warning(f"Klaviyo remove from list failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Klaviyo remove from list error: {str(e)}")
        return False


def klaviyo_sync_profile_to_list(email: str, target_list_id: str) -> bool:
    """
    Add email to target list and remove from other DMHOA lists.
    Ensures an email is only in one list at a time.
    """
    if not KLAVIYO_API_KEY:
        logger.warning("Klaviyo API key not configured, skipping list sync")
        return False

    all_list_ids = [
        KLAVIYO_QUICK_PREVIEW_LIST_ID,
        KLAVIYO_FULL_PREVIEW_LIST_ID,
        KLAVIYO_POST_PURCHASE_LIST_ID
    ]

    # Remove from all other lists first
    for list_id in all_list_ids:
        if list_id and list_id != target_list_id:
            klaviyo_remove_profile_from_list(email, list_id)

    # Add to target list
    return klaviyo_add_profile_to_list(email, target_list_id)


def klaviyo_update_profile_properties(email: str, properties: Dict) -> bool:
    """
    Update custom properties on a Klaviyo profile.

    Args:
        email: The email address of the profile to update
        properties: Dictionary of custom properties to set on the profile

    Returns:
        True if successful, False otherwise
    """
    if not KLAVIYO_API_KEY:
        logger.warning("Klaviyo API key not configured, skipping profile update")
        return False

    if not properties:
        logger.warning("No properties provided for Klaviyo profile update")
        return False

    try:
        # Get the profile ID
        profile_id = klaviyo_get_profile_id_by_email(email)
        if not profile_id:
            # Create profile if it doesn't exist
            profile_id = klaviyo_create_or_get_profile(email)
            if not profile_id:
                logger.warning(f"Could not create or get Klaviyo profile for {email}")
                return False

        # Update profile properties using PATCH
        url = f"https://a.klaviyo.com/api/profiles/{profile_id}/"
        request_payload = {
            "data": {
                "type": "profile",
                "id": profile_id,
                "attributes": {
                    "properties": properties
                }
            }
        }
        logger.info(f"Klaviyo PATCH request to {url} with properties: {properties}")
        response = requests.patch(url, headers=klaviyo_headers(), json=request_payload, timeout=TIMEOUT)

        if response.status_code in [200, 201, 204]:
            logger.info(f"Successfully updated Klaviyo profile properties for {email}: {list(properties.keys())} - Status: {response.status_code}")
            return True
        else:
            logger.warning(f"Klaviyo profile update failed: {response.status_code} - {response.text[:500]}")
            return False

    except Exception as e:
        logger.error(f"Klaviyo profile update error: {str(e)}")
        return False


def determine_klaviyo_abandonment_list(payload: Dict) -> Optional[str]:
    """
    Determine which abandonment list an email should be added to based on payload.

    Quick Preview: Has basic info but no document_id or webhook_response
    Full Preview: Has document_id and/or webhook_response indicating document processing

    Returns the appropriate list ID or None if unable to determine.
    """
    if not payload:
        return None

    # Check for full preview indicators
    has_document_id = payload.get('document_id') is not None
    has_webhook_response = payload.get('webhook_response') is not None
    has_notice_storage_path = payload.get('notice_storage_path') is not None

    is_full_preview = has_document_id or has_webhook_response or has_notice_storage_path
    logger.info(f"Klaviyo list determination: document_id={has_document_id}, webhook_response={has_webhook_response}, notice_storage_path={has_notice_storage_path} -> {'FULL' if is_full_preview else 'QUICK'} preview")

    if is_full_preview:
        return KLAVIYO_FULL_PREVIEW_LIST_ID
    else:
        return KLAVIYO_QUICK_PREVIEW_LIST_ID


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
            r'--oem 1 --psm 1 -l eng',  # Automatic page segmentation with OSD
            r'--oem 1 --psm 3 -l eng',  # Fully automatic page segmentation, but no OSD
            r'--oem 1 --psm 4 -l eng',  # Assume a single column of text of variable sizes
            r'--oem 1 --psm 6 -l eng',  # Assume a single uniform block of text
            r'--oem 3 --psm 1 -l eng',  # LSTM with automatic page segmentation
            r'--oem 3 --psm 3 -l eng',  # LSTM with fully automatic page segmentation
            r'--oem 3 --psm 6 -l eng',  # LSTM standard config
            r'--oem 3 --psm 11 -l eng',  # Sparse text - find as much text as possible
            r'--oem 3 --psm 12 -l eng',  # Sparse text with OSD
            r'--psm 6',  # No language specified fallback
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
                    alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in current_text) / max(len(current_text),
                                                                                                     1)
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

        logger.info(
            f"Final file detection - filename: '{filename}', mime_type: '{mime_type}', detected_type: {detected_type}")

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
                                   token_usage: Dict = None, latency_ms: int = None,
                                   preview_json: Optional[Dict] = None) -> bool:
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
                if preview_json and 'your_situation' in preview_json and 'rules_cited' in preview_json[
                    'your_situation']:
                    preview_json['your_situation']['rules_cited'] = clean_rules_cited(
                        preview_json['your_situation']['rules_cited'])

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
                if preview_json and 'your_situation' in preview_json and 'rules_cited' in preview_json[
                    'your_situation']:
                    preview_json['your_situation']['rules_cited'] = clean_rules_cited(
                        preview_json['your_situation']['rules_cited'])

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
    """Generate ONLY final preview when documents are ready or after 30-second timeout.

    This function handles THREE scenarios:
    1. Case has uploaded documents (in dmhoa_documents table) → use document text
    2. Case has pasted text (in payload.pastedText) → use pasted text
    3. Case has neither → generate basic preview without document content
    """
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

            # Check for pasted text in payload
            payload = case.get('payload', {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    payload = {}

            pasted_text = payload.get('pastedText', '')
            has_pasted_text = bool(pasted_text and pasted_text.strip())

            # Check document status
            documents = fetch_ready_documents_by_token(token, limit=5)
            all_documents = fetch_any_documents_status_by_token(token)

            has_processing_documents = any(doc.get('status') in ['pending', 'processing'] for doc in all_documents)

            # Check existing preview
            existing_preview = read_active_preview(case_id)

            # Determine what content source we have
            # Priority: 1) Ready documents, 2) Pasted text, 3) None
            content_source = "none"
            if documents:
                content_source = "documents"
            elif has_pasted_text:
                content_source = "pasted_text"

            logger.info(f"Preview generation for {token[:12]}...: content_source={content_source}, "
                        f"docs={len(documents)}, has_pasted_text={has_pasted_text}, "
                        f"has_processing_docs={has_processing_documents}")

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
                elif has_pasted_text:
                    # Has pasted text - generate preview using pasted text
                    should_generate = True
                    logger.info(f"Pasted text available - generating FINAL preview for case {token[:12]}...")
                elif has_processing_documents:
                    # Documents still processing - wait (don't generate preliminary)
                    logger.info(f"Documents processing for case {token[:12]}... - waiting for FINAL preview")
                    return True  # Return success but don't generate preview yet
                elif not all_documents:
                    # No documents at all and no pasted text - generate final basic preview
                    should_generate = True
                    logger.info(
                        f"No documents or pasted text - generating FINAL basic preview for case {token[:12]}...")
                else:
                    # Documents exist but not ready yet - wait
                    logger.info(f"Waiting for documents to be ready for case {token[:12]}...")
                    return True
            else:
                # Preview already exists - check if we need to upgrade
                existing_content = existing_preview.get('preview_content', {})
                existing_doc_summary = existing_content.get('doc_summary', {})
                existing_doc_status = existing_doc_summary.get('doc_status', 'none')

                if documents and existing_doc_status in ['none', 'pasted_text']:
                    # We have ready documents but existing preview doesn't - upgrade to final
                    should_generate = True
                    preview_type = "final_upgrade"
                    logger.info(f"Upgrading to FINAL preview with documents for case {token[:12]}...")
                elif has_pasted_text and existing_doc_status == 'none':
                    # We have pasted text but existing preview has none - upgrade
                    should_generate = True
                    preview_type = "final_upgrade_pasted"
                    logger.info(f"Upgrading to FINAL preview with pasted text for case {token[:12]}...")
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
            elif has_pasted_text:
                # Pasted text available - generate preview using pasted text
                logger.info(f"Generating FINAL preview with pasted text for case {token[:12]}...")
                doc_brief = {
                    "doc_status": "pasted_text",
                    "doc_count": 1,
                    "sources": [{"filename": "Pasted Text", "page_count": 1, "char_count": len(pasted_text)}],
                    "brief_text": pasted_text[:3000] if pasted_text else ""
                }
                preview_text, token_usage, latency_ms, preview_json = generate_case_preview_with_pasted_text(case,
                                                                                                             pasted_text,
                                                                                                             doc_brief)
                preview_type = "final_with_pasted_text"
            else:
                # No documents or pasted text - generate final basic preview
                logger.info(f"Generating FINAL basic preview (no documents or pasted text) for case {token[:12]}...")
                doc_brief = {
                    "doc_status": "none",
                    "doc_count": 0,
                    "sources": [],
                    "brief_text": "No documents have been uploaded for analysis."
                }
                preview_text, token_usage, latency_ms, preview_json = generate_preview_without_documents(case)
                preview_type = "final_basic"

            # Save final preview (this will deactivate existing ones automatically)
            success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms,
                                                     preview_json)

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
        return False


# @app.route('/webhooks/generate-preview', methods=['POST'])
# def generate_preview_webhook():
#     """Webhook endpoint to generate case preview - can be called after case creation."""
#     # Validate webhook secret
#     webhook_secret = request.headers.get('X-Webhook-Secret')
#     if not webhook_secret or webhook_secret != DOC_EXTRACT_WEBHOOK_SECRET:
#         logger.warning("Invalid or missing webhook secret for generate-preview")
#         return jsonify({'error': 'Unauthorized'}), 401
#
#     try:
#         # Parse JSON body
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'Invalid JSON body'}), 400
#
#         # Validate required fields
#         token = data.get('token')
#         case_id = data.get('case_id')
#
#         if not token:
#             return jsonify({'error': 'Missing required field: token'}), 400
#
#         logger.info(f"Generating preview for case - Token: {token[:8]}..., Case ID: {case_id}")
#
#         # If case_id not provided, look it up by token
#         if not case_id:
#             case = read_case_by_token(token)
#             if not case:
#                 return jsonify({'error': 'Case not found for token'}), 404
#             case_id = case.get('id')
#             if not case_id:
#                 return jsonify({'error': 'Case ID not found'}), 404
#
#         # Generate preview
#         success = auto_generate_case_preview(token, case_id)
#
#         if success:
#             return jsonify({
#                 'message': 'Preview generated successfully',
#                 'token': token,
#                 'case_id': case_id
#             }), 200
#         else:
#             return jsonify({
#                 'error': 'Failed to generate preview',
#                 'token': token,
#                 'case_id': case_id
#             }), 500
#
#     except Exception as e:
#         error_msg = f"Unexpected error generating preview: {str(e)}"
#         logger.error(error_msg)
#         return jsonify({'error': error_msg}), 500


# @app.route('/webhooks/case-created', methods=['POST'])
# def case_created_webhook():
#     """Webhook to handle case creation and trigger initial preview generation."""
#     # Validate webhook secret
#     webhook_secret = request.headers.get('X-Webhook-Secret')
#     if not webhook_secret or webhook_secret != DOC_EXTRACT_WEBHOOK_SECRET:
#         logger.warning("Invalid or missing webhook secret for case-created")
#         return jsonify({'error': 'Unauthorized'}), 401
#
#     try:
#         # Parse JSON body
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'Invalid JSON body'}), 400
#
#         token = data.get('token')
#         case_id = data.get('case_id')
#
#         if not token:
#             return jsonify({'error': 'Missing required field: token'}), 400
#
#         logger.info(f"Case created - Token: {token[:8]}..., Case ID: {case_id}")
#
#         # If case_id not provided, look it up by token
#         if not case_id:
#             case = read_case_by_token(token)
#             if not case:
#                 return jsonify({'error': 'Case not found for token'}), 404
#             case_id = case.get('id')
#
#         # Generate immediate preview
#         immediate_success = auto_generate_case_preview(token, case_id)
#
#         # Check if there are any documents that might need processing
#         all_documents = fetch_any_documents_status_by_token(token)
#         pending_or_processing_docs = [doc for doc in all_documents if doc.get('status') in ['pending', 'processing']]
#
#         # Only schedule delayed jobs if there are documents that might become ready
#         if pending_or_processing_docs:
#             logger.info(f"Found {len(pending_or_processing_docs)} pending/processing documents - scheduling delayed preview generations")
#             # Delayed: Give time for documents to be uploaded and processed, then regenerate
#         else:
#             logger.info(f"No pending/processing documents found - skipping delayed preview generations")
#
#         return jsonify({
#             'message': 'Case creation handled and preview generation scheduled',
#             'token': token,
#             'case_id': case_id,
#             'immediate_preview': immediate_success,
#             'delayed_jobs_scheduled': len(pending_or_processing_docs) > 0
#         }), 200
#
#     except Exception as e:
#         error_msg = f"Unexpected error handling case creation: {str(e)}"
#         logger.error(error_msg)
#         return jsonify({'error': error_msg}), 500


def create_case_in_supabase(case_data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """Create a new case in Supabase database or update existing one."""
    try:
        token = case_data.get('token')
        if not token:
            logger.error("No token provided for case creation")
            return False, None, None

        # First, check if case already exists
        check_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        check_params = {'token': f'eq.{token}', 'select': 'id,token,status'}
        check_headers = supabase_headers()

        check_response = requests.get(check_url, params=check_params, headers=check_headers, timeout=TIMEOUT)
        check_response.raise_for_status()
        existing_cases = check_response.json()

        # Get email from the case data (check multiple possible field names)
        email = (case_data.get('email') or
                 case_data.get('ownerEmail') or
                 case_data.get('owner_email') or
                 case_data.get('userEmail') or
                 case_data.get('user_email'))

        # Prepare case data according to the actual table structure
        case_payload = {
            'token': token,
            'email': email,
            'payload': case_data,  # Store the entire case data as JSONB in payload column
            'status': 'preview'  # Set default status
        }

        # Remove None/empty values except for payload which should always be included
        case_payload = {k: v for k, v in case_payload.items() if v is not None and (k == 'payload' or v)}

        if existing_cases and len(existing_cases) > 0:
            # Case exists, update it
            existing_case = existing_cases[0]
            case_id = existing_case.get('id')

            logger.info(f"Case with token {token[:8]}... already exists, updating with ID: {case_id}")

            # Update existing case
            update_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            update_params = {'id': f'eq.{case_id}'}
            update_headers = supabase_headers()
            update_headers['Prefer'] = 'return=representation'

            # Only update payload and email, preserve other fields
            update_payload = {
                'payload': case_data,
                'email': email
            }
            update_payload = {k: v for k, v in update_payload.items() if v is not None}

            update_response = requests.patch(update_url, params=update_params, headers=update_headers,
                                             json=update_payload, timeout=TIMEOUT)
            update_response.raise_for_status()

            result = update_response.json()
            if result and len(result) > 0:
                logger.info(f"Updated existing case successfully - ID: {case_id}, Token: {token[:8]}...")
                return True, case_id, token
            else:
                logger.warning(f"Update succeeded but no data returned for case {case_id}")
                return True, case_id, token

        else:
            # Case doesn't exist, create new one
            logger.info(f"Creating new case with token {token[:8]}...")

            url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            headers = supabase_headers()
            headers['Prefer'] = 'return=representation'

            logger.info(f"Creating case with payload keys: {list(case_payload.keys())}")

            response = requests.post(url, headers=headers, json=case_payload, timeout=TIMEOUT)
            response.raise_for_status()

            result = response.json()
            if result and len(result) > 0:
                case_id = result[0].get('id')
                token_returned = result[0].get('token')
                logger.info(f"Created case successfully - ID: {case_id}, Token: {token_returned[:8]}...")
                return True, case_id, token_returned
            else:
                logger.error("No case data returned from Supabase")
                return False, None, None

    except Exception as e:
        logger.error(f"Failed to create/update case in Supabase: {str(e)}")
        return False, None, None


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
                update_response = requests.patch(update_url, params=update_params, headers=update_headers,
                                                 json=update_data, timeout=TIMEOUT)
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

        # Sync email to appropriate Klaviyo abandonment list (non-critical)
        try:
            email_for_klaviyo = payload.get('email')
            logger.info(f"Klaviyo sync starting for email: {email_for_klaviyo}, token: {token[:20] if token else 'None'}...")
            if email_for_klaviyo:
                target_list_id = determine_klaviyo_abandonment_list(payload)
                case_token_for_klaviyo = token
                form_link = payload.get('fullCaseFormLink')
                logger.info(f"Klaviyo target list: {target_list_id}, form_link: {form_link[:50] if form_link else 'None'}...")

                if target_list_id:
                    is_full_preview = target_list_id == KLAVIYO_FULL_PREVIEW_LIST_ID
                    logger.info(f"Klaviyo is_full_preview: {is_full_preview}")

                    # Run in background to not block the response
                    def sync_klaviyo():
                        list_type = "full_preview" if is_full_preview else "quick_preview"

                        # Step 1: Build profile properties FIRST
                        profile_properties = {}

                        # Always set quick_preview_link if we have a form link
                        if form_link:
                            profile_properties['quick_preview_link'] = form_link

                        # Step 2: For full preview, generate Stripe checkout URL BEFORE syncing to Klaviyo
                        if is_full_preview:
                            # Save case preview link (never expires)
                            frontend_url = "https://disputemyhoa.com"
                            profile_properties['case_preview_link'] = f"{frontend_url}/case-preview.html?case={case_token_for_klaviyo}"

                            # Save direct Stripe checkout link (expires in 24 hours)
                            try:
                                if STRIPE_SECRET_KEY and STRIPE_PRICE_ID:
                                    checkout_session = stripe.checkout.Session.create(
                                        payment_method_types=['card'],
                                        line_items=[{
                                            'price': STRIPE_PRICE_ID,
                                            'quantity': 1,
                                        }],
                                        mode='payment',
                                        allow_promotion_codes=True,
                                        success_url=f"{frontend_url}/case.html?case={case_token_for_klaviyo}&session_id={{CHECKOUT_SESSION_ID}}",
                                        cancel_url=f"{frontend_url}/case-preview.html?case={case_token_for_klaviyo}",
                                        metadata={
                                            'case_token': case_token_for_klaviyo,
                                        }
                                    )
                                    profile_properties['stripe_checkout_link'] = checkout_session.url
                                    logger.info(f"Generated Stripe checkout URL for Klaviyo: {email_for_klaviyo} - {checkout_session.url}")
                                else:
                                    logger.warning(f"Stripe not configured - STRIPE_SECRET_KEY: {bool(STRIPE_SECRET_KEY)}, STRIPE_PRICE_ID: {bool(STRIPE_PRICE_ID)}")
                            except Exception as e:
                                logger.error(f"Failed to create Stripe checkout for Klaviyo: {str(e)}")

                        # Step 3: Sync profile to Klaviyo list
                        klaviyo_sync_profile_to_list(email_for_klaviyo, target_list_id)
                        logger.info(f"Synced {email_for_klaviyo} to Klaviyo {list_type} abandonment list")

                        # Step 4: Update properties on profile
                        if profile_properties:
                            success = klaviyo_update_profile_properties(email_for_klaviyo, profile_properties)
                            logger.info(f"Klaviyo profile properties update for {email_for_klaviyo}: success={success}, properties={list(profile_properties.keys())}")

                    klaviyo_thread = threading.Thread(target=sync_klaviyo)
                    klaviyo_thread.daemon = True
                    klaviyo_thread.start()
        except Exception as e:
            logger.warning(f"Failed to sync Klaviyo list (non-critical): {str(e)}")

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


def generate_preview_for_pasted_text(case_id: str, token: str, pasted_text: str, payload: Dict) -> bool:
    """Generate a preview specifically for cases where text was pasted instead of documents uploaded."""
    try:
        logger.info(f"Generating pasted text preview for case {token[:12]}...")

        # Use concurrency guard to prevent duplicate previews
        if not upsert_active_preview_lock(case_id):
            logger.info(f"Preview already being generated for case {case_id}")
            return True

        try:
            # Fetch case data
            case = read_case_by_token(token)
            if not case:
                logger.error(f"Case not found for token {token[:12]}...")
                return False

            # Build doc_brief for pasted text
            doc_brief = {
                "doc_status": "pasted_text",
                "doc_count": 1,
                "sources": [{"filename": "Pasted Text", "page_count": 1, "char_count": len(pasted_text)}],
                "brief_text": pasted_text[:3000] if pasted_text else ""
            }

            # Generate the preview using the pasted text
            preview_text, token_usage, latency_ms, preview_json = generate_case_preview_with_pasted_text(case,
                                                                                                         pasted_text,
                                                                                                         doc_brief)

            # Save the preview
            success = save_case_preview_to_new_table(case_id, preview_text, doc_brief, token_usage, latency_ms,
                                                     preview_json)

            if success:
                logger.info(f"Successfully generated pasted text preview for case {token[:12]}...")
                return True
            else:
                logger.error(f"Failed to save pasted text preview for case {token[:12]}...")
                return False

        finally:
            # Always release the lock
            release_preview_lock(case_id)

    except Exception as e:
        logger.error(f"Error generating pasted text preview for case {token[:12]}...: {str(e)}")
        release_preview_lock(case_id)
        return False


def generate_case_preview_with_pasted_text(case_data: Dict, pasted_text: str, doc_brief: Dict) -> Tuple[
    str, Dict, int, Optional[Dict]]:
    """Generate case preview using OpenAI with pasted text content."""
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

        # Clip pasted text to avoid token limits
        clipped_text = pasted_text[:8000] if pasted_text else ""

        # Prompt specifically for pasted text
        user_prompt = f"""Create a conversion-optimized preview for this HOA dispute case. The user has pasted text from their HOA notice. Extract specific facts from the pasted text and output ONLY valid JSON.

Case Details:
- HOA: {hoa_name}
- Violation Type: {violation_type}
- Property Address: {property_address}
- Owner: {owner_name}
- Case Description: {case_description}

PASTED TEXT FROM HOA NOTICE:
{clipped_text}

Output ONLY valid JSON with this exact structure:
{{
  "version": "preview_v2_sales",
  "headline": "string (8-14 words, specific to this case. If a deadline is present, headline must include it as 'X-Day Deadline' or the exact date)",
  "why_now": "string (1-2 sentences, tie urgency to the required action and any deadline found in the pasted text)",
  "your_situation": {{
    "alleged_violation": "string (extracted from pasted text)",
    "hoa_demands": ["string", "string", "..."],
    "deadline": "string (use exact date if present in pasted text; else 'Not stated')",
    "rules_cited": ["string (each entry must be either 'Paragraph <...>' / 'Section <...>' / 'Article <...>' if present in pasted text, otherwise 'Not stated')", "..."]
  }},
  "critical_detail_locked": {{
    "title": "Critical Response Wording (Locked)",
    "body": "The exact clause/paragraph language to cite, proof checklist (what the HOA will accept), and extension request wording (preserves rights without admitting fault) are locked. Our analysis shows specific language that could weaken your position if worded incorrectly. This critical phrasing is included in the unlock package."
  }},
  "risk_if_wrong": [
    "string (specific consequence based on pasted text)",
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
- Extract specific facts from the pasted text for your_situation fields
- Use "you" voice throughout
- Be concrete and specific to this case
- Include at least 5 concrete deliverables in what_you_get_when_you_unlock
- Make hard_stop create genuine unfinished business with concrete locked items
- For critical_detail_locked body, reference exact clause language, proof checklist, extension request wording
- Avoid 'admit liability' language unless phrased as 'without admitting fault' or 'without admitting liability'"""

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
            input_cost = (token_usage['prompt_tokens'] / 1000) * 0.00015
            output_cost = (token_usage['completion_tokens'] / 1000) * 0.0006
            token_usage['cost_usd'] = round(input_cost + output_cost, 6)

        # Parse JSON safely and create markdown
        preview_json = None
        try:
            json_start = json_response.find('{')
            json_end = json_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                clean_json = json_response[json_start:json_end]
                preview_json = json.loads(clean_json)

                # Post-processing to clean up rules_cited narrative
                if preview_json and 'your_situation' in preview_json and 'rules_cited' in preview_json[
                    'your_situation']:
                    preview_json['your_situation']['rules_cited'] = clean_rules_cited(
                        preview_json['your_situation']['rules_cited'])

                # Create markdown from JSON
                markdown_preview = render_preview_markdown(preview_json)

                logger.info(f"Generated pasted text preview: {len(markdown_preview)} characters, {latency_ms}ms")
                return markdown_preview, token_usage, latency_ms, preview_json

            else:
                raise json.JSONDecodeError("No valid JSON found", json_response, 0)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            logger.warning(f"Raw response: {json_response}")

            # Fallback to original response as markdown but no preview_json
            markdown_preview = f"# HOA Case Analysis\n\n{json_response}"
            logger.info(f"Generated fallback pasted text preview: {len(markdown_preview)} characters, {latency_ms}ms")
            return markdown_preview, token_usage, latency_ms, None

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Failed to generate pasted text preview: {str(e)}")
        return f"Error generating preview: {str(e)}", {}, latency_ms, None


def trigger_document_extraction_async(token: str, payload: dict):
    """
    Async function to trigger document extraction without blocking the save operation
    """
    try:
        logger.info(f"Checking if document extraction is needed for token: {token}")

        # Check if there are uploaded documents that need processing
        needs_extraction = (
                (payload.get('pastedText') or (
                        payload.get('additional_docs') and len(payload.get('additional_docs', [])) > 0)) and
                payload.get('extract_status') == 'pending'
        )

        if not needs_extraction:
            logger.info("No document extraction needed")
            return

        logger.info("Document extraction needed, preparing to trigger...")

        # NEW: Handle pasted text cases - generate preview immediately
        if payload.get('pastedText') and not payload.get('additional_docs'):
            logger.info("Processing pasted text case - generating specific preview")

            # Get the case ID for preview generation
            case = read_case_by_token(token)
            if case:
                case_id = case.get('id')
                pasted_text = payload.get('pastedText')

                # Generate preview for pasted text in background thread
                def generate_pasted_preview():
                    time.sleep(1)  # Brief delay to ensure case is saved
                    success = generate_preview_for_pasted_text(case_id, token, pasted_text, payload)
                    if success:
                        logger.info(f"Successfully generated preview for pasted text case: {token[:8]}...")
                    else:
                        logger.error(f"Failed to generate preview for pasted text case: {token[:8]}...")

                preview_thread = threading.Thread(target=generate_pasted_preview)
                preview_thread.daemon = True
                preview_thread.start()
            else:
                logger.warning(f"Could not find case for token {token[:8]}... to generate pasted text preview")

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


@app.route('/api/case-preview/<case_id>', methods=['GET', 'OPTIONS'])
def get_case_preview(case_id):
    """Get the active case preview for frontend display."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Validate case_id format (basic UUID check)
        if not case_id or len(case_id) < 10:
            return jsonify({'error': 'Invalid case ID format'}), 400

        logger.info(f"Fetching preview for case ID: {case_id}")

        # Get the case to check token for document status
        case = read_case_by_id(case_id)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        token = case.get('token')

        # Check document processing status
        all_documents = fetch_any_documents_status_by_token(token)
        has_processing_documents = any(doc.get('status') in ['pending', 'processing'] for doc in all_documents)

        # Get the active preview from database
        preview_data = read_active_preview(case_id)

        # NEW: If documents are processing and no preview exists, return waiting state
        if has_processing_documents and not preview_data:
            logger.info(f"Documents still processing for case {case_id}, no preview yet - returning waiting state")
            return jsonify({
                'status': 'waiting',
                'message': 'Your documents are being analyzed. The final preview will be ready shortly.',
                'case_id': case_id,
                'doc_status': 'processing',
                'processing_documents': len(
                    [doc for doc in all_documents if doc.get('status') in ['pending', 'processing']]),
                'estimated_time_remaining': '2-3 minutes'
            }), 202  # 202 Accepted indicates processing

        if not preview_data:
            return jsonify({
                'error': 'No active preview found for this case',
                'case_id': case_id
            }), 404

        # Extract the structured data for frontend
        preview_content = preview_data.get('preview_content', {})
        preview_json = preview_content.get('preview_json', {})

        # Build frontend response with all the data the frontend needs
        frontend_response = {
            'status': 'ready',  # NEW: Indicate preview is ready
            'case_id': case_id,
            'preview_id': preview_data.get('id'),
            'preview_snippet': preview_data.get('preview_snippet'),
            'generated_at': preview_content.get('generated_at'),
            'doc_status': preview_content.get('doc_summary', {}).get('doc_status', 'none'),
            'doc_count': preview_content.get('doc_summary', {}).get('doc_count', 0),

            # Main preview content from preview_json
            'headline': preview_json.get('headline', 'HOA Case Analysis'),
            'why_now': preview_json.get('why_now', ''),
            'your_situation': preview_json.get('your_situation', {}),
            'risk_if_wrong': preview_json.get('risk_if_wrong', []),
            'what_you_get_when_you_unlock': preview_json.get('what_you_get_when_you_unlock', []),
            'critical_detail_locked': preview_json.get('critical_detail_locked', {}),
            'cta': preview_json.get('cta', {
                'primary': 'Unlock full response package',
                'secondary': 'See detailed analysis'
            }),

            # Full preview text (markdown)
            'preview_text': preview_content.get('preview_text', ''),

            # Metadata
            'model': preview_data.get('model', 'gpt-4o-mini'),
            'prompt_version': preview_data.get('prompt_version', 'v2.0_sales'),
            'is_active': preview_data.get('is_active', True)
        }

        logger.info(f"Successfully retrieved preview for case {case_id}")
        return jsonify(frontend_response), 200

    except Exception as e:
        error_msg = f"Error fetching case preview: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/case-preview/by-token/<token>', methods=['GET', 'OPTIONS'])
def get_case_preview_by_token(token):
    """Get case preview by token (alternative endpoint for frontend convenience)."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Validate token format
        if not token or len(token) < 10:
            return jsonify({'error': 'Invalid token format'}), 400

        logger.info(f"Fetching case by token: {token[:8]}...")

        # First get the case to find the case_id
        case = read_case_by_token(token)
        if not case:
            return jsonify({
                'error': 'Case not found for token',
                'token': token[:8] + '...'
            }), 404

        case_id = case.get('id')
        if not case_id:
            return jsonify({'error': 'Case ID not found'}), 404

        # Now get the preview using the case_id
        return get_case_preview(case_id)

    except Exception as e:
        error_msg = f"Error fetching case preview by token: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/case-analysis', methods=['POST', 'OPTIONS'])
def case_analysis():
    """Generate HOA case analysis using OpenAI - delegates to run_case_analysis function"""
    logger.info("Case analysis API endpoint called")

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
        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = body.get('token')

        if not token:
            response = jsonify({'error': 'token is required'})
            return add_cors_headers(response), 400

        # Call the core case analysis function
        result = run_case_analysis(token)

        if result.get('ok'):
            response = jsonify(result)
            return add_cors_headers(response), 200
        else:
            error = result.get('error', 'Unknown error')
            # Determine appropriate status code
            if 'not found' in error.lower():
                status_code = 404
            elif 'not unlocked' in error.lower():
                status_code = 402
            elif 'env vars' in error.lower():
                status_code = 500
            else:
                status_code = 500

            response = jsonify({'error': error})
            return add_cors_headers(response), status_code

    except Exception as e:
        logger.error(f'case-analysis error: {str(e)}')
        response = jsonify({'error': str(e) or 'server error'})
        return add_cors_headers(response), 500


def replace_date_placeholders(text: str, payload: Dict[str, Any] = None) -> str:
    """
    Replace date placeholders in generated text with actual dates.

    Replaces:
    - [DATE SENT], [TODAY'S DATE], [CURRENT DATE], [DATE], [TODAY] -> today's date
    - [NOTICE DATE] -> notice date from payload if available

    Leaves other placeholders like [INSERT MODEL/MAKE], [YOUR NAME] as-is.
    """
    if not text:
        return text

    # Format today's date as "Month DD, YYYY"
    today = datetime.now()
    today_formatted = today.strftime("%B %d, %Y")  # e.g., "February 23, 2026"

    # Replace today's date placeholders (case-insensitive)
    today_patterns = [
        r'\[DATE SENT\]',
        r'\[DATE\s+SENT\]',
        r'\[TODAY\'?S?\s+DATE\]',
        r'\[CURRENT\s+DATE\]',
        r'\[TODAY\]',
        r'\[SEND\s+DATE\]',
        r'\[LETTER\s+DATE\]',
        r'\[DATE\s+OF\s+LETTER\]',
        r'\[DATE\s+HERE\]',
    ]

    for pattern in today_patterns:
        text = re.sub(pattern, today_formatted, text, flags=re.IGNORECASE)

    # Try to extract and use notice date from payload for deadline calculations
    if payload:
        notice_date_str = (
            payload.get('noticeDate') or
            payload.get('notice_date') or
            payload.get('violationDate') or
            payload.get('violation_date')
        )
        if notice_date_str:
            try:
                # Parse various date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%B %d, %Y']:
                    try:
                        notice_date = datetime.strptime(notice_date_str, fmt)
                        notice_formatted = notice_date.strftime("%B %d, %Y")
                        text = re.sub(r'\[NOTICE\s+DATE\]', notice_formatted, text, flags=re.IGNORECASE)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

    return text


def fill_letter_placeholders(text: str, payload: Dict[str, Any] = None) -> str:
    """
    Fill common letter placeholders with data from the case payload.
    Used for email exchanges and generated letters.

    Pre-fills:
    - Property address (full address including city, state, ZIP)
    - Owner name
    - HOA name
    - Today's date

    Leaves brackets for genuinely unknown fields.
    """
    if not text or not payload:
        return text

    # Extract available information from payload
    property_address = (
        payload.get('propertyAddress') or
        payload.get('property_address') or
        ''
    )

    owner_name = (
        payload.get('ownerName') or
        payload.get('owner_name') or
        ''
    )

    hoa_name = (
        payload.get('hoaName') or
        payload.get('hoa_name') or
        ''
    )

    hoa_address = (
        payload.get('hoaAddress') or
        payload.get('hoa_address') or
        payload.get('managementAddress') or
        payload.get('management_address') or
        ''
    )

    # Format today's date
    today_formatted = datetime.now().strftime("%B %d, %Y")

    # Replace date placeholders
    date_patterns = [
        r'\[DATE SENT\]',
        r'\[DATE\s+SENT\]',
        r'\[TODAY\'?S?\s+DATE\]',
        r'\[CURRENT\s+DATE\]',
        r'\[TODAY\]',
        r'\[DATE\]',
        r'\[SEND\s+DATE\]',
        r'\[LETTER\s+DATE\]',
        r'\[DATE\s+OF\s+LETTER\]',
        r'\[DATE\s+HERE\]',
    ]
    for pattern in date_patterns:
        text = re.sub(pattern, today_formatted, text, flags=re.IGNORECASE)

    # Replace sender/owner address placeholders
    if property_address:
        address_patterns = [
            r'\[YOUR\s+(?:FULL\s+)?ADDRESS\]',
            r'\[SENDER\'?S?\s+(?:FULL\s+)?ADDRESS\]',
            r'\[PROPERTY\s+ADDRESS\]',
            r'\[INSERT\s+(?:YOUR\s+|PROPERTY\s+)?ADDRESS\]',
            r'\[ADDRESS\]',
            r'\[CITY,?\s*STATE,?\s*ZIP\]',
            r'\[CITY\s+STATE\s+ZIP\]',
            r'\[YOUR\s+CITY,?\s*STATE,?\s*ZIP\]',
            r'\[INSERT\s+CITY,?\s*STATE,?\s*ZIP\]',
        ]
        for pattern in address_patterns:
            text = re.sub(pattern, property_address, text, flags=re.IGNORECASE)

    # Replace owner name placeholders
    if owner_name:
        name_patterns = [
            r'\[YOUR\s+(?:FULL\s+)?NAME\]',
            r'\[OWNER\'?S?\s+(?:FULL\s+)?NAME\]',
            r'\[SENDER\'?S?\s+NAME\]',
            r'\[INSERT\s+(?:YOUR\s+|OWNER\s+)?NAME\]',
            r'\[NAME\]',
            r'\[HOMEOWNER\s+NAME\]',
        ]
        for pattern in name_patterns:
            text = re.sub(pattern, owner_name, text, flags=re.IGNORECASE)

    # Replace HOA/recipient placeholders
    if hoa_name:
        hoa_patterns = [
            r'\[HOA\s+NAME\]',
            r'\[INSERT\s+HOA(?:\s+NAME)?\]',
            r'\[RECIPIENT\s+NAME\]',
            r'\[MANAGEMENT\s+COMPANY(?:\s+NAME)?\]',
        ]
        for pattern in hoa_patterns:
            text = re.sub(pattern, hoa_name, text, flags=re.IGNORECASE)

    if hoa_address:
        hoa_addr_patterns = [
            r'\[HOA\s+ADDRESS\]',
            r'\[INSERT\s+HOA\s+ADDRESS\]',
            r'\[RECIPIENT\s+ADDRESS\]',
            r'\[MANAGEMENT\s+(?:COMPANY\s+)?ADDRESS\]',
        ]
        for pattern in hoa_addr_patterns:
            text = re.sub(pattern, hoa_address, text, flags=re.IGNORECASE)

    return text


def fill_compliance_placeholders(text: str, payload: Dict[str, Any] = None) -> str:
    """
    Fill [INSERT ...] placeholders in compliance letters where info exists in payload.

    Pre-fills:
    - Appliance/equipment type (dryer, HVAC, etc.)
    - Compliance action taken
    - Property address
    - Owner name
    - HOA name

    Leaves brackets for genuinely unknown fields like vendor name, model number, inspection date.
    """
    if not text or not payload:
        return text

    # First apply common letter placeholders
    text = fill_letter_placeholders(text, payload)

    # Extract available information from payload
    appliance_type = (
        payload.get('applianceType') or
        payload.get('appliance_type') or
        payload.get('dryerType') or
        payload.get('dryer_type') or
        payload.get('equipmentType') or
        payload.get('equipment_type')
    )

    appliance_description = (
        payload.get('applianceDescription') or
        payload.get('appliance_description') or
        payload.get('equipmentDescription') or
        payload.get('equipment_description')
    )

    compliance_action = (
        payload.get('complianceAction') or
        payload.get('compliance_action') or
        payload.get('actionTaken') or
        payload.get('action_taken') or
        payload.get('remedyTaken') or
        payload.get('remedy_taken')
    )

    property_address = (
        payload.get('propertyAddress') or
        payload.get('property_address')
    )

    owner_name = (
        payload.get('ownerName') or
        payload.get('owner_name')
    )

    hoa_name = (
        payload.get('hoaName') or
        payload.get('hoa_name')
    )

    violation_type = (
        payload.get('violationType') or
        payload.get('violation_type') or
        payload.get('noticeType') or
        payload.get('notice_type')
    )

    case_description = (
        payload.get('caseDescription') or
        payload.get('case_description') or
        payload.get('description')
    )

    # Replace appliance/equipment placeholders
    if appliance_type:
        text = re.sub(r'\[INSERT\s+(?:APPLIANCE|EQUIPMENT|DRYER|HVAC)\s*(?:TYPE)?\]',
                      appliance_type, text, flags=re.IGNORECASE)
        text = re.sub(r'\[(?:APPLIANCE|EQUIPMENT|DRYER|HVAC)\s*(?:TYPE)?\]',
                      appliance_type, text, flags=re.IGNORECASE)

    if appliance_description:
        text = re.sub(r'\[INSERT\s+(?:APPLIANCE|EQUIPMENT)\s+DESCRIPTION\]',
                      appliance_description, text, flags=re.IGNORECASE)
        text = re.sub(r'\[(?:APPLIANCE|EQUIPMENT)\s+DESCRIPTION\]',
                      appliance_description, text, flags=re.IGNORECASE)

    # Replace compliance action placeholders
    if compliance_action:
        text = re.sub(r'\[INSERT\s+(?:COMPLIANCE\s+)?ACTION(?:\s+TAKEN)?\]',
                      compliance_action, text, flags=re.IGNORECASE)
        text = re.sub(r'\[(?:COMPLIANCE\s+)?ACTION(?:\s+TAKEN)?\]',
                      compliance_action, text, flags=re.IGNORECASE)
        text = re.sub(r'\[INSERT\s+(?:REMEDY|WORK|REPAIR)(?:\s+TAKEN|COMPLETED|DONE)?\]',
                      compliance_action, text, flags=re.IGNORECASE)

    # Replace property/owner/HOA placeholders
    if property_address:
        text = re.sub(r'\[INSERT\s+(?:PROPERTY\s+)?ADDRESS\]',
                      property_address, text, flags=re.IGNORECASE)
        text = re.sub(r'\[(?:PROPERTY\s+)?ADDRESS\]',
                      property_address, text, flags=re.IGNORECASE)

    if owner_name:
        text = re.sub(r'\[INSERT\s+(?:YOUR\s+|OWNER\s+)?NAME\]',
                      owner_name, text, flags=re.IGNORECASE)
        text = re.sub(r'\[(?:YOUR\s+|OWNER\s+)?NAME\]',
                      owner_name, text, flags=re.IGNORECASE)

    if hoa_name:
        text = re.sub(r'\[INSERT\s+HOA(?:\s+NAME)?\]',
                      hoa_name, text, flags=re.IGNORECASE)
        text = re.sub(r'\[HOA(?:\s+NAME)?\]',
                      hoa_name, text, flags=re.IGNORECASE)

    # Use violation type or case description for generic description placeholders
    description_fill = violation_type or case_description
    if description_fill:
        text = re.sub(r'\[INSERT\s+VIOLATION(?:\s+TYPE|DESCRIPTION)?\]',
                      description_fill, text, flags=re.IGNORECASE)

    return text


def parse_date_from_text(text: str) -> Optional[datetime]:
    """
    Try to extract a date from text like "December 27, 2025" or "12/27/2025".
    Returns datetime object if found, None otherwise.
    """
    if not text:
        return None

    # Common date patterns
    date_patterns = [
        (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', '%B %d %Y'),  # "December 27, 2025"
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%m/%d/%Y'),     # "12/27/2025"
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', '%m-%d-%Y'),     # "12-27-2025"
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),     # "2025-12-27"
    ]

    for pattern, _ in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(0).replace(',', '')
            # Try parsing with various formats
            for fmt in ['%B %d %Y', '%b %d %Y', '%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

    return None


def process_action_plan_deadlines(action_plan: List[str]) -> List[str]:
    """
    Process action plan items to flag any deadlines that have already passed.

    If a deadline date is in the past, reframes the action item to indicate urgency:
    - "Before December 27, 2025" -> "OVERDUE (was December 27, 2025) — Act immediately"
    """
    if not action_plan:
        return action_plan

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    processed = []

    for item in action_plan:
        if not isinstance(item, str):
            processed.append(item)
            continue

        # Check if item contains a date
        extracted_date = parse_date_from_text(item)

        if extracted_date and extracted_date < today:
            # This deadline has passed
            days_overdue = (today - extracted_date).days

            # Check if it mentions "before", "by", "deadline", etc.
            deadline_indicators = ['before', 'by', 'deadline', 'due', 'must', 'no later than']
            is_deadline_item = any(ind in item.lower() for ind in deadline_indicators)

            if is_deadline_item:
                # Reframe the item to indicate it's overdue
                date_str = extracted_date.strftime("%B %d, %Y")

                if days_overdue == 1:
                    overdue_text = "1 day overdue"
                elif days_overdue < 7:
                    overdue_text = f"{days_overdue} days overdue"
                elif days_overdue < 30:
                    weeks = days_overdue // 7
                    overdue_text = f"{weeks} week{'s' if weeks > 1 else ''} overdue"
                else:
                    overdue_text = f"{days_overdue} days overdue"

                # Prepend overdue notice and add urgency
                new_item = f"⚠️ OVERDUE ({overdue_text}, was {date_str}): {item} — Act immediately to establish good faith."
                processed.append(new_item)
            else:
                # Date is in the past but not clearly a deadline
                processed.append(item)
        else:
            processed.append(item)

    return processed


def process_drafts_date_placeholders(outputs: Dict[str, Any], payload: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Post-process all draft letters to:
    1. Replace date placeholders with actual dates
    2. Fill [INSERT] placeholders in compliance letter from payload data
    3. Flag overdue deadlines in action plan
    """
    if not outputs:
        return outputs

    # Process drafts
    if 'drafts' in outputs and isinstance(outputs['drafts'], dict):
        for key in outputs['drafts']:
            if isinstance(outputs['drafts'][key], str):
                # Replace date placeholders
                outputs['drafts'][key] = replace_date_placeholders(outputs['drafts'][key], payload)

                # For compliance draft, also fill [INSERT] placeholders from payload
                if key == 'compliance':
                    outputs['drafts'][key] = fill_compliance_placeholders(outputs['drafts'][key], payload)

    # Process letter_summary if it contains placeholders
    if 'letter_summary' in outputs and isinstance(outputs['letter_summary'], str):
        outputs['letter_summary'] = replace_date_placeholders(outputs['letter_summary'], payload)

    # Process action_plan to flag overdue deadlines
    if 'action_plan' in outputs and isinstance(outputs['action_plan'], list):
        outputs['action_plan'] = process_action_plan_deadlines(outputs['action_plan'])

    return outputs


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


@app.route('/api/case-data', methods=['GET', 'OPTIONS'])
def get_case_data():
    """Get case data by token for frontend display."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Get token from query parameters
        token = request.args.get('token', '').strip()

        if not token:
            return jsonify({'error': 'token is required'}), 400

        logger.info(f"Fetching case data for token: {token[:8]}...")

        # Fetch case data from Supabase
        case = read_case_by_token(token)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        # Also fetch the case outputs (full analysis) if available
        case_outputs = None
        try:
            outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
            outputs_params = {
                'case_token': f'eq.{token}',
                'select': '*'
            }
            outputs_headers = supabase_headers()

            outputs_response = requests.get(outputs_url, params=outputs_params, headers=outputs_headers,
                                            timeout=TIMEOUT)
            outputs_response.raise_for_status()
            outputs_data = outputs_response.json()

            if outputs_data:
                case_outputs = outputs_data[0]
                logger.info(f"Found case outputs for token {token[:8]}...")
            else:
                logger.info(f"No case outputs found for token {token[:8]}...")

        except Exception as e:
            logger.warning(f"Error fetching case outputs for {token[:8]}...: {str(e)}")
            # Continue without outputs - not critical

        # Build response with case data and outputs if available
        response_data = {
            'token': case.get('token'),
            'status': case.get('status', 'preview'),
            'created_at': case.get('created_at'),
            'updated_at': case.get('updated_at'),
            'payload': case.get('payload', {}),
            'outputs': case_outputs.get('outputs') if case_outputs else None,
            'outputs_status': case_outputs.get('status') if case_outputs else None,
            'fine_per_day': case_outputs.get('fine_per_day') if case_outputs else None,
            'fine_start_date': case_outputs.get('fine_start_date') if case_outputs else None
        }

        logger.info(f"Successfully retrieved case data for token {token[:8]}...")
        return jsonify(response_data), 200

    except Exception as e:
        error_msg = f"Error fetching case data: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


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


def run_case_analysis(token: str) -> Dict[str, Any]:
    """
    Run case analysis for a given token. This is the core logic extracted from the API endpoint
    so it can be called directly from the Stripe webhook after payment.

    Returns a dict with keys: ok, status, cached, outputs, error (if any)
    """
    logger.info(f"Running case analysis for token: {token[:8]}...")

    try:
        # Validate required environment variables
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
            logger.error("Missing env vars for case analysis")
            return {'ok': False, 'error': 'Missing env vars'}

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
            logger.error(f'Database error reading case: {str(e)}')
            return {'ok': False, 'error': f'Database error reading case: {str(e)}'}

        if not case_row:
            logger.error(f"Case not found for token: {token[:8]}...")
            return {'ok': False, 'error': 'Case not found'}

        if not case_row.get('unlocked'):
            logger.error(f"Case is not unlocked for token: {token[:8]}...")
            return {'ok': False, 'error': 'Case is not unlocked'}

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
            logger.error(f'Docs lookup error: {str(e)}')
            docs = []

        docs_newest = newest_updated_at(docs)

        # Consider usable if it has text, even if status still says "processing"
        usable_docs = [
            d for d in docs
            if isinstance(d.get('extracted_text'), str) and d['extracted_text'].strip()
        ]

        logger.info(f'DOCS DEBUG for {token[:8]}...: count={len(docs)}, usable={len(usable_docs)}')

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

        # 2) Check if outputs already exist and are ready
        outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        outputs_params = {
            'case_token': f'eq.{token}',
            'select': 'case_token,status,outputs,error,updated_at'
        }
        outputs_headers = supabase_headers()

        try:
            outputs_response = requests.get(outputs_url, params=outputs_params, headers=outputs_headers,
                                            timeout=TIMEOUT)
            outputs_response.raise_for_status()
            existing_outputs = outputs_response.json()
            existing_out = existing_outputs[0] if existing_outputs else None
        except Exception as e:
            logger.error(f'Database error reading outputs: {str(e)}')
            return {'ok': False, 'error': f'Database error reading outputs: {str(e)}'}

        out_updated = safe_iso(existing_out.get('updated_at') if existing_out else None)
        docs_are_newer = (
                docs_newest and out_updated and
                datetime.fromisoformat(docs_newest) > datetime.fromisoformat(out_updated)
        )

        if (existing_out and existing_out.get('status') == 'ready' and
                existing_out.get('outputs') and not docs_are_newer):
            logger.info(f"Returning cached outputs for token: {token[:8]}...")
            return {
                'ok': True,
                'status': 'ready',
                'cached': True,
                'outputs': existing_out['outputs']
            }

        payload = case_row.get('payload') or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except:
                payload = {}
        draft_titles = get_draft_titles(payload)

        # 2b) Get state-specific statute context for prompt injection
        statute_context = None
        try:
            case_state = extract_state_from_payload(payload)
            violation_type = extract_violation_type_from_payload(payload)
            if case_state:
                statute_context = get_statute_context(case_state, violation_type)
                if statute_context:
                    logger.info(f"Statute context loaded for {case_state}, token: {token[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to get statute context for token {token[:8]}...: {str(e)}")
            statute_context = None  # Graceful degradation - continue without statute context

        # 3) Anthropic Claude API call for case analysis (strict JSON schema)
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

        # Build statute context section for user message
        statute_section = ""
        if statute_context:
            statute_section = f"\n\n{statute_context}\n\n"

        system_prompt_analysis = """
You generate HOA dispute assistance for a homeowner.
This is educational drafting help, not legal advice.


LEGAL CITATION REQUIREMENTS:
- When state-specific statute information is provided, you MUST cite those specific statutes in your drafts.
- Reference the exact statute name and section numbers in your response letters.
- Mention procedural protections the homeowner is entitled to under state law.
- Include homeowner rights and notice requirements from the statute reference.

OUTPUT RULES (CRITICAL):
- ONLY "summary_html" may contain HTML.
- summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.
- ALL drafts (clarification/extension/compliance) MUST be PLAIN TEXT ONLY:
  - NO HTML tags
  - Use newlines with \\n
  - Bullets: use "- item" lines
- Return STRICT JSON that matches the schema exactly.

SUMMARY_HTML REQUIREMENTS (CRITICAL - THIS IS A COMPACT FACT SHEET, NOT A DETAILED ANALYSIS):
- summary_html must be a SHORT, COMPACT fact sheet with exactly 5-7 bullet points maximum.
- It should ONLY include these facts (if available):
  1. Notice date (when the HOA sent the notice)
  2. Who sent it (HOA name/management company)
  3. Core allegation (one line - what the violation is)
  4. Key deadline(s) with SPECIFIC DATES (not "within 30 days" - calculate actual date)
  5. Fines/amounts at risk
  6. Owner's stated position (comply, dispute, need extension, etc.)
- DO NOT include: action plans, questions to ask, lowest-cost path, detailed analysis, or guidance.
- Keep it under 150 words total. This is a quick-reference fact sheet, not the full analysis.

DRAFT QUALITY REQUIREMENTS:
- Each draft must be a complete, ready-to-send letter.
- MUST directly quote or reference concrete facts from the extracted documents when available
  (deadlines, email addresses, paragraph citations, dollar amounts, dates, etc.).
- When state law is referenced, cite specific statutes to strengthen the homeowner's position.
- Use actual dates instead of placeholders - today's date should be written out (e.g., "February 23, 2026").
- Each must include:
  - Subject line
  - Short opening with today's date
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

FORMATTING RULES — STRICTLY FOLLOW:
- Output plain text only. No markdown of any kind.
- No ** bold **, no ### headers, no # headings, no * bullets, no _ italics_
- No emojis. No symbols used as decorators
- Use ALL CAPS for section headings
- Use plain numbered lists (1. 2. 3.) or dashes (-) for bullets
- Separate sections with a blank line only
- The output will be rendered and copied as a plain text letter
"""

        user_content = (
            f"Case payload JSON:\n{json.dumps(payload)}\n\n"
            f"Document fingerprint (debug):\n{json.dumps(doc_fingerprint)}\n\n"
            f"Extracted documents:\n{docs_block}\n"
            f"{statute_section}"
            f"Draft types for this case (MUST follow exactly):\n"
            f"- drafts.clarification MUST be: \"{draft_titles['clarification']}\"\n"
            f"- drafts.extension MUST be: \"{draft_titles['extension']}\"\n"
            f"- drafts.compliance MUST be: \"{draft_titles['compliance']}\"\n\n"
            f"Also include draft_titles using these exact same strings.\n\n"
            f"summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.\n"
            f"Drafts must be PLAIN TEXT ONLY with \\n, and must NOT include any HTML tags.\n\n"
            f"Make this feel like a $30 deliverable: concrete, specific, complete.\n"
        )

        # Build JSON schema instruction for Claude
        schema_instruction = f"""
You must respond with ONLY valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}

Do not include any text before or after the JSON. The response must be parseable as JSON.
"""

        anthropic_messages = [
            {"role": "user", "content": user_content + "\n\n" + schema_instruction}
        ]

        # Make Anthropic Claude API call for better legal writing quality
        # Includes retry logic with exponential backoff for 429/529 errors
        try:
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not configured")

            max_retries = 5
            base_delay = 10  # seconds - longer initial delay to let API recover
            anthropic_response = None

            for attempt in range(max_retries):
                logger.info(f"Calling Claude claude-sonnet-4-6 for case analysis for token: {token[:8]}... (attempt {attempt + 1}/{max_retries})")

                anthropic_response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers={
                        'x-api-key': ANTHROPIC_API_KEY,
                        'Content-Type': 'application/json',
                        'anthropic-version': '2023-06-01'
                    },
                    json={
                        'model': 'claude-sonnet-4-6',
                        'max_tokens': 8192,
                        'system': system_prompt_analysis,
                        'messages': anthropic_messages
                    },
                    timeout=(10, 180)  # 10s connect, 180s read for longer legal analysis
                )

                # Check if we should retry (429 rate limit, 529 overloaded, 503 service unavailable)
                if anthropic_response.status_code in [429, 529, 503]:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter: 10-12s, 20-24s, 40-48s, 80-96s
                        import random
                        jitter = random.uniform(0, 0.2)  # Add up to 20% jitter
                        delay = base_delay * (2 ** attempt) * (1 + jitter)
                        logger.warning(f"Claude API returned {anthropic_response.status_code} (overloaded/rate-limited), retrying in {delay:.1f}s for token {token[:8]}...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Claude API still returning {anthropic_response.status_code} after {max_retries} attempts for token {token[:8]}...")
                        break
                else:
                    # Either success or a non-retryable error
                    break

            if not anthropic_response.ok:
                error_text = anthropic_response.text
                logger.error(f'Claude API call failed: {anthropic_response.status_code}, {error_text}')

                # Update outputs table with error
                error_data = {
                    'case_token': token,
                    'status': 'error',
                    'error': error_text or 'Claude API call failed',
                    'updated_at': datetime.utcnow().isoformat()
                }
                try:
                    error_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
                    error_headers = supabase_headers()
                    # Use UPSERT to ensure error is saved even if no row exists yet
                    error_headers['Prefer'] = 'resolution=merge-duplicates'
                    requests.post(error_url, headers=error_headers, json=error_data,
                                  timeout=TIMEOUT)
                except Exception:
                    pass  # Best effort

                return {'ok': False, 'error': f'Claude API call failed: {error_text}'}

            # Parse Claude response
            claude_json = anthropic_response.json()
            content = claude_json.get('content', [])
            if content and content[0].get('type') == 'text':
                response_text = content[0].get('text', '')
                # Try to extract JSON from the response
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    response_text = json_match.group(1)
                try:
                    structured = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Claude response as JSON for token {token[:8]}...")
                    structured = None
            else:
                structured = None

            if structured:
                # Post-process drafts to replace date placeholders with actual dates
                structured = process_drafts_date_placeholders(structured, payload)

                outputs_to_store = {
                    **structured,
                    'draft_titles': structured.get('draft_titles', draft_titles),
                    'doc_fingerprint': doc_fingerprint
                }
            else:
                outputs_to_store = {
                    'raw': claude_json,
                    'draft_titles': draft_titles,
                    'doc_fingerprint': doc_fingerprint
                }

            # Save successful outputs using UPSERT (insert if new, update if exists)
            success_data = {
                'case_token': token,
                'status': 'ready',
                'outputs': outputs_to_store,
                'error': None,
                'model': 'claude-sonnet-4-6',
                'prompt_version': 'v6_compliance_prefill_overdue_flags',
                'updated_at': datetime.utcnow().isoformat()
            }

            try:
                save_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
                save_headers = supabase_headers()
                # Use UPSERT (insert or update) - this handles both new and existing rows
                save_headers['Prefer'] = 'resolution=merge-duplicates'
                save_response = requests.post(save_url, headers=save_headers, json=success_data,
                                              timeout=TIMEOUT)
                save_response.raise_for_status()
                logger.info(f"Successfully saved case outputs for token: {token[:8]}...")
            except Exception as e:
                logger.error(f'Failed to save outputs: {str(e)}')
                return {'ok': False, 'error': f'Failed saving outputs: {str(e)}'}

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

            # Fine extraction step - extract fine_per_day and fine_start_date using Claude Haiku
            # This runs after main analysis is saved, so failures won't break the main flow
            extracted_fine_per_day = None
            extracted_fine_start_date = None
            try:
                if ANTHROPIC_API_KEY and structured:
                    # Build full analysis text from the structured output
                    analysis_parts = []
                    if structured.get('summary_html'):
                        analysis_parts.append(f"Summary: {structured['summary_html']}")
                    if structured.get('letter_summary'):
                        analysis_parts.append(f"Letter Summary: {structured['letter_summary']}")
                    if structured.get('risks_and_deadlines'):
                        rd = structured['risks_and_deadlines']
                        if rd.get('deadlines'):
                            analysis_parts.append(f"Deadlines: {', '.join(rd['deadlines'])}")
                        if rd.get('risks'):
                            analysis_parts.append(f"Risks: {', '.join(rd['risks'])}")
                    if structured.get('drafts'):
                        for draft_key, draft_text in structured['drafts'].items():
                            analysis_parts.append(f"Draft ({draft_key}): {draft_text}")

                    # Also include the docs_block which has the original notice text
                    analysis_parts.append(f"Original documents: {docs_block[:8000]}")

                    full_analysis_text = "\n\n".join(analysis_parts)

                    # Call Claude Haiku to extract fine information
                    fine_extraction_prompt = f"""From the following HOA case analysis, extract two values and return ONLY a valid JSON object with no explanation, no markdown, no extra text:
{{ "fine_per_day": <number or null>, "fine_start_date": "YYYY-MM-DD or null" }}

fine_per_day: the daily fine dollar amount stated in the notice (e.g. 100)
fine_start_date: the earliest specific date fines begin accruing per the notice

If either value cannot be determined from the text, return null for that field.

Case analysis text:
{full_analysis_text[:12000]}"""

                    # Retry logic for fine extraction (handles 429/529/503 errors)
                    haiku_response = None
                    for haiku_attempt in range(3):
                        haiku_response = requests.post(
                            'https://api.anthropic.com/v1/messages',
                            headers={
                                'x-api-key': ANTHROPIC_API_KEY,
                                'Content-Type': 'application/json',
                                'anthropic-version': '2023-06-01'
                            },
                            json={
                                'model': 'claude-haiku-4-5-20251001',
                                'max_tokens': 256,
                                'messages': [{'role': 'user', 'content': fine_extraction_prompt}]
                            },
                            timeout=(10, 30)
                        )

                        if haiku_response.status_code in [429, 529, 503]:
                            if haiku_attempt < 2:
                                delay = 2 * (2 ** haiku_attempt)  # 2s, 4s
                                logger.warning(f"Fine extraction: Haiku returned {haiku_response.status_code}, retrying in {delay}s...")
                                time.sleep(delay)
                                continue
                        break

                    if haiku_response and haiku_response.ok:
                        haiku_json = haiku_response.json()
                        haiku_content = haiku_json.get('content', [])
                        if haiku_content and haiku_content[0].get('type') == 'text':
                            fine_text = haiku_content[0].get('text', '').strip()

                            # Strip markdown fences if present
                            if fine_text.startswith('```'):
                                fine_text = re.sub(r'^```(?:json)?\s*', '', fine_text)
                                fine_text = re.sub(r'\s*```$', '', fine_text)

                            # Parse JSON
                            fine_data = json.loads(fine_text)
                            fine_per_day = fine_data.get('fine_per_day')
                            fine_start_date = fine_data.get('fine_start_date')

                            # Store extracted values for return
                            extracted_fine_per_day = fine_per_day
                            extracted_fine_start_date = fine_start_date

                            # Update dmhoa_case_outputs if we have values
                            if fine_per_day is not None or fine_start_date is not None:
                                fine_update_data = {}
                                if fine_per_day is not None:
                                    fine_update_data['fine_per_day'] = fine_per_day
                                if fine_start_date is not None:
                                    fine_update_data['fine_start_date'] = fine_start_date

                                fine_update_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
                                fine_update_params = {'case_token': f'eq.{token}'}
                                fine_update_headers = supabase_headers()

                                requests.patch(
                                    fine_update_url,
                                    params=fine_update_params,
                                    headers=fine_update_headers,
                                    json=fine_update_data,
                                    timeout=TIMEOUT
                                )

                                logger.info(f"Fine extraction: fine_per_day={fine_per_day} fine_start_date={fine_start_date} for token {token[:8]}...")
                            else:
                                logger.info(f"Fine extraction: no fine data found for token {token[:8]}...")
                    else:
                        logger.warning(f"Fine extraction failed: Haiku API returned {haiku_response.status_code} for token {token[:8]}...")

            except json.JSONDecodeError as e:
                logger.warning(f"Fine extraction failed: JSON parse error - {str(e)} for token {token[:8]}...")
            except Exception as e:
                logger.warning(f"Fine extraction failed: {str(e)} for token {token[:8]}...")

            # Add fine data to outputs if extracted
            if extracted_fine_per_day is not None:
                outputs_to_store['fine_per_day'] = extracted_fine_per_day
            if extracted_fine_start_date is not None:
                outputs_to_store['fine_start_date'] = extracted_fine_start_date

            logger.info(f"Case analysis completed successfully for token: {token[:8]}...")
            return {
                'ok': True,
                'status': 'ready',
                'cached': False,
                'outputs': outputs_to_store,
                'fine_per_day': extracted_fine_per_day,
                'fine_start_date': extracted_fine_start_date
            }

        except Exception as e:
            logger.error(f'Claude API error: {str(e)}')
            return {'ok': False, 'error': f'Claude API error: {str(e)}'}

    except Exception as e:
        logger.error(f'Case analysis error: {str(e)}')
        return {'ok': False, 'error': str(e)}


def send_receipt_email_direct(token: str, to_email: str, case_url: str,
                              amount_total: int = None, currency: str = "USD",
                              customer_name: str = None, stripe_session_id: str = None) -> bool:
    """
    Send receipt email directly via SMTP. This is used by the Stripe webhook
    to send emails without needing to call an external webhook.

    Returns True if email was sent successfully, False otherwise.
    """
    try:
        # Check which SMTP environment variables are available
        if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
            logger.warning(f"SMTP not configured - missing SMTP_HOST, SMTP_USER, or SMTP_PASS")
            return False

        # Personalized greeting
        greeting = f"Hi {customer_name}," if customer_name else "Hi,"

        # Format payment amount
        dollars = f"${(amount_total or 0) / 100:.2f}" if isinstance(amount_total, int) else ""
        currency_str = (currency or "USD").upper()
        payment_info = f" for {dollars} {currency_str}" if dollars else ""

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

        logger.info(f"Sending receipt email directly to {to_email} for token {token[:8]}...")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())

        logger.info(f"Receipt email sent successfully to {to_email} (token={token[:8]}...)")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Gmail authentication failed: {str(e)}")
        return False
    except Exception as e:
        logger.exception(f"Failed to send receipt email to {to_email}")
        return False


@app.route('/api/read-messages', methods=['GET', 'OPTIONS'])
def read_messages():
    """Read chat messages for a case."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Get parameters from query string
        token = request.args.get('token', '').strip()
        limit_param = request.args.get('limit', '50')

        # Parse and validate limit (min 1, max 200, default 50)
        try:
            limit = max(1, min(int(limit_param) or 50, 200))
        except (ValueError, TypeError):
            limit = 50

        if not token:
            return jsonify({'error': 'token is required'}), 400

        logger.info(f"Reading messages for token: {token[:8]}..., limit: {limit}")

        # Check if case exists
        case = read_case_by_token(token)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        # Check if case is paid/unlocked - only return messages if case is paid
        case_status = case.get('status', 'paid')
        if case_status != 'paid':
            # Return empty messages array if case is not paid/unlocked
            logger.info(f"Case {token[:8]}... not paid, returning empty messages")
            return jsonify({'ok': True, 'messages': []}), 200

        # Fetch messages for the case from dmhoa_messages table
        try:
            messages_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
            messages_params = {
                'token': f'eq.{token}',
                'select': 'id,token,role,content,created_at',
                'order': 'created_at.asc',
                'limit': str(limit)
            }
            messages_headers = supabase_headers()

            messages_response = requests.get(messages_url, params=messages_params, headers=messages_headers,
                                             timeout=TIMEOUT)
            messages_response.raise_for_status()
            messages = messages_response.json()

            logger.info(f"Found {len(messages)} messages for token {token[:8]}...")
            return jsonify({'ok': True, 'messages': messages or []}), 200

        except Exception as e:
            logger.error(f'Database error reading messages for {token[:8]}...: {str(e)}')
            return jsonify({'error': 'Database error reading messages'}), 500

    except Exception as e:
        error_msg = f"Error reading messages: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/send-message', methods=['POST', 'OPTIONS'])
def send_message():
    """Send chat message endpoint for case discussions."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Parse request body
        try:
            body = request.get_json() or {}
        except Exception:
            body = {}

        token = str(body.get('token', '')).strip()
        user_content = body.get('content', '').strip()

        if not token or not user_content:
            return jsonify({'error': 'token and content are required'}), 400

        # Limit user message length
        if len(user_content) > 1000:
            user_content = user_content[:1000]

        logger.info(f"Sending message for token: {token[:8]}..., content length: {len(user_content)}")

        # 1) Ensure case exists and is paid/unlocked
        case = read_case_by_token(token)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        case_status = case.get('status', 'paid')
        if case_status != 'paid':
            return jsonify({'error': 'Case is not unlocked'}), 402

        # 2) Save user message to database
        try:
            user_message_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
            user_message_data = {
                'token': token,
                'role': 'user',
                'content': user_content
            }
            user_message_headers = supabase_headers()
            user_message_headers['Prefer'] = 'return=representation'

            user_message_response = requests.post(user_message_url, headers=user_message_headers,
                                                  json=user_message_data, timeout=TIMEOUT)
            user_message_response.raise_for_status()
            user_message = user_message_response.json()
            user_message = user_message[0] if user_message else None

        except Exception as e:
            logger.error(f'Failed to save user message for {token[:8]}...: {str(e)}')
            return jsonify({'error': 'Failed to save user message'}), 500

        # 3) Load recent chat history (last 20 messages)
        try:
            history_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
            history_params = {
                'token': f'eq.{token}',
                'select': 'role,content,created_at',
                'order': 'created_at.asc',
                'limit': '20'
            }
            history_headers = supabase_headers()

            history_response = requests.get(history_url, params=history_params,
                                            headers=history_headers, timeout=TIMEOUT)
            history_response.raise_for_status()
            history = history_response.json() or []

        except Exception as e:
            logger.error(f'Failed to load chat history for {token[:8]}...: {str(e)}')
            return jsonify({'error': 'Failed to load chat history'}), 500

        # 4) Get case payload for context
        case_payload = case.get('payload', {})
        if isinstance(case_payload, str):
            try:
                case_payload = json.loads(case_payload)
            except:
                case_payload = {}

        # 4.5) Get preview information for additional context
        case_id = case.get('id')
        preview_info = None
        if case_id:
            preview_info = read_active_preview(case_id)

        # 4.6) Get state-specific statute context
        chat_statute_context = None
        try:
            case_state = extract_state_from_payload(case_payload)
            violation_type = extract_violation_type_from_payload(case_payload)
            if case_state:
                chat_statute_context = get_statute_context(case_state, violation_type)
                if chat_statute_context:
                    logger.info(f"Statute context loaded for chat, state={case_state}, token: {token[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to get statute context for chat, token {token[:8]}...: {str(e)}")
            chat_statute_context = None

        # 5) Generate AI response using OpenAI Chat Completions API
        try:
            system_prompt = """You are “Dispute My HOA,” an educational HOA response assistant.
This is a paid, unlocked case. The user expects complete guidance and ready-to-use drafts.
You provide drafting assistance and procedural guidance — not legal advice.

Your role is to help homeowners:
- Fully understand their HOA violation notice in plain English
- Identify exact deadlines, risks, and procedural options
- Draft complete, ready-to-send written responses
- Choose the lowest-risk path to resolve the issue without escalation

IMPORTANT SAFETY RULES:
- Do NOT present yourself as a lawyer or provide legal advice
- Do NOT guarantee outcomes
- Base guidance strictly on the documents and facts provided
- If uncertainty exists, explain it clearly and conservatively

ESCALATION & LIABILITY RULES (CRITICAL):
- Default to NON-ADMISSION language unless the user explicitly requests a compliance or admission letter
- Avoid phrases such as “I admit,” “I acknowledge the violation,” or similar
- Preserve the homeowner’s procedural rights whenever possible
- Flag language or actions that could weaken the homeowner’s position before drafting

DRAFTING RULES:
- When a letter is requested, produce a complete, ready-to-send plain-text draft
- Use calm, professional, HOA-appropriate language
- Avoid emotional, defensive, or aggressive wording
- Assume all written responses may be reviewed, logged, or used later
- Clearly label each draft by purpose (clarification, extension, compliance, hearing request)

GUIDANCE RULES:
- Explain WHEN and WHY to use each response option
- Recommend the safest sequence of actions when multiple paths exist
- Highlight deadlines, hearing rights, documentation standards, and delivery methods (email, certified mail, portal)
- Identify irreversible actions before suggesting them

QUESTION RULES:
- Ask at most 1–2 clarifying questions
- Only ask if the answer materially changes the recommended response or wording
- If documents are incomplete, explain what is missing and proceed using safe assumptions

TONE:
- Calm
- Practical
- Protective of the homeowner
- Clear and confident, but not authoritative

FORMATTING RULES — STRICTLY FOLLOW:
- Output plain text only. No markdown of any kind.
- No ** bold **, no ### headers, no # headings, no * bullets, no _ italics_
- No emojis. No symbols used as decorators
- Use ALL CAPS for section headings
- Use plain numbered lists (1. 2. 3.) or dashes (-) for bullets
- Separate sections with a blank line only
- The output will be rendered and copied as a plain text letter

Your goal is to help the homeowner resolve the issue correctly the first time,
with clear wording, proper procedure, and minimal risk — now that the case is unlocked.

"""

            # Convert history to OpenAI message format
            messages = [{'role': 'system', 'content': system_prompt}]

            # Add context about the case
            context_message = f"Case context: {json.dumps(case_payload, indent=2)}"
            messages.append({'role': 'system', 'content': context_message})

            # Add state-specific statute context if available
            if chat_statute_context:
                statute_message = f"State-specific HOA law reference for drafting:\n\n{chat_statute_context}\n\nWhen drafting responses, cite these specific statutes to strengthen the homeowner's position."
                messages.append({'role': 'system', 'content': statute_message})

            # Add preview information if available
            if preview_info:
                preview_context = f"Case preview analysis: {json.dumps(preview_info, indent=2)}"
                messages.append({'role': 'system', 'content': preview_context})

            # Add recent chat history
            for msg in history[-10:]:  # Only use last 10 messages for context
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

            openai_payload = {
                'model': 'gpt-4o-mini',
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 1000
            }

            openai_response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json=openai_payload,
                timeout=(10, 60)
            )
            openai_response.raise_for_status()

            openai_json = openai_response.json()
            assistant_text = openai_json['choices'][0]['message']['content'].strip()

            # Limit assistant response length
            if len(assistant_text) > 2000:
                assistant_text = assistant_text[:2000] + "..."

        except Exception as e:
            logger.error(f'OpenAI API error for {token[:8]}...: {str(e)}')
            assistant_text = "I'm having trouble generating a response right now. Please try again in a moment."

        # 6) Save assistant message to database
        try:
            # NEW: Apply markdown formatter to clean the response
            formatted_response = format_plain_text_response(assistant_text)

            assistant_message_url = f"{SUPABASE_URL}/rest/v1/dmhoa_messages"
            assistant_message_data = {
                'token': token,
                'role': 'assistant',
                'content': formatted_response
            }
            assistant_message_headers = supabase_headers()
            assistant_message_headers['Prefer'] = 'return=representation'

            assistant_message_response = requests.post(assistant_message_url, headers=assistant_message_headers,
                                                       json=assistant_message_data, timeout=TIMEOUT)
            assistant_message_response.raise_for_status()
            assistant_insert = assistant_message_response.json()
            assistant_message = assistant_insert[0] if assistant_insert else None

        except Exception as e:
            logger.error(f'Failed to save assistant message for {token[:8]}...: {str(e)}')
            return jsonify({'error': 'Failed to save assistant message'}), 500

        logger.info(f"Successfully processed message for token {token[:8]}...")

        return jsonify({
            'ok': True,
            'token': token,
            'user_message': user_message,
            'assistant_message': assistant_message
        }), 200

    except Exception as e:
        error_msg = f"Error sending message: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


def format_plain_text_response(text: str) -> str:
    """Remove markdown formatting symbols from AI response text."""
    if not text:
        return text

    # Remove bold markdown (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Remove italic markdown (*text*)
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'\1', text)

    # Remove bullet points made with dashes (- item)
    text = re.sub(r'^[\s]*-[\s]+', '', text, flags=re.MULTILINE)

    # Remove bullet points made with asterisks (* item)
    text = re.sub(r'^[\s]*\*[\s]+', '', text, flags=re.MULTILINE)

    # Clean up any double spaces that might result from removals
    text = re.sub(r'  +', ' ', text)

    # Clean up any multiple newlines
    text = re.sub(r'\n\n\n+', '\n\n', text)

    return text.strip()


def fetch_case_outputs(token: str) -> Optional[Dict]:
    """Fetch existing case outputs from dmhoa_case_outputs table by token."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        params = {
            'case_token': f'eq.{token}',
            'select': '*',
            'order': 'created_at.desc',
            'limit': '1'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        outputs = response.json()
        if outputs:
            logger.info(f"Found existing case outputs for token {token[:8]}...")
            return outputs[0]

        logger.info(f"No case outputs found for token {token[:8]}...")
        return None

    except Exception as e:
        logger.error(f"Failed to fetch case outputs for token {token[:8]}...: {str(e)}")
        return None


def upsert_active_preview_lock(case_id: str) -> bool:
    """Check if a preview generation is already in progress or create a lock."""
    with preview_lock:
        if case_id in preview_generation_locks:
            logger.info(f"Preview generation already in progress for case {case_id}")
            return False

        preview_generation_locks[case_id] = True
        logger.info(f"Acquired preview generation lock for case {case_id}")
        return True


def release_preview_lock(case_id: str):
    """Release the preview generation lock for a case."""
    with preview_lock:
        preview_generation_locks.pop(case_id, None)
        logger.info(f"Released preview generation lock for case {case_id}")


def send_receipt_email(token: str, to_email: str, case_url: str, amount_total: int = None,
                       currency: str = "USD", customer_name: str = None,
                       stripe_session_id: str = None) -> bool:
    """Send receipt email after successful payment."""
    try:
        # Check which SMTP environment variables are missing
        required = {
            'SMTP_HOST': SMTP_HOST,
            'SMTP_PORT': SMTP_PORT,
            'SMTP_USER': SMTP_USER,
            'SMTP_PASS': SMTP_PASS,
        }
        missing = [name for name, val in required.items() if not val]
        if missing:
            logger.error(f"SMTP not configured - missing environment variables: {missing}")
            return False

        # Personalized greeting
        greeting = f"Hi {customer_name}," if customer_name else "Hi,"

        # Format payment amount
        dollars = f"${(amount_total or 0) / 100:.2f}" if isinstance(amount_total, int) else ""
        payment_info = f" for {dollars} {currency.upper()}" if dollars else ""

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

        logger.info(f"Connecting to SMTP {SMTP_HOST}:{SMTP_PORT} to send to {to_email}")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())

        logger.info(f"Receipt email sent to {to_email} (token={token[:8]}...)")
        return True

    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Gmail authentication failed. For Gmail, you need: 1) Enable 2-Step Verification, 2) Create App Password. Error: {str(e)}"
        logger.error(error_msg)
        return False
    except Exception as e:
        logger.exception("Failed to send receipt email")
        return False


def insert_case_output(output_data: Dict) -> bool:
    """Insert a new case output record into dmhoa_case_outputs table."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        headers = supabase_headers()
        headers['Prefer'] = 'return=representation'

        # Remove None values
        output_data = {k: v for k, v in output_data.items() if v is not None}

        # Log the data being sent for debugging
        logger.info(f"Attempting to insert case output with data: {json.dumps(output_data, indent=2, default=str)}")

        response = requests.post(url, headers=headers, json=output_data, timeout=TIMEOUT)

        # Log the full response for debugging
        logger.info(f"Supabase response status: {response.status_code}")
        logger.info(f"Supabase response body: {response.text}")

        response.raise_for_status()

        result = response.json()
        if result:
            output_id = result[0]['id']
            case_token = output_data.get('case_token', 'unknown')
            logger.info(f"Inserted case output {output_id} for case_token {case_token}")
            return True

        return False

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error inserting case output: {e}")
        logger.error(f"Response status: {e.response.status_code}")
        logger.error(f"Response body: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Failed to insert case output: {str(e)}")
        return False


@app.route('/api/cases/<case_token>/email-exchange', methods=['POST', 'OPTIONS'])
def create_email_exchange(case_token):
    """Generate a counter-letter response to an HOA message."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        # Validate case_token
        if not case_token or not case_token.strip():
            return jsonify({'error': 'case_token is required'}), 400

        case_token = case_token.strip()
        logger.info(f"Creating email exchange for case_token: {case_token[:8]}...")

        # 1) Ensure case exists and is paid/unlocked
        case = read_case_by_token(case_token)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        case_status = case.get('status', '')
        if case_status != 'paid':
            return jsonify({'error': 'Case is not unlocked. Payment required.'}), 403

        # 2) Validate request body
        data = request.get_json() or {}
        hoa_message = (data.get('hoa_message') or '').strip()

        if not hoa_message:
            return jsonify({'error': 'hoa_message is required and cannot be empty'}), 400

        # 3) Get case payload
        case_payload = case.get('payload', {})
        if isinstance(case_payload, str):
            try:
                case_payload = json.loads(case_payload)
            except:
                case_payload = {}

        # 4) Get statute context (same pattern as run_case_analysis)
        statute_context = ""
        try:
            case_state = extract_state_from_payload(case_payload)
            violation_type = extract_violation_type_from_payload(case_payload)
            if case_state:
                statute_ctx = get_statute_context(case_state, violation_type)
                if statute_ctx:
                    statute_context = statute_ctx
                    logger.info(f"Statute context loaded for email exchange, state={case_state}")
        except Exception as e:
            logger.warning(f"Failed to get statute context for email exchange: {str(e)}")

        # 5) Fetch prior exchanges for this case
        prior_exchanges = []
        try:
            exchanges_url = f"{SUPABASE_URL}/rest/v1/dmhoa_email_exchanges"
            exchanges_params = {
                'case_token': f'eq.{case_token}',
                'select': 'hoa_message,generated_response,exchange_date,created_at',
                'order': 'created_at.asc'
            }
            exchanges_headers = supabase_headers()

            exchanges_response = requests.get(exchanges_url, params=exchanges_params,
                                              headers=exchanges_headers, timeout=TIMEOUT)
            exchanges_response.raise_for_status()
            prior_exchanges = exchanges_response.json() or []
        except Exception as e:
            logger.warning(f"Error fetching prior exchanges: {str(e)}")

        # Format prior exchanges as readable thread
        if prior_exchanges:
            thread_parts = []
            for ex in prior_exchanges:
                ex_date = ex.get('exchange_date') or ex.get('created_at', 'Unknown date')
                if isinstance(ex_date, str) and 'T' in ex_date:
                    ex_date = ex_date.split('T')[0]  # Just the date part
                thread_parts.append(f"HOA MESSAGE [{ex_date}]: {ex.get('hoa_message', '')}")
                thread_parts.append(f"OUR RESPONSE [{ex_date}]: {ex.get('generated_response', '')}")
            exchange_history = "\n\n".join(thread_parts)
        else:
            exchange_history = "No prior exchanges"

        # 6) Build prompt and call Claude
        today_formatted = datetime.now().strftime("%B %d, %Y")

        system_prompt = """You are an HOA dispute assistant helping a homeowner respond to HOA communications. Generate professional, legally-grounded counter-letters that cite applicable state statutes and address each specific point raised by the HOA.

FORMATTING RULES — STRICTLY FOLLOW:
- Output plain text only. No markdown of any kind.
- No ** bold **, no ### headers, no # headings, no * bullets, no _ italics_
- No emojis. No symbols used as decorators
- Use ALL CAPS for section headings
- Use plain numbered lists (1. 2. 3.) or dashes (-) for bullets
- Separate sections with a blank line only
- The output will be rendered and copied as a plain text letter"""

        user_message = f"""ORIGINAL CASE DETAILS:
{json.dumps(case_payload, indent=2)}

APPLICABLE STATE STATUTES:
{statute_context if statute_context else "No state-specific statutes available."}

PRIOR EXCHANGE HISTORY:
{exchange_history}

NEW HOA MESSAGE TO RESPOND TO:
{hoa_message}

INSTRUCTIONS:
Generate a specific counter-letter responding directly to this HOA message. Address each point raised. Cite the applicable statutes provided above. Use non-admission language. Do not generate a generic letter — this must respond to the specific points the HOA raised. Include today's date ({today_formatted}), the owner's address, and recipient address from the case payload."""

        # Call Claude with retry logic
        if not ANTHROPIC_API_KEY:
            return jsonify({'error': 'AI service not configured'}), 500

        max_retries = 3
        base_delay = 5
        anthropic_response = None

        for attempt in range(max_retries):
            anthropic_response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': ANTHROPIC_API_KEY,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-sonnet-4-6',
                    'max_tokens': 4096,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_message}]
                },
                timeout=(10, 120)
            )

            if anthropic_response.status_code in [429, 529, 503]:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Claude API returned {anthropic_response.status_code}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            break

        if not anthropic_response.ok:
            error_text = anthropic_response.text
            logger.error(f"Claude API failed for email exchange: {anthropic_response.status_code}")
            return jsonify({'error': 'Failed to generate response'}), 500

        # Parse response
        claude_json = anthropic_response.json()
        content = claude_json.get('content', [])
        if content and content[0].get('type') == 'text':
            generated_response = content[0].get('text', '').strip()
        else:
            return jsonify({'error': 'Empty response from AI'}), 500

        # Post-process: Fill in placeholders with actual data from case payload
        generated_response = fill_letter_placeholders(generated_response, case_payload)

        # 7) Save exchange to database
        exchange_data = {
            'case_token': case_token,
            'hoa_message': hoa_message,
            'generated_response': generated_response,
            'exchange_date': datetime.utcnow().isoformat()
        }

        save_url = f"{SUPABASE_URL}/rest/v1/dmhoa_email_exchanges"
        save_headers = supabase_headers()
        save_headers['Prefer'] = 'return=representation'

        save_response = requests.post(save_url, headers=save_headers, json=exchange_data, timeout=TIMEOUT)
        save_response.raise_for_status()

        saved_data = save_response.json()
        exchange_id = saved_data[0].get('id') if saved_data else None

        logger.info(f"Email exchange generated for case_token={case_token[:8]}..., prior_exchanges={len(prior_exchanges)}")

        return jsonify({
            'generated_response': generated_response,
            'exchange_id': exchange_id
        }), 200

    except Exception as e:
        error_msg = f"Error creating email exchange: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/cases/<case_token>/email-exchanges', methods=['GET', 'OPTIONS'])
def get_email_exchanges(case_token):
    """Get all email exchanges for a case."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        # Validate case_token
        if not case_token or not case_token.strip():
            return jsonify({'error': 'case_token is required'}), 400

        case_token = case_token.strip()
        logger.info(f"Fetching email exchanges for case_token: {case_token[:8]}...")

        # 1) Ensure case exists and is paid/unlocked
        case = read_case_by_token(case_token)
        if not case:
            return jsonify({'error': 'Case not found'}), 404

        case_status = case.get('status', '')
        if case_status != 'paid':
            return jsonify({'error': 'Case is not unlocked. Payment required.'}), 403

        # 2) Fetch all exchanges for this case
        exchanges_url = f"{SUPABASE_URL}/rest/v1/dmhoa_email_exchanges"
        exchanges_params = {
            'case_token': f'eq.{case_token}',
            'select': 'id,hoa_message,generated_response,exchange_date,created_at',
            'order': 'created_at.desc'
        }
        exchanges_headers = supabase_headers()

        exchanges_response = requests.get(exchanges_url, params=exchanges_params,
                                          headers=exchanges_headers, timeout=TIMEOUT)
        exchanges_response.raise_for_status()
        exchanges = exchanges_response.json() or []

        logger.info(f"Found {len(exchanges)} email exchanges for case_token {case_token[:8]}...")

        return jsonify({'exchanges': exchanges}), 200

    except Exception as e:
        error_msg = f"Error fetching email exchanges: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


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

            # Get metadata - handle both dict and StripeObject
            metadata = session.get('metadata') or {}
            if hasattr(metadata, 'to_dict'):
                metadata = metadata.to_dict()
            elif not isinstance(metadata, dict):
                metadata = dict(metadata) if metadata else {}

            logger.info(f"Webhook metadata received: {metadata}")

            # Get token from client_reference_id or metadata (check both token and case_token)
            token = (
                    session.get('client_reference_id') or
                    metadata.get('token') or
                    metadata.get('case_token')
            )
            if not token:
                logger.error(f"No token found in session. Session data: {metadata}, client_reference_id: {session.get('client_reference_id')}")
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
                    # Also check inside payload if top-level email is not set
                    if not email:
                        case_payload = updated_case.get('payload') or {}
                        if isinstance(case_payload, str):
                            try:
                                case_payload = json.loads(case_payload)
                            except:
                                case_payload = {}
                        email = case_payload.get('email')

            except Exception as e:
                logger.error(f"Failed to update case: {str(e)}")
                return jsonify({'error': 'Database update failed'}), 500

            # --- Move email to Klaviyo post-purchase list (non-fatal) ---
            if email and KLAVIYO_POST_PURCHASE_LIST_ID:
                try:
                    # Run in background to not block the webhook response
                    def sync_klaviyo_post_purchase():
                        klaviyo_sync_profile_to_list(email, KLAVIYO_POST_PURCHASE_LIST_ID)
                        logger.info(f"Moved {email} to Klaviyo post-purchase list")

                    klaviyo_thread = threading.Thread(target=sync_klaviyo_post_purchase)
                    klaviyo_thread.daemon = True
                    klaviyo_thread.start()
                except Exception as e:
                    logger.warning(f"Failed to sync Klaviyo post-purchase list (non-fatal): {str(e)}")

            # --- Run case analysis in background thread after payment (non-blocking) ---
            # This prevents Stripe webhook timeout (30s) while Claude API may take longer
            logger.info(f"Queuing case analysis in background for token: {token}")

            def run_analysis_background(analysis_token):
                """Background thread function for case analysis"""
                try:
                    # Small delay to let the system stabilize after payment
                    time.sleep(2)

                    logger.info(f"Starting background case analysis for token: {analysis_token}")
                    analysis_result = run_case_analysis(analysis_token)

                    if analysis_result.get('ok'):
                        logger.info(
                            f"Background case analysis completed successfully for token: {analysis_token}, cached={analysis_result.get('cached', False)}")
                        # Log case analysis success event
                        try:
                            event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
                            event_data = {
                                'token': analysis_token,
                                'type': 'case_analysis_completed',
                                'data': {
                                    'cached': analysis_result.get('cached', False),
                                    'status': analysis_result.get('status', 'unknown')
                                }
                            }
                            event_headers = supabase_headers()
                            requests.post(event_url, headers=event_headers, json=event_data, timeout=TIMEOUT)
                        except Exception:
                            pass  # Best effort
                    else:
                        logger.error(
                            f"Background case analysis failed for token: {analysis_token}, error: {analysis_result.get('error', 'unknown')}")
                except Exception as e:
                    logger.error(f"Background case analysis exception for token: {analysis_token}: {str(e)}")

            try:
                analysis_thread = threading.Thread(target=run_analysis_background, args=(token,))
                analysis_thread.daemon = True
                analysis_thread.start()
                logger.info(f"Background analysis thread started for token: {token}")
            except Exception as e:
                logger.error(f"Failed to start background analysis thread for token {token}: {str(e)}")

            # --- Send receipt email directly (non-fatal) ---
            case_url_link = f"{SITE_URL}/case.html?case={token}"
            if not email:
                logger.warning("No email available (Stripe + DB). Skipping receipt email send.")
            else:
                try:
                    email_sent = send_receipt_email_direct(
                        token=token,
                        to_email=email,
                        case_url=case_url_link,
                        amount_total=session.get('amount_total'),
                        currency=session.get('currency'),
                        customer_name=session.get('customer_details', {}).get('name'),
                        stripe_session_id=session['id']
                    )

                    if email_sent:
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
                    else:
                        logger.warning(f"Receipt email send failed (non-fatal) for token: {token}")
                        # Log failed email event
                        try:
                            event_url = f"{SUPABASE_URL}/rest/v1/dmhoa_events"
                            event_data = {
                                'token': token,
                                'type': 'receipt_email_failed',
                                'data': {
                                    'error': 'send_receipt_email_direct returned False'
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


@app.route('/api/create-checkout-session', methods=['POST', 'OPTIONS'])
def create_checkout_session():
    """Create a Stripe checkout session for case purchase."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Validate required environment variables
        if not STRIPE_SECRET_KEY:
            logger.error("STRIPE_SECRET_KEY not configured")
            return jsonify({'error': 'Payment system not configured'}), 500

        if not STRIPE_PRICE_ID:
            logger.error("STRIPE_PRICE_ID not configured")
            return jsonify({'error': 'Product pricing not configured'}), 500

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Accept either case_id or case_token for flexibility
        case_id = data.get('case_id')
        case_token = data.get('case_token') or data.get('token')

        logger.info(f"Checkout request received - case_id: {case_id}, case_token: {case_token}")

        if not case_id and not case_token:
            return jsonify({'error': 'case_id or case_token is required'}), 400

        # If we have a token but no case_id, look up the case_id by token
        if case_token and not case_id:
            case = read_case_by_token(case_token)
            if not case:
                return jsonify({'error': 'Case not found for token'}), 404
            case_id = case.get('id')
            if not case_id:
                return jsonify({'error': 'Case ID not found for token'}), 404
        elif case_id:
            # Validate case exists by case_id
            case = read_case_by_id(case_id)
            if not case:
                return jsonify({'error': 'Case not found'}), 404
            case_token = case.get('token')
        else:
            return jsonify({'error': 'Unable to identify case'}), 400

        # Create Stripe checkout session
        try:
            # Use the correct frontend URL with the proper format
            frontend_url = "https://disputemyhoa.com"  # Updated to match your dev environment

            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': STRIPE_PRICE_ID,
                    'quantity': 1,
                }],
                mode='payment',
                allow_promotion_codes=True,
                success_url=f"{frontend_url}/case.html?case={case_token}&session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{frontend_url}/case-preview?case={case_token}",
                metadata={
                    'case_id': case_id,
                    'case_token': case_token or '',
                }
            )

            logger.info(f"Created checkout session {checkout_session.id} for case {case_id}")

            # Update Klaviyo profile with the Stripe checkout link
            try:
                email_for_klaviyo = case.get('email') or (case.get('payload') or {}).get('email')
                if email_for_klaviyo and checkout_session.url:
                    def update_klaviyo_stripe_link():
                        klaviyo_update_profile_properties(email_for_klaviyo, {
                            'stripe_checkout_link': checkout_session.url
                        })
                        logger.info(f"Updated stripe_checkout_link for {email_for_klaviyo}")

                    klaviyo_thread = threading.Thread(target=update_klaviyo_stripe_link)
                    klaviyo_thread.daemon = True
                    klaviyo_thread.start()
            except Exception as e:
                logger.warning(f"Failed to update Klaviyo stripe_checkout_link (non-critical): {str(e)}")

            # Create response with multiple field names for frontend compatibility
            response_data = {
                'checkout_url': checkout_session.url,
                'url': checkout_session.url,  # Alternative field name
                'session_id': checkout_session.id,
                'id': checkout_session.id,  # Alternative field name
            }

            logger.info(f"Returning checkout response: {response_data}")
            return jsonify(response_data), 200

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating checkout session: {str(e)}")
            return jsonify({'error': 'Payment system error'}), 500

    except Exception as e:
        error_msg = f"Error creating checkout session: {str(e)}"
        logger.error(error_msg)
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

    logger.info(
        f"send-receipt-email payload: token_present={bool(token)}, to_email={to_email}, case_url_present={bool(case_url)}, customer_name={customer_name}")

    if not token or not to_email or not case_url:
        logger.warning("send-receipt-email missing required fields")
        return jsonify({"error": "Missing token/email/case_url"}), 400

    # Personalized greeting
    greeting = f"Hi {customer_name}," if customer_name else "Hi,"

    # Format payment amount
    dollars = f"${(amount_total or 0) / 100:.2f}" if isinstance(amount_total, int) else ""
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

        logger.info(f"Receipt email sent to {to_email} (token={token[:8]}...")
        return jsonify({"ok": True}), 200

    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Gmail authentication failed. For Gmail, you need: 1) Enable 2-Step Verification, 2) Create App Password. Error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"ok": False, "error": error_msg}), 500
    except Exception as e:
        # Log full exception with traceback to help diagnose mail failures
        logger.exception("Failed to send receipt email")
        return jsonify({"ok": False, "error": str(e)}), 500


# ============================================================================
# BLOG ENDPOINTS
# ============================================================================

@app.route('/api/blog-posts', methods=['GET', 'OPTIONS'])
def get_blog_posts():
    """Get a list of published blog posts for the blog listing page."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Get pagination parameters
        limit_param = request.args.get('limit', '10')
        offset_param = request.args.get('offset', '0')
        category = request.args.get('category', '').strip()

        # Validate limit (min 1, max 50, default 10)
        try:
            limit = max(1, min(int(limit_param), 50))
        except (ValueError, TypeError):
            limit = 10

        # Validate offset (min 0, default 0)
        try:
            offset = max(0, int(offset_param))
        except (ValueError, TypeError):
            offset = 0

        logger.info(f"Fetching blog posts: limit={limit}, offset={offset}, category={category or 'all'}")

        # Build query parameters
        url = f"{SUPABASE_URL}/rest/v1/blog_posts"
        params = {
            'status': 'eq.published',
            'select': 'id,title,slug,excerpt,image_url,image_alt,category,tags,author,read_time_minutes,published_at',
            'order': 'published_at.desc',
            'limit': str(limit),
            'offset': str(offset)
        }

        # Add category filter if provided
        if category:
            params['category'] = f'eq.{category}'

        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        posts = response.json()

        # Get total count for pagination
        count_url = f"{SUPABASE_URL}/rest/v1/blog_posts"
        count_params = {
            'status': 'eq.published',
            'select': 'id'
        }
        if category:
            count_params['category'] = f'eq.{category}'

        count_headers = supabase_headers()
        count_headers['Prefer'] = 'count=exact'
        count_headers['Range-Unit'] = 'items'

        count_response = requests.head(url, params=count_params, headers=count_headers, timeout=TIMEOUT)
        total_count = 0
        if 'content-range' in count_response.headers:
            # Parse content-range header: "0-9/42" -> 42
            content_range = count_response.headers.get('content-range', '')
            if '/' in content_range:
                total_count = int(content_range.split('/')[1])

        logger.info(f"Successfully retrieved {len(posts)} blog posts (total: {total_count})")
        return jsonify({
            'posts': posts,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': total_count,
                'has_more': offset + len(posts) < total_count
            }
        }), 200

    except Exception as e:
        error_msg = f"Error fetching blog posts: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/blog-post/<slug>', methods=['GET', 'OPTIONS'])
def get_blog_post(slug):
    """Get a single blog post by slug for the blog detail page."""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'OK'})
        return response

    try:
        # Validate slug
        if not slug or len(slug) < 1:
            return jsonify({'error': 'Invalid slug'}), 400

        logger.info(f"Fetching blog post with slug: {slug}")

        # Fetch blog post from Supabase
        url = f"{SUPABASE_URL}/rest/v1/blog_posts"
        params = {
            'slug': f'eq.{slug}',
            'status': 'eq.published',
            'select': '*'
        }
        headers = supabase_headers()

        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()

        posts = response.json()
        if not posts:
            logger.warning(f"Blog post not found for slug: {slug}")
            return jsonify({'error': 'Blog post not found'}), 404

        post = posts[0]

        # Increment view count asynchronously (fire and forget)
        try:
            update_url = f"{SUPABASE_URL}/rest/v1/blog_posts"
            update_params = {'id': f"eq.{post['id']}"}
            update_headers = supabase_headers()
            update_data = {'view_count': post.get('view_count', 0) + 1}
            requests.patch(update_url, params=update_params, headers=update_headers,
                          json=update_data, timeout=(2, 5))
        except Exception:
            pass  # Don't fail the request if view count update fails

        logger.info(f"Successfully retrieved blog post: {post.get('title', slug)}")
        return jsonify(post), 200

    except Exception as e:
        error_msg = f"Error fetching blog post: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


if __name__ == '__main__':
    app.run(debug=True)
