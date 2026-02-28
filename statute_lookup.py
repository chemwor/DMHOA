"""
Statute Lookup Module for DisputeMyHOA

Provides state-specific HOA statute information for prompt injection.
Caches results in Supabase, generates missing statutes via Claude AI.
"""

import os
import json
import logging
import re
from typing import Optional, Dict, Any
from datetime import datetime

import requests

# Configure logging
logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Request timeout
TIMEOUT = (5, 30)

# Valid violation categories
VALID_CATEGORIES = {'violation', 'fine', 'lien', 'architectural'}

# Mapping from free-text violation descriptions to normalized categories
CATEGORY_KEYWORDS = {
    'violation': [
        'violation', 'notice', 'complaint', 'infraction', 'breach', 'warning',
        'landscaping', 'lawn', 'grass', 'weeds', 'trash', 'parking', 'vehicle',
        'noise', 'pet', 'animal', 'maintenance', 'exterior', 'paint', 'color',
        'fence', 'structure', 'storage', 'debris', 'holiday', 'decoration',
        'sign', 'signage', 'rental', 'lease', 'occupancy', 'nuisance'
    ],
    'fine': [
        'fine', 'penalty', 'fee', 'charge', 'assessment', 'monetary', 'payment',
        'delinquent', 'late', 'collection', 'debt', 'owed', 'balance', 'dues'
    ],
    'lien': [
        'lien', 'foreclosure', 'title', 'encumbrance', 'judgment', 'record',
        'property claim', 'secured', 'priority'
    ],
    'architectural': [
        'architectural', 'arc', 'modification', 'improvement', 'construction',
        'build', 'addition', 'renovation', 'remodel', 'design', 'approval',
        'permit', 'plans', 'review', 'committee', 'acc', 'drc', 'arb'
    ]
}


def supabase_headers() -> Dict[str, str]:
    """Return headers for Supabase API requests."""
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json'
    }


def normalize_category(violation_description: str) -> str:
    """
    Map a free-text violation description to one of the standard categories:
    violation, fine, lien, architectural.

    Returns 'violation' as the default if no specific match is found.
    """
    if not violation_description:
        return 'violation'

    text = violation_description.lower()

    # Score each category based on keyword matches
    scores = {cat: 0 for cat in VALID_CATEGORIES}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[category] += 1
                # Boost score for exact word matches
                if re.search(rf'\b{re.escape(keyword)}\b', text):
                    scores[category] += 1

    # Find the category with the highest score
    best_category = max(scores, key=scores.get)

    # If no keywords matched, default to 'violation'
    if scores[best_category] == 0:
        return 'violation'

    return best_category


def normalize_state(state_input: str) -> Optional[str]:
    """
    Normalize state input to 2-letter abbreviation.
    Returns None if state cannot be determined.
    """
    if not state_input:
        return None

    state = state_input.strip().upper()

    # Already a 2-letter code
    if len(state) == 2 and state.isalpha():
        return state

    # Map full names to abbreviations
    state_names = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC'
    }

    return state_names.get(state)


def fetch_statute_from_db(state: str, category: str) -> Optional[Dict[str, Any]]:
    """
    Query Supabase for an existing statute record.
    Returns the row dict if found, None otherwise.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.warning("Supabase not configured for statute lookup")
        return None

    try:
        url = f"{SUPABASE_URL}/rest/v1/hoa_statutes"
        params = {
            'state': f'eq.{state}',
            'violation_category': f'eq.{category}',
            'select': '*',
            'limit': '1'
        }

        response = requests.get(url, params=params, headers=supabase_headers(), timeout=TIMEOUT)

        if response.status_code == 404:
            # Table doesn't exist yet
            logger.info(f"hoa_statutes table not found - may need migration")
            return None

        response.raise_for_status()
        rows = response.json()

        if rows:
            logger.info(f"DB HIT: Found statute for {state}/{category}")
            return rows[0]

        logger.info(f"DB MISS: No statute found for {state}/{category}")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch statute from DB for {state}/{category}: {str(e)}")
        return None


def generate_statute_with_claude(state: str, category: str) -> Optional[Dict[str, Any]]:
    """
    Generate statute information using Claude API.
    Returns a dict with statute fields if successful, None otherwise.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key not configured for statute generation")
        return None

    # Map category to human-readable form
    category_labels = {
        'violation': 'general HOA violations and enforcement',
        'fine': 'HOA fines, penalties, and assessments',
        'lien': 'HOA liens and foreclosure procedures',
        'architectural': 'architectural review and modification approvals'
    }

    category_label = category_labels.get(category, category)

    prompt = f"""You are a legal research assistant. Provide accurate information about {state} state laws governing {category_label} for homeowners associations (HOAs).

Output ONLY valid JSON with this exact structure:
{{
    "statute_name": "string (official statute name/title, e.g., 'Texas Property Code Chapter 209')",
    "statute_text": "string (brief summary of the key provisions, 2-3 sentences)",
    "procedural_requirements": "string (notice periods, hearing rights, appeal procedures)",
    "homeowner_protections": "string (key protections afforded to homeowners under this statute)",
    "notice_requirements": "string (required notice format, delivery method, timing)",
    "fine_rules": "string (limits on fines, escalation rules, caps if any)"
}}

IMPORTANT:
- Use REAL statute citations from {state} state law
- If {state} has no specific HOA statute for this category, reference the general property or corporation code that applies
- Be accurate and cite actual law - do not make up statute numbers
- Focus on provisions most relevant to homeowner rights and HOA obligations"""

    try:
        headers = {
            'x-api-key': ANTHROPIC_API_KEY,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 1024,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }

        logger.info(f"AI GENERATION: Generating statute for {state}/{category}")

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=(10, 60)
        )
        response.raise_for_status()

        result = response.json()
        content = result.get('content', [])

        if not content:
            logger.error("Empty response from Claude API")
            return None

        text = content[0].get('text', '')

        # Try to extract JSON from the response
        # Handle cases where Claude might wrap JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1)

        # Parse the JSON
        statute_data = json.loads(text)

        logger.info(f"AI GENERATION SUCCESS: Generated statute for {state}/{category}")
        return statute_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON for {state}/{category}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Claude API error for {state}/{category}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating statute for {state}/{category}: {str(e)}")
        return None


def save_statute_to_db(state: str, category: str, statute_data: Dict[str, Any]) -> bool:
    """
    Save a generated statute to the Supabase database.
    Returns True if successful, False otherwise.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False

    try:
        url = f"{SUPABASE_URL}/rest/v1/hoa_statutes"

        row = {
            'state': state,
            'violation_category': category,
            'statute_name': statute_data.get('statute_name', ''),
            'statute_text': statute_data.get('statute_text', ''),
            'procedural_requirements': statute_data.get('procedural_requirements', ''),
            'homeowner_protections': statute_data.get('homeowner_protections', ''),
            'notice_requirements': statute_data.get('notice_requirements', ''),
            'fine_rules': statute_data.get('fine_rules', ''),
            'ai_generated': True,
            'last_updated': datetime.utcnow().isoformat(),
            'created_at': datetime.utcnow().isoformat()
        }

        headers = supabase_headers()
        headers['Prefer'] = 'resolution=merge-duplicates'

        response = requests.post(url, headers=headers, json=row, timeout=TIMEOUT)

        if response.status_code in [200, 201]:
            logger.info(f"Saved AI-generated statute for {state}/{category}")
            return True
        else:
            logger.error(f"Failed to save statute: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error saving statute to DB: {str(e)}")
        return False


def format_statute_for_prompt(statute: Dict[str, Any], state: str, category: str) -> str:
    """
    Format statute data into a string suitable for prompt injection.
    """
    statute_name = statute.get('statute_name', f'{state} HOA Law')
    statute_text = statute.get('statute_text', '')
    procedural = statute.get('procedural_requirements', '')
    protections = statute.get('homeowner_protections', '')
    notice_reqs = statute.get('notice_requirements', '')
    fine_rules = statute.get('fine_rules', '')
    ai_generated = statute.get('ai_generated', False)

    parts = [
        f"=== STATE-SPECIFIC HOA LAW REFERENCE: {state} ===",
        f"Category: {category.title()}",
        f"Primary Statute: {statute_name}",
        ""
    ]

    if statute_text:
        parts.append(f"Summary: {statute_text}")
        parts.append("")

    if procedural:
        parts.append(f"Procedural Requirements: {procedural}")
        parts.append("")

    if protections:
        parts.append(f"Homeowner Protections: {protections}")
        parts.append("")

    if notice_reqs:
        parts.append(f"Notice Requirements: {notice_reqs}")
        parts.append("")

    if fine_rules:
        parts.append(f"Fine/Penalty Rules: {fine_rules}")
        parts.append("")

    parts.append("IMPORTANT: Use these statutes to inform strategy and tone. Statute citations belong ONLY in the Know Your Rights educational section, NOT in letter drafts.")

    if ai_generated:
        parts.append("(Note: This statute information was AI-generated and should be verified for specific legal applications.)")

    parts.append("=== END STATE LAW REFERENCE ===")

    return "\n".join(parts)


def get_statute_context(state: Optional[str], violation_description: Optional[str]) -> Optional[str]:
    """
    Main entry point: Get formatted statute context for prompt injection.

    Args:
        state: State abbreviation or full name (e.g., "TX" or "Texas")
        violation_description: Free-text description of the violation type

    Returns:
        Formatted string for prompt injection, or None if lookup fails
    """
    # Normalize state
    normalized_state = normalize_state(state) if state else None

    if not normalized_state:
        logger.info(f"Cannot determine state from input: {state}")
        return None

    # Normalize category from violation description
    category = normalize_category(violation_description)

    logger.info(f"Looking up statute for state={normalized_state}, category={category}")

    # Try to fetch from database first
    statute = fetch_statute_from_db(normalized_state, category)

    if statute:
        return format_statute_for_prompt(statute, normalized_state, category)

    # Generate with Claude if not in DB
    generated = generate_statute_with_claude(normalized_state, category)

    if generated:
        # Save to DB for future lookups (fire and forget)
        save_statute_to_db(normalized_state, category, generated)

        # Include ai_generated flag for formatting
        generated['ai_generated'] = True
        return format_statute_for_prompt(generated, normalized_state, category)

    # Graceful degradation - return None so caller can continue without statute context
    logger.warning(f"Failed to get statute context for {normalized_state}/{category}")
    return None


def extract_state_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract state from a case payload, checking multiple possible keys.
    """
    if not payload:
        return None

    # Try various possible keys for state
    state = (
        payload.get('propertyState') or
        payload.get('property_state') or
        payload.get('state') or
        payload.get('hoa_state') or
        payload.get('hoaState')
    )

    if state:
        return state

    # Try to extract state from property address
    address = payload.get('propertyAddress') or payload.get('property_address') or ''
    if address:
        # Try to find state abbreviation at the end of address
        # Pattern: matches ", ST 12345" or ", State 12345" or ", ST" at end
        state_match = re.search(r',\s*([A-Z]{2})\s*\d{0,5}\s*$', address.upper())
        if state_match:
            return state_match.group(1)

    return None


def extract_violation_type_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract violation type from a case payload, checking multiple possible keys.
    """
    if not payload:
        return None

    return (
        payload.get('violationType') or
        payload.get('violation_type') or
        payload.get('noticeType') or
        payload.get('notice_type') or
        payload.get('caseDescription') or
        payload.get('case_description')
    )
