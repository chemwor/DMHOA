"""
Passive Klaviyo sync. Pushes profile properties to Klaviyo so it can be used
as a passive CRM. Klaviyo never sends emails for the funnel — Resend handles
all sending. This module exists only to keep Klaviyo profiles up to date.

Uses the Klaviyo Profiles API with revision 2023-12-15.
Fire and forget: 3-second timeout, wrapped in try/except, never blocks the
caller, never raises.
"""

import os
import logging
from typing import Dict

import requests

logger = logging.getLogger(__name__)

KLAVIYO_API_KEY = os.environ.get('KLAVIYO_API_KEY')
KLAVIYO_API_URL = 'https://a.klaviyo.com/api'
KLAVIYO_REVISION = '2023-12-15'

TIMEOUT = (1, 3)  # connect, read — fire and forget


def _headers() -> Dict[str, str]:
    return {
        'Authorization': f'Klaviyo-API-Key {KLAVIYO_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'revision': KLAVIYO_REVISION,
    }


def _find_profile_id(email: str) -> str:
    """Look up an existing Klaviyo profile by email. Returns the id or empty string."""
    try:
        resp = requests.get(
            f'{KLAVIYO_API_URL}/profiles/',
            headers=_headers(),
            params={'filter': f'equals(email,"{email}")'},
            timeout=TIMEOUT,
        )
        if resp.ok:
            data = resp.json().get('data', [])
            if data:
                return data[0].get('id', '')
    except Exception:
        pass
    return ''


def sync_to_klaviyo(email: str, properties: dict) -> None:
    """Upsert a Klaviyo profile with the given custom properties.

    This is fire-and-forget. It logs failures but never raises and never
    blocks the user flow. Klaviyo is a passive CRM here.

    properties is a dict of arbitrary key/value pairs that get stored as
    custom profile properties on the Klaviyo profile.
    """
    if not KLAVIYO_API_KEY:
        logger.warning(f'KLAVIYO_API_KEY not configured, skipping sync for {email}')
        return

    if not email or '@' not in email:
        logger.warning(f'sync_to_klaviyo called with invalid email: {email!r}')
        return

    try:
        profile_id = _find_profile_id(email)

        attributes = {
            'email': email,
            'properties': properties or {},
        }

        if profile_id:
            # PATCH existing profile
            payload = {
                'data': {
                    'type': 'profile',
                    'id': profile_id,
                    'attributes': attributes,
                }
            }
            resp = requests.patch(
                f'{KLAVIYO_API_URL}/profiles/{profile_id}/',
                headers=_headers(),
                json=payload,
                timeout=TIMEOUT,
            )
        else:
            # Create new profile
            payload = {
                'data': {
                    'type': 'profile',
                    'attributes': attributes,
                }
            }
            resp = requests.post(
                f'{KLAVIYO_API_URL}/profiles/',
                headers=_headers(),
                json=payload,
                timeout=TIMEOUT,
            )

        if resp.status_code in (200, 201, 202):
            logger.info(f'Klaviyo: synced {email} with properties {list(properties.keys())}')
        else:
            logger.warning(
                f'Klaviyo sync non-2xx for {email}: HTTP {resp.status_code} - {resp.text[:200]}'
            )

    except requests.exceptions.Timeout:
        logger.warning(f'Klaviyo sync timeout for {email} (passive, ignored)')
    except Exception as e:
        logger.warning(f'Klaviyo sync exception for {email} (passive, ignored): {e}')
