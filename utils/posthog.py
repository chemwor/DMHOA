"""
Server-side PostHog event capture. Used to fire funnel events from places
where the browser may not be reachable (Stripe webhooks, background jobs).

Uses the public capture API which only needs the project's public API key.
We currently store the personal API key, so we look up the public key from
the project on first use and cache it. If anything fails, the call is a
no-op and never blocks the caller.
"""

import os
import logging
from typing import Optional, Dict

import requests

logger = logging.getLogger(__name__)

POSTHOG_PERSONAL_API_KEY = os.environ.get('POSTHOG_PERSONAL_API_KEY')
POSTHOG_PROJECT_ID = os.environ.get('POSTHOG_PROJECT_ID')
POSTHOG_HOST = 'https://us.i.posthog.com'

# The public project API key, fetched on first use and cached.
_public_api_key: Optional[str] = None

TIMEOUT = (3, 5)


def _get_public_api_key() -> Optional[str]:
    """Resolve and cache the project's public API key (the one prefixed with phc_)."""
    global _public_api_key
    if _public_api_key:
        return _public_api_key

    if not POSTHOG_PERSONAL_API_KEY or not POSTHOG_PROJECT_ID:
        return None

    try:
        resp = requests.get(
            f'https://us.posthog.com/api/projects/{POSTHOG_PROJECT_ID}/',
            headers={'Authorization': f'Bearer {POSTHOG_PERSONAL_API_KEY}'},
            timeout=TIMEOUT,
        )
        if resp.ok:
            data = resp.json()
            _public_api_key = data.get('api_token')
            return _public_api_key
        logger.warning(f'PostHog project lookup failed: HTTP {resp.status_code}')
    except Exception as e:
        logger.warning(f'PostHog project lookup exception: {e}')
    return None


def capture(distinct_id: str, event: str, properties: Optional[Dict] = None) -> None:
    """Fire a PostHog event server-side. Fire and forget, never raises.

    distinct_id should be a stable user identifier, typically email.
    """
    if not distinct_id or not event:
        return

    api_key = _get_public_api_key()
    if not api_key:
        return

    payload = {
        'api_key': api_key,
        'event': event,
        'distinct_id': distinct_id,
        'properties': properties or {},
    }

    try:
        resp = requests.post(
            f'{POSTHOG_HOST}/capture/',
            json=payload,
            timeout=TIMEOUT,
        )
        if resp.ok:
            logger.info(f'PostHog server-side: captured {event} for {distinct_id}')
        else:
            logger.warning(f'PostHog capture non-2xx: HTTP {resp.status_code}')
    except Exception as e:
        logger.warning(f'PostHog capture exception (ignored): {e}')
