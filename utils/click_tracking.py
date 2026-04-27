"""Click tracking for outbound emails.

We send plain-text emails (no HTML) to keep deliverability high in Gmail's
primary inbox. That blocks open-pixel tracking, but we can still track
*clicks* by replacing the destination URL in the email body with a short
URL that hits our backend, logs the click, and 302-redirects to the real
destination.

Usage:
    from utils.click_tracking import make_tracked_url
    tracked = make_tracked_url(
        case_link,
        link_kind='case_preview',
        case_token=token,
        recipient_email=email,
    )
    # `tracked` -> 'https://disputemyhoa.com/r/aB3xK9pQ'

If the click_tracking row can't be created (Supabase down, etc.), the
helper logs a warning and returns the original URL so the email still
works without tracking.
"""
import logging
import os
import secrets
import string
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

# Public-facing tracking host. `disputemyhoa.com/r/*` is proxied to the
# Heroku backend via the public site's netlify.toml so the URL in emails
# looks native (not `*.herokuapp.com`).
TRACKING_BASE_URL = os.environ.get(
    'TRACKING_BASE_URL',
    'https://disputemyhoa.com',
).rstrip('/')

SHORT_ID_ALPHABET = string.ascii_letters + string.digits  # 62-char base62
SHORT_ID_LENGTH = 8


def _generate_short_id() -> str:
    return ''.join(secrets.choice(SHORT_ID_ALPHABET) for _ in range(SHORT_ID_LENGTH))


def make_tracked_url(
    destination_url: str,
    link_kind: str = 'link',
    case_token: Optional[str] = None,
    recipient_email: Optional[str] = None,
    resend_email_id: Optional[str] = None,
) -> str:
    """Return a short tracked URL that 302s to destination_url.
    Falls back to the original URL on failure."""
    if not destination_url:
        return destination_url
    if not SUPABASE_URL or not SUPABASE_KEY:
        return destination_url

    short_id = _generate_short_id()
    try:
        resp = requests.post(
            f'{SUPABASE_URL}/rest/v1/dmhoa_email_clicks',
            headers={
                'apikey': SUPABASE_KEY,
                'Authorization': f'Bearer {SUPABASE_KEY}',
                'Content-Type': 'application/json',
                'Prefer': 'return=minimal',
            },
            json={
                'short_id': short_id,
                'destination_url': destination_url,
                'link_kind': link_kind,
                'case_token': case_token,
                'recipient_email': recipient_email,
                'resend_email_id': resend_email_id,
            },
            timeout=(5, 10),
        )
        if resp.status_code in (200, 201, 204):
            return f'{TRACKING_BASE_URL}/r/{short_id}'
        # Tiny chance of short_id collision; retry once with a new id.
        if resp.status_code == 409:
            short_id = _generate_short_id()
            resp2 = requests.post(
                f'{SUPABASE_URL}/rest/v1/dmhoa_email_clicks',
                headers={
                    'apikey': SUPABASE_KEY,
                    'Authorization': f'Bearer {SUPABASE_KEY}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=minimal',
                },
                json={
                    'short_id': short_id,
                    'destination_url': destination_url,
                    'link_kind': link_kind,
                    'case_token': case_token,
                    'recipient_email': recipient_email,
                    'resend_email_id': resend_email_id,
                },
                timeout=(5, 10),
            )
            if resp2.status_code in (200, 201, 204):
                return f'{TRACKING_BASE_URL}/r/{short_id}'
        logger.warning(
            f'click-tracking insert failed: {resp.status_code} {resp.text[:200]}'
        )
    except Exception as e:
        logger.warning(f'click-tracking insert exception: {e}')
    return destination_url


def lookup_and_log_click(short_id: str, user_agent: Optional[str] = None,
                         ip: Optional[str] = None) -> Optional[str]:
    """Fetch the destination_url for a short_id, log the click (first time
    + bump count), return destination_url. Returns None if not found."""
    if not short_id or not SUPABASE_URL or not SUPABASE_KEY:
        return None

    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    }

    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_email_clicks',
            params={
                'select': 'destination_url,first_clicked_at,click_count',
                'short_id': f'eq.{short_id}',
                'limit': '1',
            },
            headers=headers,
            timeout=(5, 10),
        )
        if not resp.ok:
            return None
        rows = resp.json()
        if not rows:
            return None
        row = rows[0]
        destination = row.get('destination_url')

        # Best-effort click log; we'd rather redirect even if logging fails.
        try:
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            update = {
                'last_clicked_at': now_iso,
                'click_count': (row.get('click_count') or 0) + 1,
                'last_user_agent': (user_agent or '')[:500],
                'last_ip': (ip or '')[:64],
            }
            if not row.get('first_clicked_at'):
                update['first_clicked_at'] = now_iso
            requests.patch(
                f'{SUPABASE_URL}/rest/v1/dmhoa_email_clicks',
                params={'short_id': f'eq.{short_id}'},
                headers={**headers, 'Prefer': 'return=minimal'},
                json=update,
                timeout=(5, 10),
            )
        except Exception as e:
            logger.warning(f'click log update failed for {short_id}: {e}')

        return destination
    except Exception as e:
        logger.warning(f'click lookup failed for {short_id}: {e}')
        return None
