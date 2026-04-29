"""
Email funnel state management. Each email gets one row in the email_funnel
Supabase table. Stages advance forward only and never regress.

Stage order:
    quick_preview_complete -> full_preview_viewed -> purchased

Once an email reaches a given stage, calling log_funnel_stage with that
same stage (or an earlier one) is a no-op. The hook fires the corresponding
immediate email and Klaviyo profile property sync exactly once per stage
transition.
"""

import os
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

from utils.email import send_email
from utils.klaviyo import sync_to_klaviyo
from utils import email_templates

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
TIMEOUT = (5, 15)

# Forward-only stage ordering
STAGE_ORDER = {
    'quick_preview_complete': 1,
    'full_preview_viewed': 2,
    'purchased': 3,
}

VALID_STAGES = set(STAGE_ORDER.keys())

# Domains we send FROM — these would create self-loops if they entered
# the funnel (e.g., Resend bounce addresses, our own support@). Personal
# tester emails (chemworeric@, *@astrodigitallabs.com) are NOT excluded
# here — they should flow through the funnel like real users so end-to-end
# email tests work. Dashboard metrics filtering happens separately in
# dashboard_routes.py via EXCLUDED_EMAILS.
EXCLUDED_EMAILS = set()
EXCLUDED_DOMAINS = {'disputemyhoa.com', 'mail.disputemyhoa.com'}

def _is_excluded_email(email: str) -> bool:
    """Returns True if this email is one of OUR send-from addresses and
    must not enter the customer funnel (would create reply loops)."""
    if not email:
        return True
    lower = email.lower().strip()
    if lower in EXCLUDED_EMAILS:
        return True
    domain = lower.split('@')[-1] if '@' in lower else ''
    if domain in EXCLUDED_DOMAINS:
        return True
    return False


def _supabase_headers() -> dict:
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json',
    }


def _fetch_funnel_row(email: str) -> Optional[dict]:
    """Return the existing funnel row for this email, or None."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None
    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/email_funnel',
            headers=_supabase_headers(),
            params={'email': f'eq.{email}', 'select': '*', 'limit': '1'},
            timeout=TIMEOUT,
        )
        if resp.ok:
            rows = resp.json()
            return rows[0] if rows else None
    except Exception as e:
        logger.error(f'_fetch_funnel_row failed for {email}: {e}')
    return None


def _upsert_funnel_row(email: str, stage: str) -> None:
    """Insert or update the funnel row for this email at this stage.

    The advancement check has already been performed by the caller.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        'email': email,
        'stage': stage,
        'stage_completed_at': now_iso,
        'purchased': stage == 'purchased',
    }
    # Store the case preview link if provided (used in nudge emails)
    if link:
        payload['case_link'] = link

    try:
        resp = requests.post(
            f'{SUPABASE_URL}/rest/v1/email_funnel',
            headers={
                **_supabase_headers(),
                'Prefer': 'resolution=merge-duplicates,return=representation',
            },
            params={'on_conflict': 'email'},
            json=payload,
            timeout=TIMEOUT,
        )
        if not resp.ok:
            logger.error(
                f'_upsert_funnel_row failed for {email}: HTTP {resp.status_code} - {resp.text[:300]}'
            )
    except Exception as e:
        logger.error(f'_upsert_funnel_row exception for {email}: {e}')


def _send_stage_email(stage: str, email: str, link: str = '') -> None:
    """Fire the immediate email for a stage transition."""
    try:
        if stage == 'quick_preview_complete':
            subject, body = email_templates.quick_preview_confirmation(link)
        elif stage == 'purchased':
            subject, body = email_templates.purchase_confirmation(link)
        else:
            # full_preview_viewed has no immediate email; only nudges later
            return
        send_email(email, subject, body)
    except Exception as e:
        logger.error(f'_send_stage_email failed for {email} at {stage}: {e}')


def _sync_klaviyo_for_stage(email: str, stage: str) -> None:
    """Push the new stage as a Klaviyo profile property (passive)."""
    try:
        sync_to_klaviyo(email, {
            'dmhoa_funnel_stage': stage,
            'dmhoa_funnel_stage_at': datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.warning(f'_sync_klaviyo_for_stage failed for {email}: {e}')


def log_funnel_stage(email: str, stage: str, link: str = '') -> bool:
    """Advance the funnel for this email to the given stage.

    - Stage transitions are forward-only. If the email is already at this
      stage or further, this is a no-op and returns False.
    - On a real advancement: upserts Supabase, fires the immediate email
      (if any) for that stage, and syncs the new stage to Klaviyo.
    - Never raises. Failures are logged but not propagated to the caller.

    Returns True if the funnel advanced, False otherwise.
    """
    if not email or '@' not in email:
        return False

    if _is_excluded_email(email):
        logger.info(f'Funnel: skipping internal email {email}')
        return False

    stage = (stage or '').strip()
    if stage not in VALID_STAGES:
        logger.warning(f'log_funnel_stage called with invalid stage: {stage!r}')
        return False

    new_stage_rank = STAGE_ORDER[stage]

    # Check current stage
    existing = _fetch_funnel_row(email)
    if existing:
        current_stage = existing.get('stage')
        current_rank = STAGE_ORDER.get(current_stage, 0)
        if current_rank >= new_stage_rank:
            # Already at this stage or further. No-op.
            return False

    # Advancement: upsert the row
    _upsert_funnel_row(email, stage)

    # Fire the immediate email and sync Klaviyo in background threads so we
    # don't block the calling request handler.
    def _fire():
        _send_stage_email(stage, email, link)
        _sync_klaviyo_for_stage(email, stage)

    try:
        t = threading.Thread(target=_fire, daemon=True)
        t.start()
    except Exception as e:
        logger.error(f'log_funnel_stage thread start failed for {email}: {e}')

    logger.info(f'Funnel: {email} advanced to {stage}')
    return True
