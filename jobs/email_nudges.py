"""
Scheduled email nudges. Runs every 30 minutes via APScheduler from app.py.

Three independent queries against the email_funnel table:

  Nudge 1: stage = quick_preview_complete AND nudge_1_sent = false
           AND stage_completed_at < now - 3 hours
           -> send "did something go wrong?" email, mark nudge_1_sent

  Nudge 2: stage = full_preview_viewed AND nudge_2_sent = false
           AND stage_completed_at < now - 6 hours
           AND purchased = false
           -> send "your HOA case details are still here" email, mark nudge_2_sent

  Nudge 3: nudge_2_sent = true AND nudge_3_sent = false
           AND stage_completed_at < now - 24 hours
           AND purchased = false
           -> send "last reminder" email, mark nudge_3_sent

Each query is wrapped in try/except so one failure doesn't kill the others.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import requests

from utils.email import send_email
from utils import email_templates

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
TIMEOUT = (5, 15)


def _supabase_headers() -> Dict[str, str]:
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json',
    }


def _query_funnel(params: dict) -> List[Dict]:
    """Execute a Supabase REST query against email_funnel and return rows."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.warning('Supabase env vars not configured, skipping nudge query')
        return []
    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/email_funnel',
            headers=_supabase_headers(),
            params=params,
            timeout=TIMEOUT,
        )
        if resp.ok:
            return resp.json() or []
        logger.error(f'_query_funnel failed: HTTP {resp.status_code} - {resp.text[:200]}')
    except Exception as e:
        logger.error(f'_query_funnel exception: {e}')
    return []


def _mark_nudge_sent(email: str, field: str) -> None:
    """Set nudge_N_sent = true for this email."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    try:
        requests.patch(
            f'{SUPABASE_URL}/rest/v1/email_funnel',
            headers=_supabase_headers(),
            params={'email': f'eq.{email}'},
            json={field: True},
            timeout=TIMEOUT,
        )
    except Exception as e:
        logger.error(f'_mark_nudge_sent failed for {email} ({field}): {e}')


def _iso_n_hours_ago(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _run_nudge_1():
    """Stalled at quick_preview, 3+ hours ago."""
    cutoff = _iso_n_hours_ago(3)
    rows = _query_funnel({
        'select': 'email,stage_completed_at,case_link',
        'stage': 'eq.quick_preview_complete',
        'nudge_1_sent': 'eq.false',
        'stage_completed_at': f'lt.{cutoff}',
    })
    sent_count = 0
    for row in rows:
        email = row.get('email')
        if not email:
            continue
        try:
            link = row.get('case_link') or ''
            subject, body = email_templates.nudge_1(link)
            ok = send_email(email, subject, body)
            if ok:
                _mark_nudge_sent(email, 'nudge_1_sent')
                sent_count += 1
        except Exception as e:
            logger.error(f'Nudge 1 failed for {email}: {e}')
    if sent_count > 0:
        logger.info(f'Nudge 1: sent {sent_count} emails')
    return sent_count


def _run_nudge_2():
    """Viewed full preview, 6+ hours ago, not purchased."""
    cutoff = _iso_n_hours_ago(6)
    rows = _query_funnel({
        'select': 'email,stage_completed_at,case_link',
        'stage': 'eq.full_preview_viewed',
        'nudge_2_sent': 'eq.false',
        'purchased': 'eq.false',
        'stage_completed_at': f'lt.{cutoff}',
    })
    sent_count = 0
    for row in rows:
        email = row.get('email')
        if not email:
            continue
        try:
            link = row.get('case_link') or ''
            subject, body = email_templates.nudge_2(link)
            ok = send_email(email, subject, body)
            if ok:
                _mark_nudge_sent(email, 'nudge_2_sent')
                sent_count += 1
        except Exception as e:
            logger.error(f'Nudge 2 failed for {email}: {e}')
    if sent_count > 0:
        logger.info(f'Nudge 2: sent {sent_count} emails')
    return sent_count


def _run_nudge_3():
    """Already got nudge_2, 24+ hours later, still not purchased."""
    cutoff = _iso_n_hours_ago(24)
    rows = _query_funnel({
        'select': 'email,stage_completed_at,case_link',
        'nudge_2_sent': 'eq.true',
        'nudge_3_sent': 'eq.false',
        'purchased': 'eq.false',
        'stage_completed_at': f'lt.{cutoff}',
    })
    sent_count = 0
    for row in rows:
        email = row.get('email')
        if not email:
            continue
        try:
            link = row.get('case_link') or ''
            subject, body = email_templates.nudge_3(link)
            ok = send_email(email, subject, body)
            if ok:
                _mark_nudge_sent(email, 'nudge_3_sent')
                sent_count += 1
        except Exception as e:
            logger.error(f'Nudge 3 failed for {email}: {e}')
    if sent_count > 0:
        logger.info(f'Nudge 3: sent {sent_count} emails')
    return sent_count


def run_nudges():
    """Top-level entrypoint called by APScheduler every 30 minutes.

    Each nudge query is wrapped in its own try/except so one failure does
    not block the other two.
    """
    try:
        _run_nudge_1()
    except Exception as e:
        logger.error(f'run_nudges: nudge_1 batch crashed: {e}')

    try:
        _run_nudge_2()
    except Exception as e:
        logger.error(f'run_nudges: nudge_2 batch crashed: {e}')

    try:
        _run_nudge_3()
    except Exception as e:
        logger.error(f'run_nudges: nudge_3 batch crashed: {e}')
