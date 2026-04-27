# Dashboard API Routes - Migrated from Netlify Functions
# Flask routes for the DMHOA Dashboard analytics and management endpoints

import os
import json
import hashlib
import logging
import math
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Blueprint, request, jsonify
from statute_lookup import (
    VALID_CATEGORIES, generate_statute_with_claude, save_statute_to_db,
    fetch_statute_from_db, normalize_state
)

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
dashboard_bp = Blueprint('dashboard', __name__)

# Configuration from environment
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Klaviyo Configuration (deprecated for dashboard reads — kept for legacy
# profile sync from save_case)
KLAVIYO_API_KEY = os.environ.get('KLAVIYO_API_KEY')
KLAVIYO_FULL_PREVIEW_LIST_ID = os.environ.get('KLAVIYO_FULL_PREVIEW_LIST_ID', 'T6LY99')
KLAVIYO_QUICK_PREVIEW_LIST_ID = os.environ.get('KLAVIYO_QUICK_PREVIEW_LIST_ID', 'QS6zfC')

# Resend Configuration — RESEND_API_KEY now has Full Access (read+send)
RESEND_API_KEY = os.environ.get('RESEND_API_KEY')
RESEND_API_BASE = 'https://api.resend.com'

# Google Ads Configuration
GOOGLE_ADS_DEVELOPER_TOKEN = os.environ.get('GOOGLE_ADS_DEVELOPER_TOKEN')
GOOGLE_ADS_CUSTOMER_ID = os.environ.get('GOOGLE_ADS_CUSTOMER_ID')
GOOGLE_ADS_LOGIN_CUSTOMER_ID = os.environ.get('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
GOOGLE_ADS_CLIENT_ID = os.environ.get('GOOGLE_ADS_CLIENT_ID')
GOOGLE_ADS_CLIENT_SECRET = os.environ.get('GOOGLE_ADS_CLIENT_SECRET')
GOOGLE_ADS_REFRESH_TOKEN = os.environ.get('GOOGLE_ADS_REFRESH_TOKEN')
GOOGLE_ADS_API_VERSION = 'v21'
GOOGLE_ADS_API_BASE = f'https://googleads.googleapis.com/{GOOGLE_ADS_API_VERSION}'

# PostHog Configuration
POSTHOG_PROJECT_ID = os.environ.get('POSTHOG_PROJECT_ID')
POSTHOG_PERSONAL_API_KEY = os.environ.get('POSTHOG_PERSONAL_API_KEY')
POSTHOG_API_URL = 'https://us.posthog.com'

# GitHub Configuration (for Dependabot alerts)
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_REPOS = ['chemwor/DMHOA', 'chemwor/dmohadash', 'chemwor/disputemyhoa']

# Supabase Management API (for security advisor)
SUPABASE_ACCESS_TOKEN = os.environ.get('SUPABASE_ACCESS_TOKEN')
SUPABASE_PROJECT_REF = os.environ.get('SUPABASE_PROJECT_REF', 'yvdwrkhntyutpnklxsvz')

# Lighthouse Configuration
GOOGLE_PAGESPEED_API_KEY = os.environ.get('GOOGLE_PAGESPEED_API_KEY')
PAGESPEED_API_URL = 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed'
TARGET_URL = 'https://disputemyhoa.com/'

# OpenAI Configuration
OPENAI_ADMIN_KEY = os.environ.get('OPENAI_ADMIN_KEY') or os.environ.get('OPENAI_API_KEY')
OPENAI_MONTHLY_BUDGET = float(os.environ.get('OPENAI_MONTHLY_BUDGET', 0))

# Unsplash Configuration
UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY')

# Alert / SMTP Configuration
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL')
SMTP_HOST = os.environ.get('SMTP_HOST')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SMTP_USER = (os.environ.get('SMTP_USER') or '').strip().replace('\xa0', ' ')
SMTP_PASS = (os.environ.get('SMTP_PASS') or '').strip().replace('\xa0', ' ')
SMTP_FROM = os.environ.get('SMTP_FROM', 'support@disputemyhoa.com')

# Constants
MIN_DATE_2026 = datetime(2026, 1, 1)
MIN_TIMESTAMP_2026 = int(MIN_DATE_2026.timestamp())
EXCLUDED_EMAILS = ['chemworeric@gmail.com']
TIMEOUT = (5, 60)

# OpenAI Pricing per 1K tokens
OPENAI_PRICING = {
    'gpt-4o': {'input': 0.0025, 'output': 0.01},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'default': {'input': 0.002, 'output': 0.002},
}

# Claude pricing: $/1K tokens
CLAUDE_PRICING = {
    'claude-sonnet-4-20250514': {'input': 0.003, 'output': 0.015},
    'claude-haiku-4-5-20251001': {'input': 0.001, 'output': 0.005},
    'default': {'input': 0.003, 'output': 0.015},
}


def supabase_headers() -> Dict[str, str]:
    """Return headers for Supabase API requests."""
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json'
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_date_range(period: str) -> Dict[str, str]:
    """Get date range based on period (today, week, month, all)."""
    now = datetime.now()
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    if period == 'yesterday':
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'week':
        start = now - timedelta(days=7)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'month':
        start = now - timedelta(days=30)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'all':
        start = MIN_DATE_2026
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Ensure we never go before 2026
    if start < MIN_DATE_2026:
        start = MIN_DATE_2026

    return {
        'start': start.isoformat(),
        'end': end_of_day.isoformat()
    }


def get_timestamp_range(period: str) -> Dict[str, int]:
    """Get timestamp range for Stripe API."""
    now = datetime.now()
    end_of_day = now.replace(hour=23, minute=59, second=59)
    lte = int(end_of_day.timestamp())

    if period == 'yesterday':
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        gte = int(start.timestamp())
        lte = int(yesterday.replace(hour=23, minute=59, second=59).timestamp())
    elif period == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        gte = int(start.timestamp())
    elif period == 'week':
        start = now - timedelta(days=7)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        gte = int(start.timestamp())
    elif period == 'month':
        start = now - timedelta(days=30)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        gte = int(start.timestamp())
    elif period == 'all':
        gte = MIN_TIMESTAMP_2026
    else:
        gte = int((now - timedelta(days=1)).timestamp())

    # Ensure we never go before 2026
    gte = max(gte, MIN_TIMESTAMP_2026)

    return {'gte': gte, 'lte': lte}


PLAN_START_DATE = '2026-05-01'

# Style rules appended to every Claude/OpenAI system prompt to make output
# read human and avoid AI-tells. Keep this in sync with the same constant
# in the Netlify functions.
HUMAN_VOICE_RULES = """

WRITING STYLE RULES (critical, must follow):
- Never use em-dashes (—) or en-dashes (–). Use periods, commas, colons, or parentheses instead.
- Never use these words/phrases: delve, leverage, robust, seamlessly, comprehensive, holistic, empower, streamline, cutting-edge, state-of-the-art, embark, harness, tapestry, vibrant, transformative, paramount, pivotal, moreover, furthermore, in essence, it is worth noting, in conclusion, ultimately, navigate the complexities, in today's, in the realm of.
- Do not start sentences with "Indeed", "Notably", "Importantly", or "However,".
- Do not end with a "Conclusion" or "In summary" paragraph that just restates the body.
- Write plain, direct, conversational English. Short sentences. No throat-clearing.
- Sound like a real person wrote this, not like a press release."""

def get_google_ads_date_range(period: str) -> Dict[str, str]:
    """Get date range in YYYY-MM-DD format for Google Ads API.
    Uses America/Los_Angeles timezone to match the Google Ads account timezone."""
    now = datetime.now(ZoneInfo('America/Los_Angeles'))
    today = now.strftime('%Y-%m-%d')

    if period == 'yesterday':
        yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = yesterday
        today = yesterday  # end date is also yesterday
    elif period == 'today':
        start_date = today
    elif period == 'week':
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    elif period == 'month':
        start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
    elif period == 'plan_start':
        start_date = PLAN_START_DATE
    elif period == 'all':
        start_date = '2026-01-01'
    else:
        start_date = today

    if start_date < '2026-01-01':
        start_date = '2026-01-01'

    return {'startDate': start_date, 'endDate': today}


# ============================================================================
# STRIPE DASHBOARD ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/stripe', methods=['GET', 'OPTIONS'])
def get_stripe_data():
    """Get Stripe revenue and transaction data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not STRIPE_SECRET_KEY:
        return jsonify({'error': 'Stripe API key not configured'}), 500

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY

        period = request.args.get('period', 'today')
        ts_range = get_timestamp_range(period)

        # Fetch charges for the period
        charges = stripe.Charge.list(
            created={'gte': ts_range['gte'], 'lte': ts_range['lte']},
            limit=100
        )

        successful_charges = [c for c in charges.data if c.status == 'succeeded']
        revenue = sum(c.amount for c in successful_charges) / 100
        transactions = len(successful_charges)

        # Map all successful charges
        all_transactions = [
            {
                'id': c.id,
                'amount': c.amount / 100,
                'status': c.status,
                'created': datetime.fromtimestamp(c.created).isoformat(),
                'description': c.description or 'Payment',
                'name': c.billing_details.name if c.billing_details else None,
                'email': c.billing_details.email if c.billing_details else (c.receipt_email or None),
            }
            for c in charges.data
            if c.status == 'succeeded' and not c.refunded
        ]

        # Fetch refunds
        refunds = stripe.Refund.list(
            created={'gte': ts_range['gte'], 'lte': ts_range['lte']},
            limit=100
        )
        refund_count = len(refunds.data)
        refund_amount = sum(r.amount for r in refunds.data) / 100

        # Calculate MRR
        subscriptions = stripe.Subscription.list(status='active', limit=100)
        mrr = 0
        for sub in subscriptions.data:
            if sub.items.data:
                item = sub.items.data[0]
                if item.price.unit_amount:
                    amount = item.price.unit_amount / 100
                    interval = item.price.recurring.interval if item.price.recurring else None
                    if interval == 'year':
                        mrr += amount / 12
                    else:
                        mrr += amount

        # Recent transactions
        recent_charges = stripe.Charge.list(
            created={'gte': MIN_TIMESTAMP_2026},
            limit=50
        )
        recent_transactions = [
            {
                'id': c.id,
                'amount': c.amount / 100,
                'status': c.status,
                'created': datetime.fromtimestamp(c.created).isoformat(),
                'description': c.description or 'Payment',
                'refunded': c.refunded,
                'name': c.billing_details.name if c.billing_details else None,
                'email': c.billing_details.email if c.billing_details else (c.receipt_email or None),
            }
            for c in recent_charges.data
            if c.status == 'succeeded' and not c.refunded
        ][:10]

        return jsonify({
            'revenue': revenue,
            'transactions': transactions,
            'refunds': {
                'count': refund_count,
                'amount': refund_amount,
            },
            'mrr': round(mrr, 2),
            'recentTransactions': recent_transactions,
            'allTransactions': all_transactions,
            'period': period,
            'dataFrom': '2026-01-01',
        })

    except Exception as e:
        logger.error(f'Stripe API error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch Stripe data',
            'message': str(e),
        }), 500


# ============================================================================
# SUPABASE ANALYTICS ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/supabase', methods=['GET', 'OPTIONS'])
def get_supabase_analytics():
    """Get case analytics from Supabase for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        # Return mock data
        return jsonify({
            'quickPreviewCompletions': 0,
            'fullPreviewCompletions': 0,
            'purchases': 0,
            'totalRevenue': 0,
            'totalCases': 0,
            'funnel': {
                'visitors': 0,
                'quickPreviews': 0,
                'fullPreviews': 0,
                'purchases': 0,
                'visitorToQuickPreviewRate': 0,
                'quickToFullPreviewRate': 0,
                'fullPreviewToPurchaseRate': 0,
                'overallConversionRate': 0,
            },
            'isMockData': True,
            'message': 'Supabase not configured.',
        })

    try:
        period = request.args.get('period', 'today')
        date_range = get_date_range(period)

        # Fetch cases
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {
            'select': 'id,token,email,created_at,unlocked,stripe_payment_intent_id,amount_total,status,payload',
            'created_at': f'gte.{date_range["start"]}',
            'order': 'created_at.desc'
        }
        response = requests.get(url, params=params, headers=supabase_headers(), timeout=TIMEOUT)
        response.raise_for_status()
        cases = response.json()

        # Fetch case outputs
        outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
        outputs_params = {
            'select': 'id,case_token,status,model,prompt_version,created_at,updated_at',
            'order': 'created_at.desc'
        }
        outputs_response = requests.get(outputs_url, params=outputs_params, headers=supabase_headers(), timeout=TIMEOUT)
        case_outputs = outputs_response.json() if outputs_response.ok else []

        # Create outputs map
        outputs_map = {o['case_token']: o for o in case_outputs}

        # Filter out test accounts
        filtered_cases = []
        for case in cases:
            payload = case.get('payload', {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    payload = {}
            email = (payload.get('email') or case.get('email') or '').lower()
            if not any(excluded.lower() == email for excluded in EXCLUDED_EMAILS):
                filtered_cases.append(case)

        # Process cases
        quick_previews = 0
        full_previews = 0
        purchases = 0
        total_revenue = 0
        recent_cases = []
        completed_cases = []

        for case in filtered_cases:
            payload = case.get('payload', {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    payload = {}

            # Use status field directly: 'quick_preview', 'full_preview', 'paid'
            case_status = case.get('status', '')

            # Fallback: derive status from payload for older cases without status set
            if case_status not in ('quick_preview', 'full_preview', 'paid'):
                if case.get('unlocked') or case.get('stripe_payment_intent_id'):
                    case_status = 'paid'
                elif payload.get('completionPhase') == 'simple':
                    case_status = 'quick_preview'
                else:
                    case_status = 'full_preview'

            case_output = outputs_map.get(case.get('token'))
            has_output = case_output is not None

            if case_status == 'paid':
                purchases += 1
                # If amount_total is missing, try to look it up from Stripe
                if not case.get('amount_total'):
                    try:
                        import stripe
                        stripe.api_key = STRIPE_SECRET_KEY
                        amount = None
                        backfill_data = {}
                        if case.get('stripe_payment_intent_id'):
                            pi = stripe.PaymentIntent.retrieve(case['stripe_payment_intent_id'])
                            amount = pi.get('amount') or pi.get('amount_received') or 0
                        else:
                            # No payment intent ID — search Stripe checkout sessions by token
                            token = case.get('token')
                            sessions = stripe.checkout.Session.list(limit=100)
                            for sess in sessions.auto_paging_iter():
                                meta = sess.get('metadata') or {}
                                sess_token = meta.get('token') or meta.get('case_token') or sess.get('client_reference_id')
                                if sess_token == token and sess.get('payment_status') == 'paid':
                                    amount = sess.get('amount_total') or 0
                                    backfill_data['stripe_checkout_session_id'] = sess.get('id')
                                    backfill_data['stripe_payment_intent_id'] = sess.get('payment_intent')
                                    backfill_data['currency'] = sess.get('currency')
                                    cd = sess.get('customer_details') or {}
                                    if cd.get('email') and not case.get('email'):
                                        backfill_data['email'] = cd['email']
                                    break
                        if amount:
                            case['amount_total'] = amount
                            backfill_data['amount_total'] = amount
                            try:
                                backfill_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                                requests.patch(backfill_url,
                                    params={'token': f'eq.{case.get("token")}'},
                                    headers=supabase_headers(),
                                    json=backfill_data,
                                    timeout=5)
                            except Exception:
                                pass  # non-fatal backfill
                    except Exception as e:
                        logger.warning(f"Stripe lookup failed for {case.get('token')}: {e}")
                if case.get('amount_total'):
                    total_revenue += case['amount_total'] / 100

            if case_status == 'quick_preview':
                quick_previews += 1
            elif case_status == 'full_preview':
                full_previews += 1

            if len(recent_cases) < 10:
                recent_cases.append({
                    'id': case.get('id'),
                    'token': case.get('token'),
                    'email': payload.get('email') or case.get('email'),
                    'created_at': case.get('created_at'),
                    'status': case_status,
                    'noticeType': payload.get('noticeType'),
                    'issueText': (payload.get('issueText') or '')[:100] + '...' if payload.get('issueText') else None,
                    'amount': case.get('amount_total', 0) / 100 if case.get('amount_total') else None,
                    'hasOutput': has_output,
                    'outputStatus': case_output.get('status') if case_output else None,
                })

            if case_status == 'paid' and len(completed_cases) < 20:
                completed_cases.append({
                    'id': case.get('id'),
                    'token': case.get('token'),
                    'email': payload.get('email') or case.get('email'),
                    'created_at': case.get('created_at'),
                    'noticeType': payload.get('noticeType'),
                    'amount': case.get('amount_total', 0) / 100 if case.get('amount_total') else None,
                    'hasOutput': has_output,
                    'outputStatus': case_output.get('status') if case_output else 'no_output',
                    'outputModel': case_output.get('model') if case_output else None,
                    'outputCreatedAt': case_output.get('created_at') if case_output else None,
                })

        # Output stats
        output_stats = {
            'total': len(completed_cases),
            'ready': len([c for c in completed_cases if c['outputStatus'] == 'ready']),
            'pending': len([c for c in completed_cases if c['outputStatus'] == 'pending']),
            'error': len([c for c in completed_cases if c['outputStatus'] == 'error']),
            'noOutput': len([c for c in completed_cases if c['outputStatus'] == 'no_output']),
        }

        total_cases = len(filtered_cases)

        # Calculate funnel rates
        visitor_to_quick = round((quick_previews / total_cases) * 100, 1) if total_cases > 0 else 0
        quick_to_full = round((full_previews / quick_previews) * 100, 1) if quick_previews > 0 else 0
        full_to_purchase = round((purchases / full_previews) * 100, 1) if full_previews > 0 else 0
        overall_conversion = round((purchases / total_cases) * 100, 1) if total_cases > 0 else 0

        return jsonify({
            'quickPreviewCompletions': quick_previews,
            'fullPreviewCompletions': full_previews,
            'purchases': purchases,
            'totalRevenue': round(total_revenue, 2),
            'totalCases': total_cases,
            'funnel': {
                'visitors': total_cases,
                'quickPreviews': quick_previews,
                'fullPreviews': full_previews,
                'purchases': purchases,
                'visitorToQuickPreviewRate': visitor_to_quick,
                'quickToFullPreviewRate': quick_to_full,
                'fullPreviewToPurchaseRate': full_to_purchase,
                'overallConversionRate': overall_conversion,
            },
            'recentCases': recent_cases,
            'completedCases': completed_cases,
            'outputStats': output_stats,
            'period': period,
            'dateRange': date_range,
            'dataFrom': '2026-01-01',
        })

    except Exception as e:
        logger.error(f'Supabase error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch Supabase data',
            'message': str(e),
        }), 500


# ============================================================================
# RESEND + EMAIL FUNNEL ENDPOINT
# ============================================================================

def resend_headers() -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {RESEND_API_KEY}',
        'Content-Type': 'application/json',
    }


def _fetch_resend_emails(limit: int = 100) -> List[Dict]:
    """Pull the last N emails from Resend. Returns list of dicts with
    id, to, from, subject, created_at, last_event."""
    if not RESEND_API_KEY:
        return []
    try:
        r = requests.get(
            f'{RESEND_API_BASE}/emails',
            params={'limit': limit},
            headers=resend_headers(),
            timeout=TIMEOUT,
        )
        if not r.ok:
            logger.warning(f'Resend list failed: {r.status_code} {r.text[:200]}')
            return []
        return r.json().get('data', []) or []
    except Exception as e:
        logger.warning(f'Resend list exception: {e}')
        return []


def _fetch_resend_email(email_id: str) -> Optional[Dict]:
    """Fetch detail for a single email (includes events)."""
    if not RESEND_API_KEY or not email_id:
        return None
    try:
        r = requests.get(
            f'{RESEND_API_BASE}/emails/{email_id}',
            headers=resend_headers(),
            timeout=TIMEOUT,
        )
        return r.json() if r.ok else None
    except Exception:
        return None


def _resend_metrics_for_window(emails: List[Dict], start_iso: str, end_iso: str) -> Dict:
    """Aggregate Resend status counts for emails created within the window."""
    in_window = [
        e for e in emails
        if e.get('created_at') and start_iso <= e['created_at'] <= end_iso
    ]
    counts = {'sent': 0, 'delivered': 0, 'bounced': 0, 'complained': 0,
              'opened': 0, 'clicked': 0, 'queued': 0, 'unknown': 0}
    for e in in_window:
        ev = (e.get('last_event') or 'unknown').lower()
        if ev in counts:
            counts[ev] += 1
        else:
            counts['unknown'] += 1
    total = len(in_window)
    delivery_rate = (counts['delivered'] / total * 100) if total else 0
    bounce_rate = (counts['bounced'] / total * 100) if total else 0
    return {
        'total': total,
        'counts': counts,
        'delivery_rate_pct': round(delivery_rate, 1),
        'bounce_rate_pct': round(bounce_rate, 1),
    }


def _fetch_click_metrics(period: str = 'yesterday') -> Dict:
    """Email click stats from dmhoa_email_clicks for a window. Counts
    distinct recipients clicked + total clicks within the period."""
    out = {
        'distinct_clickers': 0,
        'total_clicks': 0,
        'links_sent': 0,
        'click_rate_pct': 0.0,
        'period': period,
    }
    if not SUPABASE_URL:
        return out
    date_range = get_date_range(period)
    try:
        # Links sent in window (rows created in window)
        sent_resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_email_clicks",
            params={
                'select': 'short_id,recipient_email,first_clicked_at,click_count',
                'created_at': f'gte.{date_range["start"]}',
                'and': f'(created_at.lte.{date_range["end"]})',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT,
        )
        if sent_resp.ok:
            rows = sent_resp.json()
            out['links_sent'] = len(rows)
            clicked = [r for r in rows if r.get('first_clicked_at')]
            distinct = {r.get('recipient_email') for r in clicked if r.get('recipient_email')}
            out['distinct_clickers'] = len(distinct)
            out['total_clicks'] = sum((r.get('click_count') or 0) for r in clicked)
            if out['links_sent']:
                out['click_rate_pct'] = round(
                    len(clicked) / out['links_sent'] * 100, 1
                )
    except Exception as e:
        logger.warning(f'_fetch_click_metrics: {e}')
    return out


def _fetch_email_funnel_metrics(period: str = 'yesterday') -> Dict:
    """Aggregate funnel stages from Supabase email_funnel for a window."""
    out = {
        'total_unique': 0,
        'by_stage': {},
        'nudges_sent': 0,
        'period': period,
    }
    if not SUPABASE_URL:
        return out

    date_range = get_date_range(period)
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/email_funnel",
            params={
                'select': 'email,stage,nudge_1_sent,nudge_2_sent,nudge_3_sent,purchased,created_at',
                'created_at': f'gte.{date_range["start"]}',
                'and': f'(created_at.lte.{date_range["end"]})',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT,
        )
        if not r.ok:
            return out
        rows = r.json()
        excluded = {e.lower() for e in EXCLUDED_EMAILS}
        rows = [x for x in rows if (x.get('email') or '').lower() not in excluded]
        unique_emails = {x.get('email') for x in rows if x.get('email')}
        out['total_unique'] = len(unique_emails)
        from collections import Counter
        out['by_stage'] = dict(Counter(x.get('stage', 'unknown') for x in rows))
        out['nudges_sent'] = sum(
            1 for x in rows
            if x.get('nudge_1_sent') or x.get('nudge_2_sent') or x.get('nudge_3_sent')
        )
    except Exception as e:
        logger.warning(f'_fetch_email_funnel_metrics: {e}')

    return out


def _fetch_email_metrics() -> Dict:
    """Combined Resend + funnel + click metrics used by the daily digest
    snapshot. Replaces _fetch_klaviyo_metrics for dashboard reads."""
    today_range = get_date_range('today')
    yest_range = get_date_range('yesterday')
    week_range = get_date_range('week')

    emails = _fetch_resend_emails(limit=200)

    return {
        'resend': {
            'today': _resend_metrics_for_window(emails, today_range['start'], today_range['end']),
            'yesterday': _resend_metrics_for_window(emails, yest_range['start'], yest_range['end']),
            'week': _resend_metrics_for_window(emails, week_range['start'], week_range['end']),
        },
        'funnel': {
            'today': _fetch_email_funnel_metrics('today'),
            'yesterday': _fetch_email_funnel_metrics('yesterday'),
            'week': _fetch_email_funnel_metrics('week'),
        },
        'clicks': {
            'today': _fetch_click_metrics('today'),
            'yesterday': _fetch_click_metrics('yesterday'),
            'week': _fetch_click_metrics('week'),
        },
    }


@dashboard_bp.route('/api/dashboard/email', methods=['GET', 'OPTIONS'])
def get_email_dashboard():
    """Combined Resend + funnel data for the Email dashboard page.
    Returns sends overview, funnel breakdown, and recent send list."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        emails = _fetch_resend_emails(limit=100)
        excluded = {e.lower() for e in EXCLUDED_EMAILS}

        def _to_email(addr_list):
            if isinstance(addr_list, list):
                return addr_list[0] if addr_list else ''
            return addr_list or ''

        recent = []
        for e in emails:
            to = _to_email(e.get('to'))
            if to.lower() in excluded:
                continue
            recent.append({
                'id': e.get('id'),
                'to': to,
                'from': e.get('from'),
                'subject': e.get('subject'),
                'last_event': e.get('last_event'),
                'created_at': e.get('created_at'),
            })

        today_range = get_date_range('today')
        yest_range = get_date_range('yesterday')
        week_range = get_date_range('week')

        return jsonify({
            'resend': {
                'today': _resend_metrics_for_window(emails, today_range['start'], today_range['end']),
                'yesterday': _resend_metrics_for_window(emails, yest_range['start'], yest_range['end']),
                'week': _resend_metrics_for_window(emails, week_range['start'], week_range['end']),
            },
            'funnel': {
                'today': _fetch_email_funnel_metrics('today'),
                'yesterday': _fetch_email_funnel_metrics('yesterday'),
                'week': _fetch_email_funnel_metrics('week'),
            },
            'clicks': {
                'today': _fetch_click_metrics('today'),
                'yesterday': _fetch_click_metrics('yesterday'),
                'week': _fetch_click_metrics('week'),
            },
            'recent_sends': recent[:50],
            'has_full_access': bool(RESEND_API_KEY),
        })
    except Exception as e:
        logger.error(f'get_email_dashboard error: {e}')
        return jsonify({'error': str(e)}), 500


# ============================================================================
# KLAVIYO ENDPOINT (legacy — kept for backward compat, no longer used in
# daily digest or dashboard email page; Resend is the source of truth now)
# ============================================================================

def klaviyo_headers() -> Dict[str, str]:
    """Return headers for Klaviyo API requests."""
    return {
        'Authorization': f'Klaviyo-API-Key {KLAVIYO_API_KEY}',
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'revision': '2024-02-15'
    }


def get_klaviyo_list_count(list_id: str) -> int:
    """Get profile count for a Klaviyo list."""
    if not KLAVIYO_API_KEY:
        return 0

    try:
        total_count = 0
        next_page_url = f'/lists/{list_id}/relationships/profiles/'

        while next_page_url:
            response = requests.get(
                f'https://a.klaviyo.com/api{next_page_url}',
                headers=klaviyo_headers(),
                timeout=TIMEOUT
            )

            if not response.ok:
                return 0

            data = response.json()
            total_count += len(data.get('data', []))

            if data.get('links', {}).get('next'):
                from urllib.parse import urlparse
                parsed = urlparse(data['links']['next'])
                next_page_url = parsed.path + ('?' + parsed.query if parsed.query else '')
            else:
                next_page_url = None

        return total_count
    except Exception as e:
        logger.error(f'Error fetching Klaviyo list {list_id}: {str(e)}')
        return 0


@dashboard_bp.route('/api/dashboard/klaviyo', methods=['GET', 'OPTIONS'])
def get_klaviyo_data():
    """Get Klaviyo email marketing data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not KLAVIYO_API_KEY:
        return jsonify({
            'isMockData': True,
            'error': 'Klaviyo API key not configured',
            'totalProfiles': 0,
            'totalEmailsInFlow': 0,
            'fullPreviewEmails': 0,
            'quickPreviewEmails': 0,
            'emailsCollectedToday': 0,
            'lists': [],
            'flowStats': [],
        })

    try:
        # Fetch data in parallel (simplified for Python)
        full_preview_count = get_klaviyo_list_count(KLAVIYO_FULL_PREVIEW_LIST_ID)
        quick_preview_count = get_klaviyo_list_count(KLAVIYO_QUICK_PREVIEW_LIST_ID)

        # Get total profiles by paginating through all profiles
        total_profiles = 0
        next_url = '/profiles/?page[size]=100'
        while next_url:
            response = requests.get(
                f'https://a.klaviyo.com/api{next_url}',
                headers=klaviyo_headers(),
                timeout=TIMEOUT
            )
            if not response.ok:
                break
            page_data = response.json()
            total_profiles += len(page_data.get('data', []))
            next_link = page_data.get('links', {}).get('next')
            if next_link:
                from urllib.parse import urlparse
                parsed = urlparse(next_link)
                next_url = parsed.path + ('?' + parsed.query if parsed.query else '')
            else:
                next_url = None

        # Get new profiles today
        since = (datetime.now() - timedelta(hours=24)).isoformat()
        new_response = requests.get(
            f'https://a.klaviyo.com/api/profiles/?filter=greater-or-equal(created,{since})&page[size]=100',
            headers=klaviyo_headers(),
            timeout=TIMEOUT
        )
        emails_collected_today = len(new_response.json().get('data', [])) if new_response.ok else 0

        # Get flows
        flows_response = requests.get(
            'https://a.klaviyo.com/api/flows/',
            headers=klaviyo_headers(),
            timeout=TIMEOUT
        )
        flows = flows_response.json().get('data', []) if flows_response.ok else []

        # Fetch performance stats for each flow via Reporting API
        flow_stats = []
        reporting_headers = {
            **klaviyo_headers(),
            'revision': '2024-10-15',
        }
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT00:00:00+00:00')
        now_iso = datetime.now().strftime('%Y-%m-%dT23:59:59+00:00')

        for f in flows[:5]:
            flow_id = f.get('id', '')
            flow_name = f.get('attributes', {}).get('name', 'Unknown Flow')
            flow_status = f.get('attributes', {}).get('status', 'unknown')

            stats = {
                'name': flow_name,
                'status': flow_status,
                'sends': 0,
                'opens': 0,
                'clicks': 0,
                'openRate': 0,
                'clickRate': 0,
                'deliveryRate': 0,
                'bounces': 0,
                'unsubscribes': 0,
            }

            if flow_id:
                try:
                    report_response = requests.post(
                        'https://a.klaviyo.com/api/flow-values-reports/',
                        headers=reporting_headers,
                        json={
                            'data': {
                                'type': 'flow-values-report',
                                'attributes': {
                                    'timeframe': {
                                        'start': thirty_days_ago,
                                        'end': now_iso,
                                    },
                                    'conversion_metric_id': 'UE778r',
                                    'statistics': [
                                        'recipients',
                                        'opens',
                                        'opens_unique',
                                        'clicks',
                                        'clicks_unique',
                                        'delivery_rate',
                                        'open_rate',
                                        'click_rate',
                                    ],
                                    'filter': f'equals(flow_id,"{flow_id}")',
                                    'group_by': ['flow_message_id', 'flow_id'],
                                }
                            }
                        },
                        timeout=TIMEOUT
                    )
                    if report_response.ok:
                        report_data = report_response.json()
                        results = report_data.get('data', {}).get('attributes', {}).get('results', [])
                        logger.info(f'Flow {flow_name} ({flow_id}): got {len(results)} result rows')
                        if results:
                            # Aggregate across all flow messages
                            total_sends = 0
                            total_opens = 0
                            total_clicks = 0
                            weighted_delivery = 0  # sum of (delivery_rate * recipients) per message
                            for row in results:
                                row_stats = row.get('statistics', {})
                                recip = int(row_stats.get('recipients', 0) or 0)
                                total_sends += recip
                                total_opens += int(row_stats.get('opens_unique', 0) or 0)
                                total_clicks += int(row_stats.get('clicks_unique', 0) or 0)
                                dr = float(row_stats.get('delivery_rate', 0) or 0)
                                weighted_delivery += dr * recip
                            stats['sends'] = total_sends
                            stats['opens'] = total_opens
                            stats['clicks'] = total_clicks
                            # Compute rates from aggregated totals
                            stats['openRate'] = round((total_opens / total_sends * 100), 1) if total_sends > 0 else 0
                            stats['clickRate'] = round((total_clicks / total_sends * 100), 1) if total_sends > 0 else 0
                            stats['deliveryRate'] = round((weighted_delivery / total_sends * 100), 1) if total_sends > 0 else 0
                    else:
                        logger.warning(f'Flow report API returned {report_response.status_code} for flow {flow_id}: {report_response.text[:200]}')
                except Exception as e:
                    logger.warning(f'Failed to fetch flow stats for {flow_name}: {e}')

            flow_stats.append(stats)
            if flow_id:
                time.sleep(1.5)  # Avoid Klaviyo rate limits (429 at 0.5s)

        total_emails_in_flow = full_preview_count + quick_preview_count

        # Fetch leads from Supabase (more reliable than Klaviyo for lead count)
        case_metrics = _fetch_supabase_case_metrics('today')
        case_metrics_all = _fetch_supabase_case_metrics('all')

        return jsonify({
            'totalProfiles': total_profiles,
            'totalEmailsInFlow': total_emails_in_flow,
            'fullPreviewEmails': full_preview_count,
            'quickPreviewEmails': quick_preview_count,
            'emailsCollectedToday': emails_collected_today,
            'leadsToday': case_metrics.get('new_cases', 0),
            'totalLeads': case_metrics_all.get('total_cases', 0),
            'paidCasesToday': case_metrics.get('paid_cases', 0),
            'lists': [
                {'id': KLAVIYO_FULL_PREVIEW_LIST_ID, 'name': 'Full Preview Abandonment', 'count': full_preview_count},
                {'id': KLAVIYO_QUICK_PREVIEW_LIST_ID, 'name': 'Quick Preview Abandonment', 'count': quick_preview_count},
            ],
            'flowStats': flow_stats,
            'totalFlows': len(flows),
            'isMockData': False,
        })

    except Exception as e:
        logger.error(f'Klaviyo API error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch Klaviyo data',
            'message': str(e),
        }), 500


# ============================================================================
# GOOGLE ADS ENDPOINT
# ============================================================================

def get_google_ads_access_token() -> Optional[str]:
    """Get OAuth2 access token for Google Ads API."""
    if not all([GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN]):
        return None

    try:
        response = requests.post(
            'https://oauth2.googleapis.com/token',
            data={
                'client_id': GOOGLE_ADS_CLIENT_ID,
                'client_secret': GOOGLE_ADS_CLIENT_SECRET,
                'refresh_token': GOOGLE_ADS_REFRESH_TOKEN,
                'grant_type': 'refresh_token',
            },
            timeout=TIMEOUT
        )

        if response.ok:
            return response.json().get('access_token')
        return None
    except Exception as e:
        logger.error(f'Failed to get Google Ads access token: {str(e)}')
        return None


def query_google_ads(customer_id: str, access_token: str, query: str) -> List[Dict]:
    """Execute a Google Ads API query."""
    url = f'{GOOGLE_ADS_API_BASE}/customers/{customer_id}/googleAds:search'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'developer-token': GOOGLE_ADS_DEVELOPER_TOKEN,
        'Content-Type': 'application/json',
    }

    if GOOGLE_ADS_LOGIN_CUSTOMER_ID:
        headers['login-customer-id'] = GOOGLE_ADS_LOGIN_CUSTOMER_ID

    response = requests.post(url, headers=headers, json={'query': query}, timeout=TIMEOUT)

    if not response.ok:
        raise Exception(f'Google Ads API error: {response.status_code} - {response.text}')

    return response.json().get('results', [])


@dashboard_bp.route('/api/dashboard/google-ads', methods=['GET', 'OPTIONS'])
def get_google_ads_data():
    """Get Google Ads campaign data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    # Check if credentials are configured
    has_credentials = all([
        GOOGLE_ADS_DEVELOPER_TOKEN,
        GOOGLE_ADS_CUSTOMER_ID,
        GOOGLE_ADS_CLIENT_ID,
        GOOGLE_ADS_CLIENT_SECRET,
        GOOGLE_ADS_REFRESH_TOKEN
    ])

    if not has_credentials:
        # Return mock data
        return jsonify({
            'dailySpend': 0,
            'clicks': 0,
            'impressions': 0,
            'cpc': 0,
            'ctr': 0,
            'conversions': 0,
            'costPerConversion': 0,
            'campaigns': [],
            'keywords': [],
            'searchTerms': [],
            'ads': [],
            'targetCampaign': 'DMHOA - DIY Response - Phrase - March',
            'isMockData': True,
            'message': 'Google Ads not configured.',
        })

    try:
        period = request.args.get('period', 'today')
        date_range = get_google_ads_date_range(period)

        # Optional campaign filter (single campaign by exact name)
        campaign_filter = request.args.get('campaign', '').strip()
        # Sanitize: only allow alphanumeric, dashes, spaces, underscores
        if campaign_filter and not all(ch.isalnum() or ch in ' -_' for ch in campaign_filter):
            return jsonify({'error': 'Invalid campaign name'}), 400
        campaign_clause = f"AND campaign.name = '{campaign_filter}'" if campaign_filter else ''

        access_token = get_google_ads_access_token()
        if not access_token:
            raise Exception('Failed to get access token')

        # Query campaign performance — only campaigns with ad spend
        query = f"""
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                metrics.impressions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.conversions,
                metrics.ctr,
                metrics.average_cpc
            FROM campaign
            WHERE segments.date BETWEEN '{date_range["startDate"]}' AND '{date_range["endDate"]}'
                AND campaign.status != 'REMOVED'
                AND metrics.cost_micros > 0
                {campaign_clause}
            ORDER BY metrics.cost_micros DESC
        """

        results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, query)

        total_spend = 0
        total_clicks = 0
        total_impressions = 0
        total_conversions = 0

        campaigns = []
        for row in results:
            campaign = row.get('campaign', {})
            metrics = row.get('metrics', {})

            spend = int(metrics.get('costMicros', 0) or 0) / 1_000_000
            clicks = int(metrics.get('clicks', 0) or 0)
            impressions = int(metrics.get('impressions', 0) or 0)
            conversions = float(metrics.get('conversions', 0) or 0)

            total_spend += spend
            total_clicks += clicks
            total_impressions += impressions
            total_conversions += conversions

            campaigns.append({
                'id': campaign.get('id'),
                'name': campaign.get('name'),
                'status': campaign.get('status'),
                'spend': round(spend, 2),
                'clicks': clicks,
                'impressions': impressions,
                'conversions': round(conversions, 2),
                'cpc': round(spend / clicks, 2) if clicks > 0 else 0,
                'ctr': round((clicks / impressions) * 100, 2) if impressions > 0 else 0,
            })

        # --- Keyword performance ---
        keywords = []
        try:
            kw_query = f"""
                SELECT
                    campaign.name,
                    ad_group.name,
                    ad_group_criterion.keyword.text,
                    ad_group_criterion.keyword.match_type,
                    ad_group_criterion.status,
                    ad_group_criterion.quality_info.quality_score,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.conversions_value,
                    metrics.search_impression_share,
                    metrics.search_top_impression_share
                FROM keyword_view
                WHERE segments.date BETWEEN '{date_range["startDate"]}' AND '{date_range["endDate"]}'
                    AND campaign.status = 'ENABLED'
                    AND ad_group_criterion.status != 'REMOVED'
                    AND metrics.impressions > 0
                    {campaign_clause}
                ORDER BY metrics.clicks DESC
                LIMIT 50
            """
            kw_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, kw_query)
            for row in kw_results:
                m = row.get('metrics', {})
                crit = row.get('adGroupCriterion', {})
                kw_info = crit.get('keyword', {})
                qi = crit.get('qualityInfo', {})
                kw_clicks = int(m.get('clicks', 0) or 0)
                kw_impr = int(m.get('impressions', 0) or 0)
                kw_spend = int(m.get('costMicros', 0) or 0) / 1_000_000
                kw_conv = float(m.get('conversions', 0) or 0)
                kw_conv_val = float(m.get('conversionsValue', 0) or 0)
                qs_raw = qi.get('qualityScore')
                qs = int(qs_raw) if qs_raw is not None else None
                sis_raw = m.get('searchImpressionShare')
                sis = round(float(sis_raw), 4) if sis_raw is not None else None
                tis_raw = m.get('searchTopImpressionShare')
                tis = round(float(tis_raw), 4) if tis_raw is not None else None
                keywords.append({
                    'campaignName': row.get('campaign', {}).get('name', ''),
                    'adGroupName': row.get('adGroup', {}).get('name', ''),
                    'keyword': kw_info.get('text', ''),
                    'matchType': kw_info.get('matchType', 'BROAD'),
                    'status': crit.get('status', ''),
                    'qualityScore': qs,
                    'impressions': kw_impr,
                    'clicks': kw_clicks,
                    'spend': round(kw_spend, 2),
                    'conversions': round(kw_conv, 2),
                    'conversionValue': round(kw_conv_val, 2),
                    'costPerConversion': round(kw_spend / kw_conv, 2) if kw_conv > 0 else 0,
                    'ctr': round((kw_clicks / kw_impr) * 100, 2) if kw_impr > 0 else 0,
                    'cpc': round(kw_spend / kw_clicks, 2) if kw_clicks > 0 else 0,
                    'searchImpressionShare': sis,
                    'topImpressionShare': tis,
                })
        except Exception as e:
            logger.warning(f'Failed to fetch keyword data: {str(e)}')

        # --- Search terms ---
        search_terms = []
        try:
            st_query = f"""
                SELECT
                    campaign.name,
                    ad_group.name,
                    search_term_view.search_term,
                    search_term_view.status,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions
                FROM search_term_view
                WHERE segments.date BETWEEN '{date_range["startDate"]}' AND '{date_range["endDate"]}'
                    AND campaign.status = 'ENABLED'
                    AND metrics.impressions > 0
                    {campaign_clause}
                ORDER BY metrics.clicks DESC
                LIMIT 50
            """
            st_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, st_query)
            for row in st_results:
                m = row.get('metrics', {})
                stv = row.get('searchTermView', {})
                st_clicks = int(m.get('clicks', 0) or 0)
                st_impr = int(m.get('impressions', 0) or 0)
                st_spend = int(m.get('costMicros', 0) or 0) / 1_000_000
                st_conv = float(m.get('conversions', 0) or 0)
                search_terms.append({
                    'campaignName': row.get('campaign', {}).get('name', ''),
                    'adGroupName': row.get('adGroup', {}).get('name', ''),
                    'searchTerm': stv.get('searchTerm', ''),
                    'status': stv.get('status', 'NONE'),
                    'impressions': st_impr,
                    'clicks': st_clicks,
                    'spend': round(st_spend, 2),
                    'conversions': round(st_conv, 2),
                    'ctr': round((st_clicks / st_impr) * 100, 2) if st_impr > 0 else 0,
                })
        except Exception as e:
            logger.warning(f'Failed to fetch search term data: {str(e)}')

        # --- Ad copy performance ---
        ads = []
        try:
            ad_query = f"""
                SELECT
                    campaign.name,
                    ad_group.name,
                    ad_group_ad.ad.id,
                    ad_group_ad.ad.responsive_search_ad.headlines,
                    ad_group_ad.ad.responsive_search_ad.descriptions,
                    ad_group_ad.ad.final_urls,
                    ad_group_ad.status,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions
                FROM ad_group_ad
                WHERE segments.date BETWEEN '{date_range["startDate"]}' AND '{date_range["endDate"]}'
                    AND campaign.status = 'ENABLED'
                    AND ad_group_ad.status != 'REMOVED'
                    AND ad_group_ad.ad.type = 'RESPONSIVE_SEARCH_AD'
                    AND metrics.impressions > 0
                    {campaign_clause}
                ORDER BY metrics.clicks DESC
                LIMIT 20
            """
            ad_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, ad_query)
            for row in ad_results:
                m = row.get('metrics', {})
                aga = row.get('adGroupAd', {})
                ad = aga.get('ad', {})
                rsa = ad.get('responsiveSearchAd', {})
                ad_clicks = int(m.get('clicks', 0) or 0)
                ad_impr = int(m.get('impressions', 0) or 0)
                ad_spend = int(m.get('costMicros', 0) or 0) / 1_000_000
                ad_conv = float(m.get('conversions', 0) or 0)
                headlines = [h.get('text', '') for h in (rsa.get('headlines') or [])]
                descriptions = [d.get('text', '') for d in (rsa.get('descriptions') or [])]
                final_urls = ad.get('finalUrls') or []
                ads.append({
                    'adId': str(ad.get('id', '')),
                    'campaignName': row.get('campaign', {}).get('name', ''),
                    'adGroupName': row.get('adGroup', {}).get('name', ''),
                    'headlines': headlines,
                    'descriptions': descriptions,
                    'finalUrl': final_urls[0] if final_urls else '',
                    'status': aga.get('status', ''),
                    'impressions': ad_impr,
                    'clicks': ad_clicks,
                    'spend': round(ad_spend, 2),
                    'conversions': round(ad_conv, 2),
                    'ctr': round((ad_clicks / ad_impr) * 100, 2) if ad_impr > 0 else 0,
                })
        except Exception as e:
            logger.warning(f'Failed to fetch ad copy data: {str(e)}')

        # Calculate daily budget from plan (use account timezone)
        import calendar
        ads_now = datetime.now(ZoneInfo('America/Los_Angeles'))
        days_in_month = calendar.monthrange(ads_now.year, ads_now.month)[1]
        # Map current calendar month to plan month index (May=1, June=2, ..., October=6)
        plan_month_index = ads_now.month - 4
        monthly_budget = 600  # default
        for pm in PLAN_MONTHS:
            if pm['month'] == plan_month_index:
                monthly_budget = pm['budget_planned']
                break
        daily_budget = round(monthly_budget / days_in_month, 2)

        return jsonify({
            'dailySpend': round(total_spend, 2),
            'clicks': total_clicks,
            'impressions': total_impressions,
            'cpc': round(total_spend / total_clicks, 2) if total_clicks > 0 else 0,
            'ctr': round((total_clicks / total_impressions) * 100, 2) if total_impressions > 0 else 0,
            'conversions': round(total_conversions, 2),
            'costPerConversion': round(total_spend / total_conversions, 2) if total_conversions > 0 else 0,
            'campaigns': campaigns,
            'keywords': keywords,
            'searchTerms': search_terms,
            'ads': ads,
            'targetCampaign': campaign_filter or 'DMHOA - DIY Response - Phrase - March',
            'campaignFilter': campaign_filter or None,
            'dailyBudget': daily_budget,
            'period': period,
            'dateRange': date_range,
            'isMockData': False,
            'dataFrom': '2026-01-01',
        })

    except Exception as e:
        logger.error(f'Google Ads API error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch Google Ads data',
            'message': str(e),
            'isMockData': True,
        }), 500


# ============================================================================
# GOOGLE ADS — CUSTOMER MATCH (Similar Audiences)
# ============================================================================

def _google_ads_api_headers(access_token: str) -> Dict[str, str]:
    """Common headers for Google Ads API mutate requests."""
    headers = {
        'Authorization': f'Bearer {access_token}',
        'developer-token': GOOGLE_ADS_DEVELOPER_TOKEN,
        'Content-Type': 'application/json',
    }
    if GOOGLE_ADS_LOGIN_CUSTOMER_ID:
        headers['login-customer-id'] = GOOGLE_ADS_LOGIN_CUSTOMER_ID
    return headers


def _hash_email(email: str) -> str:
    """Normalize and SHA-256 hash an email for Google Ads."""
    return hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()


@dashboard_bp.route('/api/dashboard/google-ads/customer-match', methods=['POST', 'OPTIONS'])
def google_ads_customer_match():
    """Upload lead emails to Google Ads as a Customer Match list for Similar Audiences."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    has_credentials = all([
        GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CUSTOMER_ID,
        GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN
    ])
    if not has_credentials:
        return jsonify({'error': 'Google Ads credentials not configured'}), 400

    try:
        access_token = get_google_ads_access_token()
        if not access_token:
            raise Exception('Failed to get Google Ads access token')

        headers = _google_ads_api_headers(access_token)
        customer_id = GOOGLE_ADS_CUSTOMER_ID.replace('-', '')

        # 1. Fetch all lead emails from Supabase (top-level + payload fallback)
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {'select': 'email,payload', 'order': 'created_at.desc', 'limit': '5000'}
        resp = requests.get(url, params=params, headers=supabase_headers(), timeout=TIMEOUT)

        if not resp.ok:
            raise Exception(f'Failed to fetch emails from Supabase: {resp.text}')

        cases = resp.json() or []
        excluded = {e.lower() for e in EXCLUDED_EMAILS}
        emails_set = set()
        for c in cases:
            email = (c.get('email') or '').strip()
            if not email:
                # Fallback: extract from payload JSONB
                payload = c.get('payload') or {}
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {}
                email = (payload.get('email') or payload.get('ownerEmail') or '').strip()
            if email and email.lower() not in excluded:
                emails_set.add(email.lower())
        emails = list(emails_set)

        if not emails:
            return jsonify({'error': 'No emails found to upload', 'count': 0}), 400

        # 2. Check if user list already exists (search by name)
        list_name = 'DMHOA Lead Signups'
        search_query = f"""
            SELECT user_list.id, user_list.name, user_list.size_for_display
            FROM user_list
            WHERE user_list.name = '{list_name}'
        """
        try:
            existing = query_google_ads(customer_id, access_token, search_query)
            user_list_resource = None
            if existing:
                user_list_id = existing[0].get('userList', {}).get('id')
                if user_list_id:
                    user_list_resource = f'customers/{customer_id}/userLists/{user_list_id}'
        except Exception:
            user_list_resource = None

        # 3. Create user list if it doesn't exist
        if not user_list_resource:
            create_url = f'{GOOGLE_ADS_API_BASE}/customers/{customer_id}/userLists:mutate'
            create_payload = {
                'operations': [{
                    'create': {
                        'name': list_name,
                        'description': 'DisputeMyHOA lead signups for Similar Audiences targeting',
                        'membershipLifeSpan': 540,
                        'crmBasedUserList': {
                            'uploadKeyType': 'CONTACT_INFO',
                            'dataSourceType': 'FIRST_PARTY'
                        }
                    }
                }]
            }
            create_resp = requests.post(create_url, headers=headers, json=create_payload, timeout=TIMEOUT)
            if not create_resp.ok:
                raise Exception(f'Failed to create user list: {create_resp.text}')

            result = create_resp.json()
            user_list_resource = result['results'][0]['resourceName']
            logger.info(f'Created Customer Match list: {user_list_resource}')

        # 4. Create offline user data job
        job_url = f'{GOOGLE_ADS_API_BASE}/customers/{customer_id}/offlineUserDataJobs:create'
        job_payload = {
            'job': {
                'type': 'CUSTOMER_MATCH_USER_LIST',
                'customerMatchUserListMetadata': {
                    'userList': user_list_resource
                }
            }
        }
        job_resp = requests.post(job_url, headers=headers, json=job_payload, timeout=TIMEOUT)
        if not job_resp.ok:
            raise Exception(f'Failed to create data job: {job_resp.text}')

        job_resource = job_resp.json()['resourceName']

        # 5. Add hashed emails in batches of 100
        batch_size = 100
        total_uploaded = 0

        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            operations = [{
                'create': {
                    'userIdentifiers': [{
                        'hashedEmail': _hash_email(email)
                    }]
                }
            } for email in batch]

            add_url = f'{GOOGLE_ADS_API_BASE}/{job_resource}:addOperations'
            add_payload = {'operations': operations}
            add_resp = requests.post(add_url, headers=headers, json=add_payload, timeout=TIMEOUT)

            if not add_resp.ok:
                logger.error(f'Failed to add batch {i}: {add_resp.text}')
                continue

            total_uploaded += len(batch)

        # 6. Run the job
        run_url = f'{GOOGLE_ADS_API_BASE}/{job_resource}:run'
        run_resp = requests.post(run_url, headers=headers, timeout=TIMEOUT)

        if not run_resp.ok:
            raise Exception(f'Failed to run data job: {run_resp.text}')

        logger.info(f'Customer Match: uploaded {total_uploaded} emails, job running')

        return jsonify({
            'success': True,
            'emailsUploaded': total_uploaded,
            'totalEmails': len(emails),
            'listName': list_name,
            'userListResource': user_list_resource,
            'jobResource': job_resource,
            'message': f'Uploaded {total_uploaded} lead emails to Google Ads. '
                       f'Similar Audiences will be available within 24-48 hours.'
        })

    except Exception as e:
        logger.error(f'Customer Match error: {str(e)}')
        return jsonify({'error': 'Failed to upload customer match list', 'message': str(e)}), 500


# ============================================================================
# GOOGLE ADS — OFFLINE CONVERSION IMPORT
# ============================================================================

@dashboard_bp.route('/api/dashboard/google-ads/offline-conversions', methods=['POST', 'OPTIONS'])
def google_ads_offline_conversions():
    """Upload paid cases as offline conversions to Google Ads for bidding optimization."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    has_credentials = all([
        GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CUSTOMER_ID,
        GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN
    ])
    if not has_credentials:
        return jsonify({'error': 'Google Ads credentials not configured'}), 400

    try:
        access_token = get_google_ads_access_token()
        if not access_token:
            raise Exception('Failed to get Google Ads access token')

        headers = _google_ads_api_headers(access_token)
        customer_id = GOOGLE_ADS_CUSTOMER_ID.replace('-', '')

        # 1. Fetch paid cases from Supabase (with payload fallback for email)
        url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
        params = {
            'select': 'id,token,email,payload,created_at,updated_at,amount_total,gclid',
            'status': 'eq.paid',
            'order': 'created_at.desc',
            'limit': '1000'
        }
        resp = requests.get(url, params=params, headers=supabase_headers(), timeout=TIMEOUT)

        if not resp.ok:
            raise Exception(f'Failed to fetch paid cases: {resp.text}')

        excluded = {e.lower() for e in EXCLUDED_EMAILS}
        raw_cases = resp.json() or []
        cases = []
        for c in raw_cases:
            email = (c.get('email') or '').strip()
            if not email:
                payload = c.get('payload') or {}
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {}
                email = (payload.get('email') or payload.get('ownerEmail') or '').strip()
            if email and email.lower() not in excluded:
                c['email'] = email
                cases.append(c)

        if not cases:
            return jsonify({'error': 'No paid cases found to upload', 'count': 0}), 400

        # 2. Ensure the conversion action exists — search for it
        conversion_action_name = 'Paid Case - Offline Import'
        search_query = f"""
            SELECT conversion_action.id, conversion_action.name
            FROM conversion_action
            WHERE conversion_action.name = '{conversion_action_name}'
        """
        conversion_action_resource = None
        try:
            results = query_google_ads(customer_id, access_token, search_query)
            if results:
                ca_id = results[0].get('conversionAction', {}).get('id')
                if ca_id:
                    conversion_action_resource = f'customers/{customer_id}/conversionActions/{ca_id}'
        except Exception:
            pass

        # 3. Create conversion action if it doesn't exist
        if not conversion_action_resource:
            create_url = f'{GOOGLE_ADS_API_BASE}/customers/{customer_id}/conversionActions:mutate'
            create_payload = {
                'operations': [{
                    'create': {
                        'name': conversion_action_name,
                        'type': 'UPLOAD_CLICKS',
                        'category': 'PURCHASE',
                        'status': 'ENABLED',
                        'valueSettings': {
                            'defaultValue': 49.0,
                            'defaultCurrencyCode': 'USD',
                            'alwaysUseDefaultValue': False
                        }
                    }
                }]
            }
            create_resp = requests.post(create_url, headers=headers, json=create_payload, timeout=TIMEOUT)
            if not create_resp.ok:
                raise Exception(f'Failed to create conversion action: {create_resp.text}')

            result = create_resp.json()
            conversion_action_resource = result['results'][0]['resourceName']
            logger.info(f'Created conversion action: {conversion_action_resource}')

        # 4. Build conversion list
        conversions = []
        for case in cases:
            email = case['email'].strip().lower()
            # Use updated_at (payment time) or created_at
            conversion_time = case.get('updated_at') or case['created_at']
            # Google Ads needs format: yyyy-mm-dd hh:mm:ss+|-hh:mm
            try:
                dt = datetime.fromisoformat(conversion_time.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
            except Exception:
                formatted_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+00:00')

            amount_cents = case.get('amount_total') or 4900
            amount_dollars = amount_cents / 100.0

            conversion = {
                'conversionAction': conversion_action_resource,
                'conversionDateTime': formatted_time,
                'conversionValue': amount_dollars,
                'currencyCode': 'USD',
                'orderId': case.get('token', f"case_{case['id']}"),
            }

            # Prefer gclid if available, otherwise use email for enhanced conversions
            if case.get('gclid'):
                conversion['gclid'] = case['gclid']
            else:
                conversion['userIdentifiers'] = [{
                    'hashedEmail': _hash_email(email)
                }]

            conversions.append(conversion)

        # 5. Upload conversions
        upload_url = f'{GOOGLE_ADS_API_BASE}/customers/{customer_id}:uploadClickConversions'
        upload_payload = {
            'conversions': conversions,
            'partialFailure': True
        }
        upload_resp = requests.post(upload_url, headers=headers, json=upload_payload, timeout=(5, 120))

        if not upload_resp.ok:
            raise Exception(f'Failed to upload conversions: {upload_resp.text}')

        result = upload_resp.json()
        partial_errors = result.get('partialFailureError')
        uploaded_count = len(conversions)
        error_count = 0

        if partial_errors:
            error_details = partial_errors.get('details', [])
            error_count = len(error_details)
            uploaded_count -= error_count
            logger.warning(f'Offline conversions: {error_count} partial failures')

        logger.info(f'Offline conversions: uploaded {uploaded_count} of {len(conversions)}')

        return jsonify({
            'success': True,
            'conversionsUploaded': uploaded_count,
            'totalCases': len(cases),
            'errors': error_count,
            'conversionAction': conversion_action_name,
            'conversionActionResource': conversion_action_resource,
            'message': f'Uploaded {uploaded_count} paid case conversions to Google Ads. '
                       f'Attribution data will be available within 24 hours.'
        })

    except Exception as e:
        logger.error(f'Offline conversion import error: {str(e)}')
        return jsonify({'error': 'Failed to upload offline conversions', 'message': str(e)}), 500


# ============================================================================
# POSTHOG ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/posthog', methods=['GET', 'OPTIONS'])
def get_posthog_data():
    """Get PostHog analytics data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not POSTHOG_PERSONAL_API_KEY or not POSTHOG_PROJECT_ID:
        # Return mock data
        return jsonify({
            'totalSessions': 0, 'totalPageViews': 0, 'pagesPerSession': 0,
            'avgScrollDepth': 0, 'avgTimeOnPage': 0, 'bounceRate': 0,
            'totalVisits': 0, 'uniqueVisitors': 0, 'returningVisitors': 0,
            'rageClicks': 0, 'deadClicks': 0, 'quickBacks': 0, 'excessiveScrolling': 0,
            'jsErrors': 0, 'slowPageLoads': 0,
            'dailyVisits': [], 'pagesWithIssues': [], 'topPages': [],
            'webVitals': {'fcp': {'p75': 0, 'p90': 0}, 'lcp': {'p75': 0, 'p90': 0},
                          'cls': {'p75': 0, 'p90': 0}, 'inp': {'p75': 0, 'p90': 0}, 'sampleCount': 0},
            'funnel': {'landing': 0, 'preview': 0, 'purchase': 0},
            'suggestions': [], 'compositeGrade': {'rumScore': 0, 'uxScore': 0, 'engagementScore': 0},
            'isMockData': True, 'message': 'PostHog not configured.',
        })

    try:
        # Check cache first
        force_refresh = request.args.get('refresh') == 'true'
        cache_key = 'posthog_data'

        if not force_refresh and SUPABASE_URL:
            cache_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/api_cache",
                params={'cache_key': f'eq.{cache_key}', 'select': 'data,updated_at'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if cache_response.ok:
                cache_data = cache_response.json()
                if cache_data:
                    updated_at = datetime.fromisoformat(cache_data[0]['updated_at'].replace('Z', '+00:00'))
                    cache_age = (datetime.now(updated_at.tzinfo) - updated_at).total_seconds()
                    if cache_age < 12 * 60 * 60:  # 12 hours
                        return jsonify({
                            **cache_data[0]['data'],
                            'fromCache': True,
                            'cacheAge': f'{int(cache_age / 60)} minutes',
                        })

        posthog_headers = {
            'Authorization': f'Bearer {POSTHOG_PERSONAL_API_KEY}',
            'Content-Type': 'application/json',
        }

        posthog_query_url = f'{POSTHOG_API_URL}/api/projects/{POSTHOG_PROJECT_ID}/query/'

        def run_hogql(query_str):
            """Execute a single HogQL query and return the response."""
            return requests.post(
                posthog_query_url,
                headers=posthog_headers,
                json={'query': {'kind': 'HogQLQuery', 'query': query_str}},
                timeout=(10, 30)
            )

        # Define all 10 HogQL queries
        hogql_queries = {
            'sessions': """
                SELECT
                    count(DISTINCT $session_id) as total_sessions,
                    count(DISTINCT person_id) as unique_users,
                    avg(session.$session_duration) as avg_session_duration,
                    count(*) as total_pageviews,
                    count(*) / greatest(count(DISTINCT $session_id), 1) as pages_per_session,
                    countIf(session.$session_duration < 10) * 100.0 / greatest(count(DISTINCT $session_id), 1) as bounce_rate
                FROM events
                WHERE event = '$pageview'
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
            'rage': """
                SELECT count(*) as rage_clicks
                FROM events
                WHERE event = '$rageclick'
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
            'dead': """
                SELECT count(*) as dead_clicks
                FROM events
                WHERE event = '$dead_click'
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
            'js_errors': """
                SELECT count(*) as js_errors
                FROM events
                WHERE event = '$exception'
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
            'slow_loads': """
                SELECT count(DISTINCT $session_id) as slow_loads
                FROM events
                WHERE event = '$web_vitals'
                  AND properties.$web_vitals_LCP_value > 3000
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
            'daily_visits': """
                SELECT
                    toDate(timestamp) as day,
                    count(*) as pageviews,
                    count(DISTINCT $session_id) as sessions
                FROM events
                WHERE event = '$pageview'
                  AND timestamp > now() - INTERVAL 3 DAY
                GROUP BY day
                ORDER BY day ASC
            """,
            'pages_issues': """
                SELECT
                    properties.$current_url as page_url,
                    countIf(event = '$rageclick') as rage_clicks,
                    countIf(event = '$dead_click') as dead_clicks
                FROM events
                WHERE event IN ('$rageclick', '$dead_click')
                  AND timestamp > now() - INTERVAL 3 DAY
                GROUP BY page_url
                ORDER BY rage_clicks + dead_clicks DESC
                LIMIT 10
            """,
            'web_vitals': """
                SELECT
                    quantile(0.75)(properties.$web_vitals_FCP_value) as fcp_p75,
                    quantile(0.90)(properties.$web_vitals_FCP_value) as fcp_p90,
                    quantile(0.75)(properties.$web_vitals_LCP_value) as lcp_p75,
                    quantile(0.90)(properties.$web_vitals_LCP_value) as lcp_p90,
                    quantile(0.75)(properties.$web_vitals_CLS_value) as cls_p75,
                    quantile(0.90)(properties.$web_vitals_CLS_value) as cls_p90,
                    quantile(0.75)(properties.$web_vitals_INP_value) as inp_p75,
                    quantile(0.90)(properties.$web_vitals_INP_value) as inp_p90,
                    count(*) as sample_count
                FROM events
                WHERE event = '$web_vitals'
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
            'top_pages': """
                SELECT
                    properties.$current_url as page_url,
                    count(*) as views
                FROM events
                WHERE event = '$pageview'
                  AND timestamp > now() - INTERVAL 3 DAY
                GROUP BY page_url
                ORDER BY views DESC
                LIMIT 5
            """,
            'funnel': """
                SELECT
                    count(DISTINCT $session_id) as landing,
                    count(DISTINCT CASE WHEN properties.$current_url LIKE '%/preview%' OR properties.$current_url LIKE '%/quick-preview%' THEN $session_id END) as preview,
                    count(DISTINCT CASE WHEN properties.$current_url LIKE '%/checkout%' OR properties.$current_url LIKE '%/purchase%' OR properties.$current_url LIKE '%/payment%' THEN $session_id END) as purchase
                FROM events
                WHERE event = '$pageview'
                  AND timestamp > now() - INTERVAL 3 DAY
            """,
        }

        # Execute all queries in parallel
        responses = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_key = {executor.submit(run_hogql, q): k for k, q in hogql_queries.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    responses[key] = future.result()
                except Exception as e:
                    logger.warning(f'PostHog query "{key}" failed: {e}')
                    responses[key] = None

        # --- Parse results ---

        # Sessions
        sessions_data = {'total_sessions': 0, 'unique_users': 0, 'avg_session_duration': 0,
                         'total_pageviews': 0, 'pages_per_session': 0, 'bounce_rate': 0}
        resp = responses.get('sessions')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                row = rows[0]
                sessions_data = {
                    'total_sessions': int(row[0] or 0),
                    'unique_users': int(row[1] or 0),
                    'avg_session_duration': round(float(row[2] or 0)),
                    'total_pageviews': int(row[3] or 0),
                    'pages_per_session': round(float(row[4] or 0), 1),
                    'bounce_rate': round(float(row[5] or 0), 1),
                }

        # Rage clicks
        rage_clicks = 0
        resp = responses.get('rage')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                rage_clicks = int(rows[0][0] or 0)

        # Dead clicks
        dead_clicks = 0
        resp = responses.get('dead')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                dead_clicks = int(rows[0][0] or 0)

        # JS errors
        js_errors = 0
        resp = responses.get('js_errors')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                js_errors = int(rows[0][0] or 0)

        # Slow page loads
        slow_page_loads = 0
        resp = responses.get('slow_loads')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                slow_page_loads = int(rows[0][0] or 0)

        # Daily visits
        daily_visits = []
        resp = responses.get('daily_visits')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            for row in rows:
                daily_visits.append({
                    'date': str(row[0]),
                    'visits': int(row[1] or 0),
                    'sessions': int(row[2] or 0),
                })

        # Pages with issues
        pages_with_issues = []
        resp = responses.get('pages_issues')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            for row in rows:
                url = str(row[0] or '')
                page = url.replace('https://disputemyhoa.com', '').replace('http://disputemyhoa.com', '') or '/'
                pages_with_issues.append({
                    'page': page,
                    'rageClicks': int(row[1] or 0),
                    'deadClicks': int(row[2] or 0),
                })

        # Real User Web Vitals
        web_vitals = {
            'fcp': {'p75': 0, 'p90': 0}, 'lcp': {'p75': 0, 'p90': 0},
            'cls': {'p75': 0, 'p90': 0}, 'inp': {'p75': 0, 'p90': 0},
            'sampleCount': 0,
        }
        resp = responses.get('web_vitals')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                row = rows[0]
                web_vitals = {
                    'fcp': {'p75': round(float(row[0] or 0)), 'p90': round(float(row[1] or 0))},
                    'lcp': {'p75': round(float(row[2] or 0)), 'p90': round(float(row[3] or 0))},
                    'cls': {'p75': round(float(row[4] or 0), 3), 'p90': round(float(row[5] or 0), 3)},
                    'inp': {'p75': round(float(row[6] or 0)), 'p90': round(float(row[7] or 0))},
                    'sampleCount': int(row[8] or 0),
                }

        # Top pages
        top_pages = []
        resp = responses.get('top_pages')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            for row in rows:
                url = str(row[0] or '')
                page = url.replace('https://disputemyhoa.com', '').replace('http://disputemyhoa.com', '') or '/'
                top_pages.append({'page': page, 'views': int(row[1] or 0)})

        # Conversion funnel
        funnel = {'landing': 0, 'preview': 0, 'purchase': 0}
        resp = responses.get('funnel')
        if resp and resp.ok:
            rows = resp.json().get('results', [])
            if rows and len(rows) > 0:
                row = rows[0]
                funnel = {
                    'landing': int(row[0] or 0),
                    'preview': int(row[1] or 0),
                    'purchase': int(row[2] or 0),
                }

        # --- Performance suggestions ---
        suggestions = []

        if web_vitals['sampleCount'] > 0:
            if web_vitals['lcp']['p75'] > 2500:
                suggestions.append({'type': 'warning', 'category': 'Web Vitals', 'title': 'Optimize Largest Contentful Paint',
                    'description': f"LCP p75 is {web_vitals['lcp']['p75']}ms (target: <2500ms). Consider optimizing images, preloading key resources, and reducing server response time."})
            if web_vitals['cls']['p75'] > 0.1:
                suggestions.append({'type': 'warning', 'category': 'Web Vitals', 'title': 'Fix Layout Shifts',
                    'description': f"CLS p75 is {web_vitals['cls']['p75']} (target: <0.1). Set explicit dimensions on images/videos and avoid dynamic content insertion above the fold."})
            if web_vitals['inp']['p75'] > 200:
                suggestions.append({'type': 'warning', 'category': 'Web Vitals', 'title': 'Improve Interaction Responsiveness',
                    'description': f"INP p75 is {web_vitals['inp']['p75']}ms (target: <200ms). Reduce JavaScript execution time and break up long tasks."})
            if web_vitals['fcp']['p75'] > 1800:
                suggestions.append({'type': 'info', 'category': 'Web Vitals', 'title': 'Optimize First Contentful Paint',
                    'description': f"FCP p75 is {web_vitals['fcp']['p75']}ms (target: <1800ms). Consider inlining critical CSS and deferring non-essential scripts."})

        if rage_clicks > 5:
            suggestions.append({'type': 'warning', 'category': 'User Experience', 'title': 'Investigate Rage Clicks',
                'description': f"{rage_clicks} rage clicks detected in 3 days. Users are repeatedly clicking on non-responsive elements."})
        if sessions_data['bounce_rate'] > 70:
            suggestions.append({'type': 'warning', 'category': 'Engagement', 'title': 'High Bounce Rate',
                'description': f"Bounce rate is {sessions_data['bounce_rate']}% (target: <70%). Review landing page content, load speed, and above-the-fold messaging."})
        if js_errors > 3:
            suggestions.append({'type': 'error', 'category': 'Technical', 'title': 'Fix JavaScript Errors',
                'description': f"{js_errors} JS errors in 3 days. Check PostHog error tracking for stack traces and fix the root causes."})
        if slow_page_loads > 3:
            suggestions.append({'type': 'warning', 'category': 'Performance', 'title': 'Address Slow Page Loads',
                'description': f"{slow_page_loads} sessions experienced LCP > 3s. Optimize images, reduce JS bundles, and enable caching."})

        if not suggestions:
            suggestions.append({'type': 'success', 'category': 'Overall', 'title': 'Site Performing Well',
                'description': 'No major performance or UX issues detected in the last 3 days. Keep monitoring.'})

        # --- Composite grade components ---
        def _web_vital_score(lcp, cls_val, fcp, inp):
            score = 0
            score += 25 if lcp <= 2500 else (15 if lcp <= 4000 else 5)
            score += 25 if cls_val <= 0.1 else (15 if cls_val <= 0.25 else 5)
            score += 25 if fcp <= 1800 else (15 if fcp <= 3000 else 5)
            score += 25 if inp <= 200 else (15 if inp <= 500 else 5)
            return score

        rum_score = _web_vital_score(
            web_vitals['lcp']['p75'], web_vitals['cls']['p75'],
            web_vitals['fcp']['p75'], web_vitals['inp']['p75']
        ) if web_vitals['sampleCount'] > 0 else 50

        frustration_total = rage_clicks + dead_clicks
        ux_score = max(0, min(100, 100 - (frustration_total * 2) - (js_errors * 5)))

        bounce_penalty = max(0, sessions_data['bounce_rate'] - 40) * 1.5
        pages_bonus = min(sessions_data['pages_per_session'] * 15, 50)
        time_bonus = min(sessions_data['avg_session_duration'] * 0.5, 50)
        engagement_score = max(0, min(100, pages_bonus + time_bonus - bounce_penalty))

        data = {
            'totalSessions': sessions_data['total_sessions'],
            'totalPageViews': sessions_data['total_pageviews'],
            'pagesPerSession': sessions_data['pages_per_session'],
            'avgScrollDepth': 0,
            'avgTimeOnPage': sessions_data['avg_session_duration'],
            'bounceRate': sessions_data['bounce_rate'],
            'totalVisits': sessions_data['total_sessions'],
            'uniqueVisitors': sessions_data['unique_users'],
            'returningVisitors': 0,
            'rageClicks': rage_clicks,
            'deadClicks': dead_clicks,
            'quickBacks': 0,
            'excessiveScrolling': 0,
            'jsErrors': js_errors,
            'slowPageLoads': slow_page_loads,
            'dailyVisits': daily_visits,
            'pagesWithIssues': pages_with_issues,
            'webVitals': web_vitals,
            'topPages': top_pages,
            'funnel': funnel,
            'suggestions': suggestions,
            'compositeGrade': {
                'rumScore': rum_score,
                'uxScore': round(ux_score),
                'engagementScore': round(engagement_score),
            },
            'isMockData': False,
            'dataRange': 'Last 3 days',
            'lastFetched': datetime.now().isoformat(),
        }

        # Cache the data
        if SUPABASE_URL:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/api_cache",
                headers={**supabase_headers(), 'Prefer': 'resolution=merge-duplicates'},
                json={
                    'cache_key': cache_key,
                    'data': data,
                    'updated_at': datetime.now().isoformat(),
                },
                timeout=TIMEOUT
            )

        return jsonify(data)

    except Exception as e:
        logger.error(f'PostHog API error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch PostHog data',
            'message': str(e),
            'isMockData': True,
        }), 500


# ============================================================================
# LIGHTHOUSE (PAGESPEED) ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/lighthouse', methods=['GET', 'OPTIONS'])
def get_lighthouse_data():
    """Get Google PageSpeed Insights data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not GOOGLE_PAGESPEED_API_KEY:
        return jsonify({
            'performanceScore': 0,
            'seoScore': 0,
            'accessibilityScore': 0,
            'bestPracticesScore': 0,
            'isMockData': True,
            'message': 'PageSpeed API key not configured.',
        })

    try:
        # Check cache
        force_refresh = request.args.get('refresh') == 'true'
        cache_key = 'lighthouse_data'

        if not force_refresh and SUPABASE_URL:
            cache_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/api_cache",
                params={'cache_key': f'eq.{cache_key}', 'select': 'data,updated_at'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if cache_response.ok:
                cache_data = cache_response.json()
                if cache_data:
                    updated_at = datetime.fromisoformat(cache_data[0]['updated_at'].replace('Z', '+00:00'))
                    cache_age = (datetime.now(updated_at.tzinfo) - updated_at).total_seconds()
                    if cache_age < 6 * 60 * 60:  # 6 hours
                        return jsonify({
                            **cache_data[0]['data'],
                            'fromCache': True,
                            'cacheAge': f'{int(cache_age / 60)} minutes',
                        })

        # Fetch from PageSpeed API
        params = {
            'url': TARGET_URL,
            'strategy': 'mobile',
            'category': ['performance', 'seo', 'accessibility', 'best-practices'],
            'key': GOOGLE_PAGESPEED_API_KEY,
        }

        try:
            response = requests.get(PAGESPEED_API_URL, params=params, timeout=(10, 60))
        except requests.exceptions.RequestException as req_err:
            logger.warning(f'PageSpeed API request failed: {req_err}')
            response = None

        if not response or not response.ok:
            status = response.status_code if response else 'timeout'
            logger.warning(f'PageSpeed API returned {status}, returning stale cache or empty data')
            # Try returning stale cache regardless of age
            if SUPABASE_URL:
                stale = requests.get(
                    f"{SUPABASE_URL}/rest/v1/api_cache",
                    params={'cache_key': f'eq.{cache_key}', 'select': 'data,updated_at'},
                    headers=supabase_headers(), timeout=TIMEOUT
                )
                if stale.ok and stale.json():
                    stale_data = stale.json()[0]['data']
                    updated_at = datetime.fromisoformat(stale_data.get('lastTested', datetime.now().isoformat()).replace('Z', '+00:00'))
                    return jsonify({**stale_data, 'fromCache': True, 'stale': True,
                                    'message': f'PageSpeed API unavailable ({status}). Showing last cached data.'})
            return jsonify({
                'performanceScore': 0, 'seoScore': 0, 'accessibilityScore': 0, 'bestPracticesScore': 0,
                'isMockData': True, 'message': f'PageSpeed API unavailable ({status}). Try again later.',
            })

        api_data = response.json()
        lighthouse = api_data.get('lighthouseResult', {})
        categories = lighthouse.get('categories', {})
        audits = lighthouse.get('audits', {})

        def get_metric(audit_name):
            audit = audits.get(audit_name, {})
            return {
                'value': audit.get('numericValue', 0),
                'displayValue': audit.get('displayValue', 'N/A'),
                'score': round((audit.get('score', 0) or 0) * 100),
            }

        data = {
            'url': TARGET_URL,
            'strategy': 'mobile',
            'performanceScore': round((categories.get('performance', {}).get('score', 0) or 0) * 100),
            'seoScore': round((categories.get('seo', {}).get('score', 0) or 0) * 100),
            'accessibilityScore': round((categories.get('accessibility', {}).get('score', 0) or 0) * 100),
            'bestPracticesScore': round((categories.get('best-practices', {}).get('score', 0) or 0) * 100),
            'fcp': get_metric('first-contentful-paint'),
            'lcp': get_metric('largest-contentful-paint'),
            'tbt': get_metric('total-blocking-time'),
            'cls': get_metric('cumulative-layout-shift'),
            'speedIndex': get_metric('speed-index'),
            'tti': get_metric('interactive'),
            'serverResponseTime': get_metric('server-response-time'),
            'lastTested': datetime.now().isoformat(),
            'isMockData': False,
        }

        # Cache the data
        if SUPABASE_URL:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/api_cache",
                headers={**supabase_headers(), 'Prefer': 'resolution=merge-duplicates'},
                json={
                    'cache_key': cache_key,
                    'data': data,
                    'updated_at': datetime.now().isoformat(),
                },
                timeout=TIMEOUT
            )

        return jsonify(data)

    except Exception as e:
        logger.error(f'Lighthouse API error: {str(e)}')
        return jsonify({
            'performanceScore': 0, 'seoScore': 0, 'accessibilityScore': 0, 'bestPracticesScore': 0,
            'error': 'Failed to fetch Lighthouse data',
            'message': str(e),
            'isMockData': True,
        })


# ============================================================================
# OPENAI USAGE ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/openai-usage', methods=['GET', 'OPTIONS'])
def get_openai_usage():
    """Get OpenAI usage and cost data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not OPENAI_ADMIN_KEY:
        return jsonify({
            'error': 'OpenAI Admin API key not configured',
            'isMockData': True,
            'totalCost': 0,
            'todayCost': 0,
            'totalRequests': 0,
            'avgCostPerRequest': 0,
            'costPerPurchase': 0,
            'profitMargin': 0,
            'dailySpendAlert': False,
            'dailyData': [],
        })

    try:
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        today = end_date.strftime('%Y-%m-%d')

        # Fetch usage from OpenAI Admin API
        url = 'https://api.openai.com/v1/organization/usage/completions'
        params = {
            'start_time': int(start_date.timestamp()),
            'end_time': int(end_date.timestamp()),
            'bucket_width': '1d',
        }

        response = requests.get(
            url,
            params=params,
            headers={
                'Authorization': f'Bearer {OPENAI_ADMIN_KEY}',
                'Content-Type': 'application/json',
            },
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'OpenAI API error: {response.status_code} - {response.text}')

        usage_data = response.json()
        buckets = usage_data.get('data', []) or usage_data.get('buckets', [])

        daily_data = []
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_requests = 0
        today_cost = 0

        for bucket in buckets:
            bucket_date = datetime.fromtimestamp(bucket.get('start_time', 0)).strftime('%Y-%m-%d') if bucket.get('start_time') else 'unknown'

            day_cost = 0
            day_input = 0
            day_output = 0
            day_requests = 0

            results = bucket.get('results', [bucket])
            for item in results:
                input_tokens = item.get('input_tokens', 0) or item.get('prompt_tokens', 0) or 0
                output_tokens = item.get('output_tokens', 0) or item.get('completion_tokens', 0) or 0
                model = item.get('model', 'default')
                requests_count = item.get('num_model_requests', 1)

                pricing = OPENAI_PRICING.get(model, OPENAI_PRICING['default'])
                cost = (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']

                day_cost += cost
                day_input += input_tokens
                day_output += output_tokens
                day_requests += requests_count

            if bucket_date == today:
                today_cost = day_cost

            if day_input > 0 or day_output > 0:
                daily_data.append({
                    'date': bucket_date,
                    'cost': round(day_cost, 2),
                    'inputTokens': day_input,
                    'outputTokens': day_output,
                    'requests': day_requests,
                })

            total_cost += day_cost
            total_input_tokens += day_input
            total_output_tokens += day_output
            total_requests += day_requests

        # Get purchase count from Supabase
        purchase_count = 0
        if SUPABASE_URL:
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            purchases_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
                params={
                    'select': 'id',
                    'status': 'eq.paid',
                    'created_at': f'gte.{thirty_days_ago}',
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if purchases_response.ok:
                purchase_count = len(purchases_response.json())

        avg_cost_per_request = round(total_cost / total_requests, 3) if total_requests > 0 else 0
        cost_per_purchase = round(total_cost / purchase_count, 2) if purchase_count > 0 else 0
        price_per_purchase = 29
        profit_per_purchase = price_per_purchase - cost_per_purchase
        profit_margin = round((profit_per_purchase / price_per_purchase) * 100, 1) if cost_per_purchase > 0 else 100

        return jsonify({
            'totalCost': round(total_cost, 2),
            'todayCost': round(today_cost, 2),
            'totalRequests': total_requests,
            'totalInputTokens': total_input_tokens,
            'totalOutputTokens': total_output_tokens,
            'avgCostPerRequest': avg_cost_per_request,
            'purchaseCount': purchase_count,
            'costPerPurchase': cost_per_purchase,
            'pricePerPurchase': price_per_purchase,
            'profitPerPurchase': round(profit_per_purchase, 2),
            'profitMargin': profit_margin,
            'credits': None,
            'budgetLimit': {
                'monthlyLimit': OPENAI_MONTHLY_BUDGET,
                'remaining': max(0, OPENAI_MONTHLY_BUDGET - total_cost),
                'percentUsed': round((total_cost / OPENAI_MONTHLY_BUDGET) * 100) if OPENAI_MONTHLY_BUDGET > 0 else 0,
            } if OPENAI_MONTHLY_BUDGET > 0 else None,
            'dailySpendAlert': today_cost > 50,
            'dailySpendThreshold': 50,
            'dailyData': sorted(daily_data, key=lambda x: x['date'])[-7:],
            'period': '30 days',
            'lastUpdated': datetime.now().isoformat(),
            'isMockData': False,
        })

    except Exception as e:
        logger.error(f'OpenAI usage error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch OpenAI usage data',
            'message': str(e),
            'isMockData': True,
            'totalCost': 0,
            'todayCost': 0,
            'totalRequests': 0,
            'dailyData': [],
        }), 500


# ============================================================================
# HOA NEWS ENDPOINT
# ============================================================================

HOA_QUERIES = [
    'HOA homeowners association news',
    'HOA legislation law 2026',
    'homeowners association disputes fines',
    'community association management',
    'HOA reform homeowner rights',
]

NEWS_MAX_AGE_DAYS = 30


def categorize_article(title: str, description: str) -> str:
    """Categorize an article based on content."""
    text = (title + ' ' + description).lower()

    if any(word in text for word in ['legislation', 'law', 'bill', 'senate', 'house']):
        return 'legislation'
    if any(word in text for word in ['enforcement', 'fine', 'violation', 'compliance']):
        return 'enforcement'
    if any(word in text for word in ['court', 'lawsuit', 'ruling', 'judge']):
        return 'legal'
    if any(word in text for word in ['fee', 'assessment', 'budget', 'reserve']):
        return 'financial'
    return 'general'


def get_priority(title: str, description: str) -> str:
    """Get priority based on HOA term mentions."""
    text = (title + ' ' + description).lower()
    hoa_terms = ['hoa', 'homeowners association', 'homeowner association', 'condo association',
                 'community association', 'property owners association']

    match_count = sum(1 for term in hoa_terms if term in text)

    if match_count >= 2:
        return 'high'
    if match_count == 1:
        return 'medium'
    return 'low'


def _clean_html(text):
    """Strip HTML tags and decode entities from text."""
    if not text:
        return text
    import re
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'<[^>]*$', '', text)
    return text.strip()


def _fetch_google_news_rss(query):
    """Fetch articles from Google News RSS for a given query."""
    import xml.etree.ElementTree as ET
    encoded_query = requests.utils.quote(query)
    url = f'https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en'
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; HOADashboard/1.0)'}, timeout=10)
        if not resp.ok:
            return []
        root = ET.fromstring(resp.text)
        items = []
        for item in root.findall('.//item'):
            title = _clean_html(item.findtext('title', ''))
            link = item.findtext('link', '')
            description = _clean_html(item.findtext('description', '') or '')
            if description:
                description = description[:300]
            pub_date = item.findtext('pubDate', '')
            source_el = item.find('source')
            source = _clean_html(source_el.text) if source_el is not None and source_el.text else ''
            if not source and link:
                try:
                    from urllib.parse import urlparse
                    source = urlparse(link).hostname.replace('www.', '')
                except Exception:
                    source = 'Unknown'
            if title and link:
                items.append({
                    'title': title,
                    'link': link,
                    'description': description,
                    'pub_date': pub_date,
                    'source': source,
                    'query': query,
                    'fetched_from': 'google_news',
                })
        return items
    except Exception as e:
        logger.warning(f'Failed to fetch Google News for "{query}": {e}')
        return []


def scan_hoa_news():
    """Fetch, dedupe, and save HOA news articles. Returns dict with counts.
    Callable from the route handler or a scheduler job."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import timedelta, timezone as _tz
    from email.utils import parsedate_to_datetime

    all_articles = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch_google_news_rss, q): q for q in HOA_QUERIES}
        for future in as_completed(futures, timeout=20):
            try:
                all_articles.extend(future.result())
            except Exception:
                pass

    seen = set()
    unique = []
    for a in all_articles:
        norm = a['title'].lower()[:50]
        if norm not in seen:
            seen.add(norm)
            unique.append(a)

    existing_resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
        params={'select': 'link', 'limit': '500'},
        headers=supabase_headers(),
        timeout=TIMEOUT
    )
    existing_links = set()
    if existing_resp.ok:
        existing_links = {a.get('link') for a in existing_resp.json()}

    new_articles = [a for a in unique if a['link'] not in existing_links]
    saved = 0
    skipped_old = 0
    now = datetime.now().isoformat()
    cutoff = datetime.now(_tz.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)

    for a in new_articles:
        category = categorize_article(a['title'], a.get('description', ''))
        priority = get_priority(a['title'], a.get('description', ''))

        pub_date_iso = None
        pub_dt = None
        if a.get('pub_date'):
            try:
                pub_dt = parsedate_to_datetime(a['pub_date'])
                pub_date_iso = pub_dt.isoformat()
            except Exception:
                pub_dt = None

        if pub_dt and pub_dt < cutoff:
            skipped_old += 1
            continue

        insert_data = {
            'link': a['link'],
            'title': a['title'],
            'description': a.get('description', ''),
            'source': a.get('source', ''),
            'pub_date': pub_date_iso,
            'query': a.get('query'),
            'fetched_from': a.get('fetched_from', 'google_news'),
            'category': category,
            'priority': priority,
            'status': 'new',
            'first_seen_at': now,
            'last_seen_at': now,
        }

        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
            headers=supabase_headers(),
            json=insert_data,
            timeout=TIMEOUT
        )
        if resp.ok or resp.status_code == 201:
            saved += 1

    return {
        'scanned': len(unique),
        'saved': saved,
        'skipped_old': skipped_old,
        'last_updated': now,
    }


@dashboard_bp.route('/api/dashboard/hoa-news', methods=['GET', 'POST', 'PATCH', 'OPTIONS'])
def handle_hoa_news():
    """Get, scan, or update HOA news articles."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    # Handle POST for scanning new articles from RSS
    if request.method == 'POST':
        try:
            r = scan_hoa_news()
            return jsonify({
                'articles': [],
                'count': r['saved'],
                'lastUpdated': r['last_updated'],
                'message': f"Scanned {r['scanned']} articles, saved {r['saved']} new (skipped {r['skipped_old']} older than {NEWS_MAX_AGE_DAYS}d)"
            })
        except Exception as e:
            logger.error(f'HOA News scan error: {str(e)}')
            return jsonify({'error': 'Failed to scan news', 'message': str(e)}), 500

    # Handle PATCH for updating article status
    if request.method == 'PATCH':
        try:
            data = request.get_json() or {}

            # Support { id, status } format (from Intel service)
            if data.get('id') and data.get('status'):
                article_id = data['id']
                status = data['status']
                if status not in ('new', 'reviewed', 'archived'):
                    return jsonify({'error': 'Invalid status'}), 400

                update_data = {'status': status}
                if status == 'archived':
                    update_data['dismissed'] = True

                response = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                    params={'id': f'eq.{article_id}'},
                    headers=supabase_headers(),
                    json=update_data,
                    timeout=TIMEOUT
                )
                if not response.ok:
                    raise Exception(f'Failed to update article: {response.text}')

                return jsonify({'success': True, 'article': {'id': article_id, 'status': status}})

            # Support { id, action: 'saveNotes', notes } format
            if data.get('id') and data.get('action') == 'saveNotes':
                article_id = data['id']
                notes_text = data.get('notes', '')
                update_data = {'notes': notes_text if notes_text.strip() else None}

                response = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                    params={'id': f'eq.{article_id}'},
                    headers=supabase_headers(),
                    json=update_data,
                    timeout=TIMEOUT
                )
                if not response.ok:
                    raise Exception(f'Failed to save notes: {response.text}')

                return jsonify({'success': True, 'article': {'id': article_id, 'notes': update_data['notes']}})

            # Support { articleId, action } format (from News service)
            article_id = data.get('articleId')
            action = data.get('action')

            if not article_id or not action:
                return jsonify({'error': 'Missing articleId/action or id/status'}), 400

            update_data = {}
            if action == 'bookmark':
                update_data = {'bookmarked': True}
            elif action == 'unbookmark':
                update_data = {'bookmarked': False}
            elif action == 'dismiss':
                update_data = {'dismissed': True, 'status': 'archived'}
            elif action == 'undismiss':
                update_data = {'dismissed': False, 'status': 'new'}
            elif action == 'markUsed':
                update_data = {'used_for_content': True}
            elif action == 'unmarkUsed':
                update_data = {'used_for_content': False}
            else:
                return jsonify({'error': 'Invalid action'}), 400

            response = requests.patch(
                f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                params={'id': f'eq.{article_id}'},
                headers=supabase_headers(),
                json=update_data,
                timeout=TIMEOUT
            )

            if not response.ok:
                raise Exception(f'Failed to update article: {response.text}')

            return jsonify({'success': True, 'action': action, 'articleId': article_id})

        except Exception as e:
            logger.error(f'Error updating article: {str(e)}')
            return jsonify({'error': 'Failed to update article'}), 500

    # Handle GET
    try:
        include_dismissed = request.args.get('includeDismissed') == 'true'
        bookmarked_only = request.args.get('bookmarked') == 'true'
        has_notes = request.args.get('hasNotes') == 'true'
        filter_status = request.args.get('status', '')
        filter_category = request.args.get('category', '')

        # Fetch articles from database
        params = {
            'select': '*',
            'order': 'pub_date.desc.nullsfirst',
            'limit': '100'
        }
        if not include_dismissed:
            params['dismissed'] = 'eq.false'
        if has_notes:
            params['notes'] = 'not.is.null'
        if filter_status:
            params['status'] = f'eq.{filter_status}'
        if filter_category:
            params['category'] = f'eq.{filter_category}'

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
            params=params,
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to fetch articles: {response.text}')

        db_articles = response.json()

        if bookmarked_only:
            db_articles = [a for a in db_articles if a.get('bookmarked')]

        # Format articles
        articles = [
            {
                'id': a.get('id'),
                'title': a.get('title'),
                'link': a.get('link'),
                'description': a.get('description'),
                'pubDate': a.get('pub_date'),
                'source': a.get('source'),
                'query': a.get('query'),
                'fetchedFrom': a.get('fetched_from'),
                'category': a.get('category'),
                'priority': a.get('priority'),
                'status': a.get('status', 'new'),
                'timestamp': a.get('pub_date') or a.get('created_at'),
                'bookmarked': a.get('bookmarked', False),
                'usedForContent': a.get('used_for_content', False),
                'dismissed': a.get('dismissed', False),
                'firstSeenAt': a.get('first_seen_at'),
                'lastSeenAt': a.get('last_seen_at'),
                'notes': a.get('notes'),
            }
            for a in db_articles
        ]

        # Sort by priority then timestamp
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        articles.sort(key=lambda x: (priority_order.get(x.get('priority', 'low'), 3), x.get('timestamp') or ''))

        # Calculate stats
        stats = {
            'total': len(articles),
            'byCategory': {
                'legislation': len([a for a in articles if a['category'] == 'legislation']),
                'enforcement': len([a for a in articles if a['category'] == 'enforcement']),
                'legal': len([a for a in articles if a['category'] == 'legal']),
                'financial': len([a for a in articles if a['category'] == 'financial']),
                'general': len([a for a in articles if a['category'] == 'general']),
            },
            'byPriority': {
                'high': len([a for a in articles if a['priority'] == 'high']),
                'medium': len([a for a in articles if a['priority'] == 'medium']),
                'low': len([a for a in articles if a['priority'] == 'low']),
            },
            'sources': len(set(a.get('source') for a in articles if a.get('source'))),
            'bookmarked': len([a for a in articles if a.get('bookmarked')]),
            'usedForContent': len([a for a in articles if a.get('usedForContent')]),
        }

        return jsonify({
            'articles': articles[:50],
            'stats': stats,
            'lastUpdated': datetime.now().isoformat(),
            'queriesUsed': HOA_QUERIES,
            'fromDatabase': True,
        })

    except Exception as e:
        logger.error(f'HOA News error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch HOA news',
            'message': str(e),
        }), 500


@dashboard_bp.route('/api/dashboard/hoa-news/analyze-notes', methods=['POST', 'OPTIONS'])
def analyze_hoa_notes():
    """Analyze all HOA news notes with AI to extract feature ideas, key points, and business analysis."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        # Fetch all articles with notes
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
            params={
                'select': 'id,title,category,priority,notes,description',
                'notes': 'not.is.null',
                'order': 'pub_date.desc',
                'limit': '50'
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to fetch noted articles: {response.text}')

        noted_articles = response.json()

        if not noted_articles:
            return jsonify({'error': 'No articles with notes found'}), 400

        # Build context for Claude
        articles_context = []
        for a in noted_articles:
            articles_context.append({
                'title': a.get('title', ''),
                'category': a.get('category', ''),
                'priority': a.get('priority', ''),
                'notes': a.get('notes', ''),
                'summary': (a.get('description') or '')[:300]
            })

        prompt = f"""You are a business analyst for DisputeMyHOA, a $29 self-service SaaS product that helps homeowners respond to HOA violations.

Below are notes taken by the team on {len(noted_articles)} HOA industry news articles. Analyze these notes and provide:

1. **Feature Ideas**: 3-5 specific product or website feature ideas inspired by these notes. Each should have a title, description, and priority (high/medium/low).

2. **Key Points**: 5-8 important themes or patterns across all the notes. These should be concise observations.

3. **Business Analysis**: A paragraph analyzing how the information in these notes can benefit DisputeMyHOA — opportunities, threats, market trends, and strategic implications.

ARTICLES WITH NOTES:
{json.dumps(articles_context, indent=2)}

Respond in this exact JSON format:
{{
  "feature_ideas": [
    {{"title": "...", "description": "...", "priority": "high|medium|low"}}
  ],
  "key_points": ["...", "..."],
  "business_analysis": "..."
}}"""

        text, usage = call_claude_sonnet(prompt, system_prompt='You are a business analyst. Return only valid JSON.')

        # Parse response
        result = json.loads(text)

        return jsonify({
            'feature_ideas': result.get('feature_ideas', []),
            'key_points': result.get('key_points', []),
            'business_analysis': result.get('business_analysis', ''),
            'articles_analyzed': len(noted_articles),
            'tokens_used': {'input': usage.input_tokens, 'output': usage.output_tokens}
        })

    except json.JSONDecodeError:
        logger.error('Failed to parse Claude response for notes analysis')
        return jsonify({'error': 'Failed to parse AI response'}), 500
    except Exception as e:
        logger.error(f'Notes analysis error: {str(e)}')
        return jsonify({'error': f'Failed to analyze notes: {str(e)}'}), 500


# ============================================================================
# BLOG GENERATOR ENDPOINTS
# ============================================================================

def generate_slug(title: str) -> str:
    """Generate URL slug from title."""
    import re
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'-+', '-', slug)
    return slug[:100]


def estimate_read_time(content: str) -> int:
    """Estimate reading time in minutes."""
    words = len(content.split())
    return max(1, math.ceil(words / 200))


def _log_claude_usage(model: str, input_tokens: int, output_tokens: int, endpoint: str = ''):
    """Log Claude API usage to Supabase for cost tracking."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    try:
        pricing = CLAUDE_PRICING.get(model, CLAUDE_PRICING['default'])
        cost = (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']
        requests.post(
            f"{SUPABASE_URL}/rest/v1/dmhoa_claude_usage",
            headers=supabase_headers(),
            json={
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': round(cost, 6),
                'endpoint': endpoint,
            },
            timeout=5
        )
    except Exception as e:
        logger.warning(f'Failed to log Claude usage: {str(e)}')


def call_claude_api(prompt: str, system_prompt: str, max_tokens: int = 4096, model: str = 'claude-sonnet-4-20250514') -> str:
    """Call Claude API."""
    response = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
        },
        json={
            'model': model,
            'max_tokens': max_tokens,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=(10, 180)
    )

    if not response.ok:
        raise Exception(f'Claude API error: {response.status_code} - {response.text}')

    data = response.json()
    usage = data.get('usage', {})
    _log_claude_usage(
        model=model,
        input_tokens=usage.get('input_tokens', 0),
        output_tokens=usage.get('output_tokens', 0),
        endpoint=model.split('-')[1] if '-' in model else model
    )
    return data['content'][0]['text']


@dashboard_bp.route('/api/dashboard/blog-generator', methods=['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'])
def handle_blog_generator():
    """Handle blog idea generation and management."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    # GET - List ideas or blogs
    if request.method == 'GET':
        try:
            item_type = request.args.get('type', 'ideas')
            status = request.args.get('status')

            if item_type == 'blogs':
                table = 'blog_posts'
                order = 'published_at.desc'
            else:
                table = 'blog_ideas'
                order = 'created_at.desc'

            params = {'select': '*', 'order': order}
            if status:
                params['status'] = f'eq.{status}'

            response = requests.get(
                f"{SUPABASE_URL}/rest/v1/{table}",
                params=params,
                headers=supabase_headers(),
                timeout=TIMEOUT
            )

            if not response.ok:
                raise Exception(f'Failed to fetch {item_type}: {response.text}')

            items = response.json()

            if item_type == 'blogs':
                return jsonify({'blogs': items, 'count': len(items)})
            return jsonify({'ideas': items, 'count': len(items)})

        except Exception as e:
            logger.error(f'Blog generator GET error: {str(e)}')
            return jsonify({'error': str(e)}), 500

    # POST - Generate ideas or blog
    if request.method == 'POST':
        try:
            data = request.get_json() or {}
            action = data.get('action', 'generate-ideas')

            if action == 'generate-ideas':
                if not ANTHROPIC_API_KEY:
                    return jsonify({'error': 'Anthropic API key not configured'}), 500

                # Fetch articles for ideas
                articles_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                    params={
                        'select': 'id,title,description,category,priority,pub_date,source,link',
                        'dismissed': 'eq.false',
                        'used_for_content': 'eq.false',
                        'order': 'priority.asc,pub_date.desc',
                        'limit': '10'
                    },
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )

                articles = articles_response.json() if articles_response.ok else []

                if len(articles) < 3:
                    return jsonify({
                        'error': 'No articles available',
                        'message': 'Not enough HOA news articles found.',
                    }), 400

                # Fetch existing blog titles and idea titles to avoid duplicates
                existing_blogs_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/blog_posts",
                    params={'select': 'title', 'order': 'published_at.desc', 'limit': '50'},
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )
                existing_blog_titles = [b['title'] for b in (existing_blogs_response.json() if existing_blogs_response.ok else [])]

                existing_ideas_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/blog_ideas",
                    params={'select': 'title', 'status': 'in.(pending,approved,generating)'},
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )
                existing_idea_titles = [i['title'] for i in (existing_ideas_response.json() if existing_ideas_response.ok else [])]

                all_existing = existing_blog_titles + existing_idea_titles

                # Generate ideas using Claude
                article_summaries = '\n'.join([
                    f'- "{a["title"]}" [{a["category"]}]: {a.get("description", "No description")}'
                    for a in articles
                ])

                existing_topics_text = '\n'.join([f'- {t}' for t in all_existing]) if all_existing else 'None yet'

                system_prompt = 'You are an expert content strategist for an HOA dispute resolution platform. Generate unique blog post ideas that help homeowners understand their rights and navigate HOA issues. Respond with valid JSON only.'

                prompt = f'''Based on these HOA news articles, generate 3-5 unique blog post ideas:

ARTICLES:
{article_summaries}

IMPORTANT - DO NOT repeat or closely resemble any of these existing topics. Each idea must cover a distinctly different angle:
{existing_topics_text}

For each idea, provide:
1. A compelling title (50-70 chars)
2. A brief description of the blog angle (2-3 sentences)
3. The unique angle/perspective
4. Target SEO keywords

Respond with this JSON:
{{"ideas": [{{"title": "Engaging blog title", "description": "What this blog will cover", "angle": "The unique perspective", "target_keywords": ["keyword1", "keyword2"], "source_article_indices": [0, 1]}}]}}'''

                response_text = call_claude_api(prompt, system_prompt, 1500)
                cleaned = response_text.replace('```json', '').replace('```', '').strip()
                result = json.loads(cleaned)

                # Save ideas
                saved_ideas = []
                for idea in result.get('ideas', []):
                    source_ids = [articles[i]['id'] for i in idea.get('source_article_indices', []) if i < len(articles)]

                    insert_response = requests.post(
                        f"{SUPABASE_URL}/rest/v1/blog_ideas",
                        headers={**supabase_headers(), 'Prefer': 'return=representation'},
                        json={
                            'title': idea['title'],
                            'description': idea.get('description'),
                            'angle': idea.get('angle'),
                            'target_keywords': idea.get('target_keywords', []),
                            'source_article_ids': source_ids,
                            'status': 'pending',
                        },
                        timeout=TIMEOUT
                    )

                    if insert_response.ok:
                        saved_ideas.extend(insert_response.json())

                return jsonify({
                    'ideas': saved_ideas,
                    'articlesAnalyzed': len(articles),
                    'message': f'Generated {len(saved_ideas)} blog ideas',
                }), 201

            elif action == 'generate-blog':
                idea_id = data.get('ideaId')
                if not idea_id:
                    return jsonify({'error': 'Missing ideaId'}), 400

                if not ANTHROPIC_API_KEY:
                    return jsonify({'error': 'Anthropic API key not configured'}), 500

                # Fetch idea
                idea_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/blog_ideas",
                    params={'id': f'eq.{idea_id}', 'select': '*'},
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )

                if not idea_response.ok or not idea_response.json():
                    return jsonify({'error': 'Idea not found'}), 404

                idea = idea_response.json()[0]

                # Update status to generating
                requests.patch(
                    f"{SUPABASE_URL}/rest/v1/blog_ideas",
                    params={'id': f'eq.{idea_id}'},
                    headers=supabase_headers(),
                    json={'status': 'generating', 'updated_at': datetime.now().isoformat()},
                    timeout=TIMEOUT
                )

                # Fetch source articles
                articles = []
                if idea.get('source_article_ids'):
                    for aid in idea['source_article_ids']:
                        art_response = requests.get(
                            f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                            params={'id': f'eq.{aid}', 'select': 'id,title,description,category'},
                            headers=supabase_headers(),
                            timeout=TIMEOUT
                        )
                        if art_response.ok and art_response.json():
                            articles.extend(art_response.json())

                # Generate blog content
                system_prompt = 'You are an expert content writer for DisputeMyHOA. Write engaging, informative, SEO-optimized blog posts. Respond with valid JSON only.'

                article_context = '\n'.join([f'- "{a["title"]}": {a.get("description", "")}' for a in articles])

                prompt = f'''Write a comprehensive blog post based on this idea:

TITLE: {idea["title"]}
ANGLE: {idea.get("angle", "")}
DESCRIPTION: {idea.get("description", "")}
TARGET KEYWORDS: {", ".join(idea.get("target_keywords", []))}

SOURCE ARTICLES:
{article_context}

Write an 800-1200 word blog post. Respond with this JSON:
{{"title": "{idea['title']}", "content": "Full blog content in markdown", "excerpt": "150-200 char summary", "seo_title": "SEO title (max 60 chars)", "seo_description": "Meta description (150-160 chars)", "seo_keywords": ["keyword1", "keyword2"], "image_search_query": "2-3 word query"}}'''

                response_text = call_claude_api(prompt, system_prompt, 4096)
                cleaned = response_text.replace('```json', '').replace('```', '').strip()
                blog_data = json.loads(cleaned)

                # Save blog post
                slug = generate_slug(blog_data['title'])
                read_time = estimate_read_time(blog_data.get('content', ''))

                insert_data = {
                    'title': blog_data['title'],
                    'slug': slug,
                    'excerpt': blog_data.get('excerpt'),
                    'content': blog_data.get('content'),
                    'category': 'hoa-news',
                    'tags': blog_data.get('seo_keywords', []),
                    'status': 'published',
                    'seo_title': blog_data.get('seo_title'),
                    'seo_description': blog_data.get('seo_description'),
                    'seo_keywords': blog_data.get('seo_keywords', []),
                    'source_article_ids': idea.get('source_article_ids', []),
                    'read_time_minutes': read_time,
                    'published_at': datetime.now().isoformat(),
                }

                blog_response = requests.post(
                    f"{SUPABASE_URL}/rest/v1/blog_posts",
                    headers={**supabase_headers(), 'Prefer': 'return=representation'},
                    json=insert_data,
                    timeout=TIMEOUT
                )

                if not blog_response.ok:
                    raise Exception(f'Failed to save blog: {blog_response.text}')

                blog = blog_response.json()[0]

                # Update idea status
                requests.patch(
                    f"{SUPABASE_URL}/rest/v1/blog_ideas",
                    params={'id': f'eq.{idea_id}'},
                    headers=supabase_headers(),
                    json={'status': 'generated', 'generated_blog_id': blog['id'], 'updated_at': datetime.now().isoformat()},
                    timeout=TIMEOUT
                )

                return jsonify({'blog': blog, 'message': 'Blog generated and published!'}), 201

            return jsonify({'error': 'Invalid action'}), 400

        except Exception as e:
            logger.error(f'Blog generator POST error: {str(e)}')
            return jsonify({'error': str(e)}), 500

    # PATCH - Update status
    if request.method == 'PATCH':
        try:
            data = request.get_json() or {}
            item_type = data.get('type')
            item_id = data.get('id')
            status = data.get('status')

            if not item_id or not status:
                return jsonify({'error': 'Missing id or status'}), 400

            if item_type == 'idea':
                if status not in ['pending', 'approved', 'rejected']:
                    return jsonify({'error': 'Invalid idea status'}), 400
                table = 'blog_ideas'
            elif item_type == 'blog':
                if status not in ['draft', 'published', 'archived']:
                    return jsonify({'error': 'Invalid blog status'}), 400
                table = 'blog_posts'
            else:
                return jsonify({'error': 'Invalid type'}), 400

            response = requests.patch(
                f"{SUPABASE_URL}/rest/v1/{table}",
                params={'id': f'eq.{item_id}'},
                headers={**supabase_headers(), 'Prefer': 'return=representation'},
                json={'status': status, 'updated_at': datetime.now().isoformat()},
                timeout=TIMEOUT
            )

            if not response.ok:
                raise Exception(f'Failed to update: {response.text}')

            result = response.json()
            if item_type == 'idea':
                return jsonify({'idea': result[0] if result else None})
            return jsonify({'blog': result[0] if result else None})

        except Exception as e:
            logger.error(f'Blog generator PATCH error: {str(e)}')
            return jsonify({'error': str(e)}), 500

    # DELETE
    if request.method == 'DELETE':
        try:
            item_type = request.args.get('type', 'idea')
            item_id = request.args.get('id')

            if not item_id:
                return jsonify({'error': 'Missing id'}), 400

            table = 'blog_posts' if item_type == 'blog' else 'blog_ideas'

            response = requests.delete(
                f"{SUPABASE_URL}/rest/v1/{table}",
                params={'id': f'eq.{item_id}'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )

            if not response.ok:
                raise Exception(f'Failed to delete: {response.text}')

            return jsonify({'success': True})

        except Exception as e:
            logger.error(f'Blog generator DELETE error: {str(e)}')
            return jsonify({'error': str(e)}), 500


# ============================================================================
# LEGALITY SCORECARD ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/legality-scorecard', methods=['GET', 'POST', 'OPTIONS'])
def handle_legality_scorecard():
    """Get or generate legality scorecard analysis."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        force_refresh = request.args.get('refresh') == 'true' or request.method == 'POST'
        include_history = request.args.get('history') == 'true'

        # Check for cached scorecard
        if not force_refresh:
            cache_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/legality_scorecard",
                params={
                    'status': 'eq.completed',
                    'order': 'analysis_date.desc',
                    'limit': '1'
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )

            if cache_response.ok:
                scorecards = cache_response.json()
                if scorecards:
                    latest = scorecards[0]
                    analysis_date = datetime.fromisoformat(latest['analysis_date'].replace('Z', '+00:00'))
                    age_hours = (datetime.now(analysis_date.tzinfo) - analysis_date).total_seconds() / 3600

                    if age_hours < 24:
                        # Parse stored analysis
                        full_analysis = latest.get('full_analysis', {})
                        if isinstance(full_analysis, str):
                            try:
                                full_analysis = json.loads(full_analysis)
                            except:
                                full_analysis = {}

                        result = {
                            'scorecard': latest,
                            'analysis': full_analysis.get('analysis'),
                            'rawData': full_analysis.get('rawData'),
                            'fromCache': True,
                            'ageHours': round(age_hours),
                        }

                        if include_history:
                            history_response = requests.get(
                                f"{SUPABASE_URL}/rest/v1/legality_scorecard",
                                params={
                                    'status': 'eq.completed',
                                    'select': 'id,analysis_date,summary,cases_analyzed',
                                    'order': 'analysis_date.desc',
                                    'limit': '10'
                                },
                                headers=supabase_headers(),
                                timeout=TIMEOUT
                            )
                            result['history'] = history_response.json() if history_response.ok else []

                        return jsonify(result)

        # Generate new scorecard
        if not ANTHROPIC_API_KEY:
            return jsonify({'error': 'Anthropic API key not configured'}), 500

        # Fetch cases
        cases_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
            params={
                'select': 'id,token,email,created_at,unlocked,status,payload,amount_total,stripe_payment_intent_id',
                'order': 'created_at.desc',
                'limit': '1000'
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        cases = cases_response.json() if cases_response.ok else []

        if not cases:
            return jsonify({
                'scorecard': None,
                'message': 'No case data available for analysis',
                'casesFound': 0,
            })

        # Analyze cases
        by_type = {}
        by_state = {}
        total_paid = 0
        total_revenue = 0

        for case in cases:
            payload = case.get('payload', {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    payload = {}

            notice_type = payload.get('noticeType') or payload.get('violationType') or 'Unknown'
            if not isinstance(notice_type, str):
                notice_type = str(notice_type) if notice_type else 'Unknown'
            state = payload.get('state') or payload.get('hoaState') or 'Unknown'
            if not isinstance(state, str):
                state = str(state) if state else 'Unknown'

            if notice_type not in by_type:
                by_type[notice_type] = {'count': 0, 'paid': 0, 'revenue': 0}
            by_type[notice_type]['count'] += 1

            if state not in by_state:
                by_state[state] = {'count': 0, 'paid': 0, 'revenue': 0}
            by_state[state]['count'] += 1

            is_paid = case.get('status') == 'paid' or case.get('unlocked')
            if is_paid:
                total_paid += 1
                by_type[notice_type]['paid'] += 1
                by_state[state]['paid'] += 1
                amount = (case.get('amount_total') or 0) / 100
                total_revenue += amount
                by_type[notice_type]['revenue'] += amount
                by_state[state]['revenue'] += amount

        # Prepare summary for Claude
        top_types = sorted(by_type.items(), key=lambda x: x[1]['count'], reverse=True)[:8]
        top_states = sorted(by_state.items(), key=lambda x: x[1]['count'], reverse=True)[:5]

        # --- Gather additional data sources ---

        # 1. Case Previews
        preview_insights = {'headlines': [], 'risks': [], 'deadlines': [], 'unlock_items': [], 'total_previews': 0}
        try:
            previews_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_case_previews",
                params={
                    'select': 'preview_content,created_at',
                    'is_active': 'eq.true',
                    'order': 'created_at.desc',
                    'limit': '200'
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if previews_resp.ok:
                previews = previews_resp.json()
                preview_insights['total_previews'] = len(previews)
                for p in previews:
                    content = p.get('preview_content', {})
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except:
                            content = {}
                    pj = content.get('preview_json', {})
                    if not pj:
                        continue
                    headline = pj.get('headline', '')
                    if headline:
                        preview_insights['headlines'].append(headline)
                    risks = pj.get('risk_if_wrong', [])
                    if isinstance(risks, list):
                        preview_insights['risks'].extend(risks[:2])
                    situation = pj.get('your_situation', {})
                    if isinstance(situation, dict):
                        deadline = situation.get('deadline', '')
                        if deadline and str(deadline).lower() not in ('not stated', 'unknown', ''):
                            preview_insights['deadlines'].append(str(deadline))
                    unlock = pj.get('what_you_get_when_you_unlock', [])
                    if isinstance(unlock, list):
                        preview_insights['unlock_items'].extend(unlock[:2])
                # Deduplicate and limit (items may be dicts, so stringify for dedup)
                def _dedup(items, limit):
                    seen = set()
                    result = []
                    for item in items:
                        key = str(item)
                        if key not in seen:
                            seen.add(key)
                            result.append(item if isinstance(item, str) else str(item))
                        if len(result) >= limit:
                            break
                    return result
                preview_insights['headlines'] = _dedup(preview_insights['headlines'], 20)
                preview_insights['risks'] = _dedup(preview_insights['risks'], 15)
                preview_insights['deadlines'] = _dedup(preview_insights['deadlines'], 10)
                preview_insights['unlock_items'] = _dedup(preview_insights['unlock_items'], 10)
        except Exception as e:
            logger.warning(f'Scorecard - preview fetch failed: {e}')

        # 2. Live Business Snapshot
        business_snapshot = {}
        try:
            business_snapshot = _build_live_data_snapshot()
        except Exception as e:
            logger.warning(f'Scorecard - live snapshot failed: {e}')

        # 3. PostHog Analytics
        posthog_data = {}
        try:
            posthog_data = _read_api_cache('posthog_data') or {}
        except Exception as e:
            logger.warning(f'Scorecard - posthog cache read failed: {e}')

        # 4. HOA News
        news_articles = []
        try:
            news_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                params={
                    'select': 'title,category,priority,pub_date',
                    'order': 'pub_date.desc',
                    'limit': '20',
                    'dismissed': 'eq.false'
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if news_resp.ok:
                news_articles = news_resp.json()
        except Exception as e:
            logger.warning(f'Scorecard - news fetch failed: {e}')

        # 5. Lighthouse
        lighthouse_data = {}
        try:
            lighthouse_data = _read_api_cache('lighthouse_data') or {}
        except Exception as e:
            logger.warning(f'Scorecard - lighthouse cache read failed: {e}')

        # --- Build enriched Claude prompt ---
        snapshot_summary = {}
        if business_snapshot:
            snapshot_summary = {
                'revenue': {
                    'today': business_snapshot.get('stripe', {}).get('revenue_today', 0),
                    'week': business_snapshot.get('stripe', {}).get('revenue_week', 0),
                    'month': business_snapshot.get('stripe', {}).get('revenue_month', 0),
                },
                'cases_today': business_snapshot.get('cases', {}).get('new_today', 0),
                'paid_today': business_snapshot.get('cases', {}).get('paid_today', 0),
                'monthly_conversion': business_snapshot.get('cases', {}).get('conversion_rate', 0),
                'ads': {
                    'impressions': business_snapshot.get('google_ads', {}).get('impressions', 0),
                    'clicks': business_snapshot.get('google_ads', {}).get('clicks', 0),
                    'cpa': business_snapshot.get('google_ads', {}).get('cpa', 0),
                    'ctr_pct': business_snapshot.get('google_ads', {}).get('ctr_pct', 0),
                },
                'email_profiles': business_snapshot.get('klaviyo', {}).get('total_profiles', 0),
            }

        posthog_summary = {}
        if posthog_data:
            posthog_summary = {
                'bounce_rate': posthog_data.get('bounceRate', 0),
                'avg_time_on_page': posthog_data.get('avgTimeOnPage', 0),
                'pages_per_session': posthog_data.get('pagesPerSession', 0),
                'rage_clicks': posthog_data.get('rageClicks', 0),
                'dead_clicks': posthog_data.get('deadClicks', 0),
                'funnel': posthog_data.get('funnel', {}),
                'web_vitals': posthog_data.get('webVitals', {}),
            }

        lighthouse_summary = {}
        if lighthouse_data:
            lighthouse_summary = {
                'performance': lighthouse_data.get('performanceScore', 0),
                'seo': lighthouse_data.get('seoScore', 0),
                'accessibility': lighthouse_data.get('accessibilityScore', 0),
            }

        news_titles = [a.get('title', '') for a in news_articles[:10]]

        system_prompt = 'You are an HOA legal tech analyst for DisputeMyHOA, a $29 self-service SaaS helping homeowners respond to HOA violation notices. Analyze real business data to produce actionable insights. Respond with valid JSON only, no markdown code fences.'

        conversion_rate = round(total_paid / len(cases) * 100, 1) if cases else 0
        violations_json = json.dumps([{"type": t, **d} for t, d in top_types])
        states_summary = ", ".join([f"{s}({d['count']} cases, ${round(d['revenue'],0)} rev)" for s, d in top_states])
        prompt = f'''Analyze this HOA dispute platform data comprehensively.

=== CASE DATA ===
VIOLATIONS (top 8 by count): {violations_json}
CONVERSION: Total={len(cases)}, Paid={total_paid}, Rate={conversion_rate}%, Revenue=${round(total_revenue, 2)}
TOP STATES: {states_summary}

=== CASE PREVIEW INSIGHTS (from {preview_insights['total_previews']} analyzed previews) ===
SAMPLE HEADLINES: {json.dumps(preview_insights['headlines'][:10])}
COMMON RISKS CUSTOMERS FACE: {json.dumps(preview_insights['risks'][:10])}
DEADLINE PATTERNS: {json.dumps(preview_insights['deadlines'][:8])}
VALUED UNLOCK ITEMS: {json.dumps(preview_insights['unlock_items'][:8])}

=== LIVE BUSINESS METRICS ===
{json.dumps(snapshot_summary, default=str) if snapshot_summary else 'Not available'}

=== USER ENGAGEMENT (PostHog) ===
{json.dumps(posthog_summary, default=str) if posthog_summary else 'Not available'}

=== SITE PERFORMANCE (Lighthouse) ===
{json.dumps(lighthouse_summary, default=str) if lighthouse_summary else 'Not available'}

=== HOA INDUSTRY NEWS (recent) ===
{json.dumps(news_titles) if news_titles else 'No recent news'}

Respond with this JSON structure. Fill ALL fields with real analysis based on the data:
{{
  "trends_summary": {{
    "most_common_violations": [{{"type":"","count":0,"percentage":0,"insight":""}}],
    "highest_converting_cases": [{{"type":"","conversion_rate":0,"avg_revenue":0,"insight":""}}],
    "seasonal_patterns": {{"peak_month":"","peak_day":"","peak_time":"","insight":""}},
    "geographic_insights": {{"top_states":[],"underserved_markets":[],"expansion_opportunities":""}},
    "preview_themes": {{"dominant_risk_patterns":[],"urgency_indicators":[],"most_valued_features":[]}}
  }},
  "conversion_analysis": {{
    "overall_rate":0,
    "best_performing_segment":"",
    "worst_performing_segment":"",
    "improvement_opportunities":[],
    "funnel_health": {{"landing_to_preview":"","preview_to_purchase":"","biggest_drop_off":"","recommended_fix":""}},
    "ad_efficiency": {{"cpa_assessment":"","ctr_assessment":"","recommendation":""}}
  }},
  "feature_suggestions": [{{"feature":"","rationale":"","priority":"high","expected_impact":""}}],
  "product_research_insights": {{
    "customer_pain_points":[],
    "unmet_needs":[],
    "content_opportunities":[],
    "partnership_opportunities":[],
    "case_theme_analysis": {{"top_themes":[],"emerging_patterns":[],"underserved_violations":[]}}
  }},
  "risk_assessment": {{
    "categories": [{{"name":"","case_count":0,"conversion_rate":0,"revenue":0,"risk_level":"medium","trend_direction":"stable","top_states":[],"strategic_notes":"","common_defenses":[]}}],
    "highest_risk_category":"",
    "fastest_growing_category":"",
    "most_profitable_category":""
  }},
  "engagement_health": {{
    "bounce_rate_assessment":"",
    "time_on_page_assessment":"",
    "ux_issues":[],
    "performance_impact":"",
    "overall_grade":""
  }},
  "news_context": {{
    "relevant_trends":[],
    "opportunities":[],
    "risks":[]
  }},
  "strategic_recommendations": [{{"recommendation":"","category":"marketing","priority":"high","expected_outcome":"","data_basis":""}}],
  "executive_summary":""
}}

Be specific and data-driven. Reference actual numbers. For recommendations, cite the specific metric in data_basis.'''

        try:
            analysis_text = call_claude_api(prompt, system_prompt, 6144)
            cleaned = analysis_text.replace('```json', '').replace('```', '').strip()
            analysis = json.loads(cleaned)
        except Exception as e:
            logger.error(f'Failed to generate analysis: {str(e)}')
            return jsonify({'error': 'Failed to generate analysis', 'details': str(e)}), 500

        raw_data = {
            'violationTypes': {t: d for t, d in top_types},
            'geography': {'topStates': [{'state': s, **d} for s, d in top_states]},
            'conversionFunnel': {
                'totalCases': len(cases),
                'paidCases': total_paid,
                'totalRevenue': round(total_revenue, 2),
                'overallConversionRate': round(total_paid / len(cases) * 100, 1) if cases else 0,
            },
            'previewInsights': {
                'totalPreviews': preview_insights.get('total_previews', 0),
                'sampleHeadlines': preview_insights.get('headlines', [])[:5],
                'commonRisks': preview_insights.get('risks', [])[:5],
                'deadlinePatterns': preview_insights.get('deadlines', [])[:5],
            },
            'businessSnapshot': {
                'revenueToday': business_snapshot.get('stripe', {}).get('revenue_today', 0),
                'revenueWeek': business_snapshot.get('stripe', {}).get('revenue_week', 0),
                'revenueMonth': business_snapshot.get('stripe', {}).get('revenue_month', 0),
                'adCPA': business_snapshot.get('google_ads', {}).get('cpa', 0),
                'adCTR': business_snapshot.get('google_ads', {}).get('ctr_pct', 0),
                'emailProfiles': business_snapshot.get('klaviyo', {}).get('total_profiles', 0),
            } if business_snapshot else {},
            'posthogMetrics': {
                'bounceRate': posthog_data.get('bounceRate', 0),
                'avgTimeOnPage': posthog_data.get('avgTimeOnPage', 0),
                'pagesPerSession': posthog_data.get('pagesPerSession', 0),
                'rageClicks': posthog_data.get('rageClicks', 0),
                'funnel': posthog_data.get('funnel', {}),
            } if posthog_data else {},
            'newsContext': {
                'articleCount': len(news_articles),
                'recentTitles': [a.get('title', '') for a in news_articles[:5]],
            },
        }

        # Save scorecard
        scorecard_data = {
            'period_start': (datetime.now() - timedelta(days=365)).isoformat(),
            'period_end': datetime.now().isoformat(),
            'categories': analysis.get('risk_assessment', {}).get('categories', []),
            'summary': {
                **analysis.get('trends_summary', {}),
                **analysis.get('conversion_analysis', {}),
                'executive_summary': analysis.get('executive_summary'),
                'feature_suggestions': analysis.get('feature_suggestions'),
                'strategic_recommendations': analysis.get('strategic_recommendations'),
            },
            'full_analysis': json.dumps({'analysis': analysis, 'rawData': raw_data}),
            'cases_analyzed': len(cases),
            'news_articles_referenced': len(news_articles),
            'status': 'completed',
        }

        save_response = requests.post(
            f"{SUPABASE_URL}/rest/v1/legality_scorecard",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json=scorecard_data,
            timeout=TIMEOUT
        )

        scorecard = save_response.json()[0] if save_response.ok and save_response.json() else scorecard_data

        return jsonify({
            'scorecard': scorecard,
            'analysis': analysis,
            'rawData': raw_data,
            'fromCache': False,
            'generated': True,
        })

    except Exception as e:
        logger.error(f'Legality scorecard error: {str(e)}', exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


# ============================================================================
# AD SUGGESTIONS ENDPOINT
# ============================================================================

# In-memory job store for ad suggestions
ad_suggestion_jobs = {}


def _build_plan_context() -> Dict:
    """Build 6-month plan context for campaign brief AI prompts.
    Plan runs May 2026 (Month 1) through October 2026 (Month 6)."""
    now = datetime.now()
    if now.strftime('%Y-%m-%d') < PLAN_START_DATE:
        # Pre-plan ramp-up — clamp to Month 1 for context purposes
        current_month = 1
    else:
        # May (5) -> Month 1, October (10) -> Month 6
        current_month = max(1, min(6, now.month - 4))

    current_plan = PLAN_MONTHS[current_month - 1]
    days_since_plan_start = (now - datetime.strptime(PLAN_START_DATE, '%Y-%m-%d')).days

    # Fetch live metrics for grading
    stripe_month = _fetch_stripe_metrics('month')
    ads = _fetch_google_ads_metrics('plan_start') or {}
    posthog = _fetch_posthog_metrics_for_plan()

    monthly_revenue = stripe_month.get('revenue', 0)
    ads_spend = ads.get('spend', 0)
    ads_ctr = ads.get('ctr', 0)
    ads_conversions = ads.get('conversions', 0)
    monthly_visitors = posthog.get('unique_visitors', 0)

    roas = (monthly_revenue / ads_spend) if ads_spend > 0 else 0
    cac = (ads_spend / ads_conversions) if ads_conversions > 0 else 0

    traffic_grade = _grade(monthly_visitors, GRADING['traffic']['monthly_visitors'])
    ctr_grade = _grade(ads_ctr, GRADING['traffic']['google_ads_ctr'])
    revenue_grade = _grade(monthly_revenue, GRADING['revenue']['monthly_revenue'])
    roas_grade = _grade(roas, GRADING['revenue']['roas'])

    return {
        'plan_start_date': PLAN_START_DATE,
        'days_since_start': days_since_plan_start,
        'current_month_number': current_month,
        'current_month_name': current_plan['name'],
        'current_theme': current_plan['theme'],
        'budget_planned': current_plan['budget_planned'],
        'budget_actual': round(ads_spend, 2),
        'monthly_revenue': round(monthly_revenue, 2),
        'roas': round(roas, 2),
        'monthly_visitors': monthly_visitors,
        'ads_ctr_pct': round(ads_ctr * 100, 2) if ads_ctr < 1 else round(ads_ctr, 2),
        'ads_conversions': ads_conversions,
        'cac': round(cac, 2),
        'grades': {
            'traffic': traffic_grade,
            'ctr': ctr_grade,
            'revenue': revenue_grade,
            'roas': roas_grade,
        },
        'scenario': 'good' if traffic_grade != 'F' and revenue_grade != 'F' else ('ugly' if traffic_grade == 'F' and revenue_grade == 'F' else 'bad'),
        'all_months': [
            {'month': pm['month'], 'name': pm['name'], 'theme': pm['theme'], 'budget': pm['budget_planned']}
            for pm in PLAN_MONTHS
        ],
    }


# ─── HOA Statutes Management ──────────────────────────────────────────────────

ALL_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]
ALL_STATUTE_CATEGORIES = sorted(VALID_CATEGORIES)


@dashboard_bp.route('/api/dashboard/statutes', methods=['GET', 'OPTIONS'])
def get_statutes():
    """Get all statutes with coverage summary."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        headers = supabase_headers()
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/hoa_statutes",
            params={'select': '*', 'order': 'state.asc,violation_category.asc', 'limit': '1000'},
            headers=headers, timeout=TIMEOUT
        )
        resp.raise_for_status()
        statutes = resp.json()

        # Build coverage summary
        by_state = {}
        by_category = {}
        for s in statutes:
            st = s.get('state', '')
            cat = s.get('violation_category', '')
            by_state[st] = by_state.get(st, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1

        total_possible = len(ALL_STATES) * len(ALL_STATUTE_CATEGORIES)

        return jsonify({
            'statutes': statutes,
            'coverage': {
                'total_possible': total_possible,
                'total_existing': len(statutes),
                'by_state': by_state,
                'by_category': by_category,
            },
            'categories': ALL_STATUTE_CATEGORIES,
            'states': ALL_STATES,
        })

    except Exception as e:
        logger.error(f"Error fetching statutes: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/api/dashboard/statutes/scan', methods=['POST', 'OPTIONS'])
def scan_statutes():
    """Bulk-generate missing statutes using Claude Haiku."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    body = request.get_json(silent=True) or {}
    req_states = body.get('states', ['ALL'])
    req_categories = body.get('categories', ['ALL'])

    # Resolve targets
    target_states = ALL_STATES if 'ALL' in req_states else [s.upper() for s in req_states if s.upper() in ALL_STATES]
    target_categories = ALL_STATUTE_CATEGORIES if 'ALL' in req_categories else [c for c in req_categories if c in VALID_CATEGORIES]

    if not target_states or not target_categories:
        return jsonify({'error': 'No valid states or categories specified'}), 400

    # Fetch existing statutes to find gaps
    try:
        headers = supabase_headers()
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/hoa_statutes",
            params={'select': 'state,violation_category', 'limit': '1000'},
            headers=headers, timeout=TIMEOUT
        )
        resp.raise_for_status()
        existing = {(r['state'], r['violation_category']) for r in resp.json()}
    except Exception as e:
        logger.error(f"Error fetching existing statutes: {e}")
        return jsonify({'error': f'Failed to check existing statutes: {e}'}), 500

    # Find missing pairs
    missing = []
    for state in target_states:
        for cat in target_categories:
            if (state, cat) not in existing:
                missing.append((state, cat))

    # Cap at 3 per request to stay within Heroku 30s timeout (~8s each)
    batch = missing[:3]
    generated = 0
    failed = 0
    details = []

    for state, cat in batch:
        try:
            result = generate_statute_with_claude(state, cat)
            if result:
                saved = save_statute_to_db(state, cat, result)
                if saved:
                    generated += 1
                    details.append({'state': state, 'category': cat, 'status': 'generated'})
                else:
                    failed += 1
                    details.append({'state': state, 'category': cat, 'status': 'save_failed'})
            else:
                failed += 1
                details.append({'state': state, 'category': cat, 'status': 'generation_failed'})
        except Exception as e:
            failed += 1
            details.append({'state': state, 'category': cat, 'status': f'error: {str(e)}'})

    return jsonify({
        'generated': generated,
        'failed': failed,
        'skipped': len(missing) - len(batch),
        'total_missing': len(missing),
        'details': details,
    })


@dashboard_bp.route('/api/dashboard/statutes/<state>/<category>', methods=['DELETE', 'OPTIONS'])
def delete_statute(state, category):
    """Delete a single statute for regeneration."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        headers = supabase_headers()
        resp = requests.delete(
            f"{SUPABASE_URL}/rest/v1/hoa_statutes",
            params={'state': f'eq.{state.upper()}', 'violation_category': f'eq.{category}'},
            headers=headers, timeout=TIMEOUT
        )
        resp.raise_for_status()
        return jsonify({'deleted': True, 'state': state.upper(), 'category': category})
    except Exception as e:
        logger.error(f"Error deleting statute {state}/{category}: {e}")
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/api/dashboard/ad-suggestions', methods=['GET', 'POST', 'OPTIONS'])
def handle_ad_suggestions():
    """Handle AI-powered Google Ads optimization suggestions."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    # GET - Check job status
    if request.method == 'GET':
        job_id = request.args.get('jobId')

        if not job_id:
            return jsonify({'status': 'error', 'error': 'jobId parameter required'}), 400

        # Check in-memory store
        if job_id in ad_suggestion_jobs:
            job = ad_suggestion_jobs[job_id]
            if job['status'] == 'complete':
                return jsonify({'status': 'complete', 'result': job['result']})
            if job['status'] == 'error':
                return jsonify({'status': 'error', 'error': job.get('error')})
            return jsonify({'status': 'processing', 'jobId': job_id})

        # Check Supabase
        if SUPABASE_URL:
            response = requests.get(
                f"{SUPABASE_URL}/rest/v1/ad_suggestion_jobs",
                params={'job_id': f'eq.{job_id}', 'select': '*'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )

            if response.ok and response.json():
                job = response.json()[0]
                return jsonify({
                    'status': job['status'],
                    'result': job.get('result'),
                    'error': job.get('error'),
                })

        return jsonify({'status': 'processing', 'jobId': job_id, 'message': 'Job still initializing...'})

    # POST - Start new job
    if not ANTHROPIC_API_KEY:
        return jsonify({
            'status': 'error',
            'error': 'AI suggestions not configured. Add ANTHROPIC_API_KEY.',
        })

    try:
        data = request.get_json() or {}
        start_date = data.get('startDate')
        end_date = data.get('endDate')
        customer_id = data.get('customerId')
        campaign_name = data.get('campaignName')
        campaign_id = data.get('campaignId')
        include_plan_context = data.get('includePlanContext', False)

        # Campaign brief mode: default to full plan range if no dates specified
        if campaign_name or campaign_id:
            if not start_date:
                start_date = PLAN_START_DATE
            if not end_date:
                now_la = datetime.now(ZoneInfo('America/Los_Angeles'))
                end_date = now_la.strftime('%Y-%m-%d')
            include_plan_context = True
        elif not start_date or not end_date:
            return jsonify({'status': 'error', 'error': 'startDate and endDate are required'}), 400

        # Generate job ID
        import uuid
        job_id = f"job_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:9]}"

        # Save initial job status
        ad_suggestion_jobs[job_id] = {'status': 'processing'}

        if SUPABASE_URL:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/ad_suggestion_jobs",
                headers={**supabase_headers(), 'Prefer': 'resolution=merge-duplicates'},
                json={
                    'job_id': job_id,
                    'status': 'processing',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                },
                timeout=TIMEOUT
            )

        # Start background processing (simplified - runs synchronously for now)
        # In production, this should use a task queue like Celery
        import threading

        def process_analysis():
            try:
                # Build campaign filter clause for GAQL
                campaign_filter = ''
                if campaign_id:
                    campaign_filter = f" AND campaign.id = '{campaign_id}'"
                elif campaign_name:
                    safe_name = campaign_name.replace("'", "\\'")
                    campaign_filter = f" AND campaign.name = '{safe_name}'"

                # Fetch real Google Ads data to feed into the prompt
                ads_data = {}
                try:
                    access_token = get_google_ads_access_token()
                    if access_token:
                        date_range_ads = {'startDate': start_date, 'endDate': end_date}

                        # Campaign performance
                        camp_query = f"""
                            SELECT campaign.name, campaign.status,
                                metrics.impressions, metrics.clicks, metrics.cost_micros,
                                metrics.conversions, metrics.ctr, metrics.average_cpc
                            FROM campaign
                            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                                AND campaign.status != 'REMOVED' AND metrics.cost_micros > 0
                                {campaign_filter}
                            ORDER BY metrics.cost_micros DESC
                        """
                        camp_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, camp_query)
                        ads_data['campaigns'] = []
                        for row in camp_results:
                            c = row.get('campaign', {})
                            m = row.get('metrics', {})
                            ads_data['campaigns'].append({
                                'name': c.get('name'), 'status': c.get('status'),
                                'clicks': m.get('clicks', 0), 'impressions': m.get('impressions', 0),
                                'spend': round(int(m.get('costMicros', 0) or 0) / 1_000_000, 2),
                                'conversions': m.get('conversions', 0),
                            })

                        # Keyword performance
                        kw_query = f"""
                            SELECT campaign.name, ad_group.name,
                                ad_group_criterion.keyword.text, ad_group_criterion.keyword.match_type,
                                ad_group_criterion.quality_info.quality_score,
                                metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions
                            FROM keyword_view
                            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                                AND campaign.status = 'ENABLED' AND metrics.impressions > 0
                                {campaign_filter}
                            ORDER BY metrics.clicks DESC LIMIT 30
                        """
                        kw_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, kw_query)
                        ads_data['keywords'] = []
                        for row in kw_results:
                            crit = row.get('adGroupCriterion', {})
                            kw = crit.get('keyword', {})
                            m = row.get('metrics', {})
                            qs = crit.get('qualityInfo', {}).get('qualityScore')
                            ads_data['keywords'].append({
                                'keyword': kw.get('text', ''), 'matchType': kw.get('matchType', ''),
                                'qualityScore': int(qs) if qs is not None else None,
                                'clicks': m.get('clicks', 0), 'impressions': m.get('impressions', 0),
                                'spend': round(int(m.get('costMicros', 0) or 0) / 1_000_000, 2),
                                'conversions': m.get('conversions', 0),
                            })

                        # Search terms
                        st_query = f"""
                            SELECT campaign.name, search_term_view.search_term,
                                metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions
                            FROM search_term_view
                            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                                AND campaign.status = 'ENABLED' AND metrics.impressions > 0
                                {campaign_filter}
                            ORDER BY metrics.clicks DESC LIMIT 30
                        """
                        st_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, st_query)
                        ads_data['searchTerms'] = []
                        for row in st_results:
                            stv = row.get('searchTermView', {})
                            m = row.get('metrics', {})
                            ads_data['searchTerms'].append({
                                'searchTerm': stv.get('searchTerm', ''),
                                'clicks': m.get('clicks', 0), 'impressions': m.get('impressions', 0),
                                'spend': round(int(m.get('costMicros', 0) or 0) / 1_000_000, 2),
                                'conversions': m.get('conversions', 0),
                            })

                        # Ad copy performance (responsive search ads)
                        ad_query = f"""
                            SELECT campaign.name, ad_group.name,
                                ad_group_ad.ad.id,
                                ad_group_ad.ad.responsive_search_ad.headlines,
                                ad_group_ad.ad.responsive_search_ad.descriptions,
                                ad_group_ad.ad.final_urls,
                                ad_group_ad.status,
                                metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions
                            FROM ad_group_ad
                            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                                AND campaign.status = 'ENABLED'
                                AND ad_group_ad.status != 'REMOVED'
                                AND ad_group_ad.ad.type = 'RESPONSIVE_SEARCH_AD'
                                AND metrics.impressions > 0
                                {campaign_filter}
                            ORDER BY metrics.clicks DESC LIMIT 20
                        """
                        ad_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, ad_query)
                        ads_data['adCopy'] = []
                        for row in ad_results:
                            aga = row.get('adGroupAd', {})
                            ad = aga.get('ad', {})
                            rsa = ad.get('responsiveSearchAd', {})
                            m = row.get('metrics', {})
                            ad_clicks = int(m.get('clicks', 0) or 0)
                            ad_impr = int(m.get('impressions', 0) or 0)
                            ad_spend = int(m.get('costMicros', 0) or 0) / 1_000_000
                            headlines = [h.get('text', '') for h in (rsa.get('headlines') or [])]
                            descriptions = [d.get('text', '') for d in (rsa.get('descriptions') or [])]
                            ads_data['adCopy'].append({
                                'campaignName': row.get('campaign', {}).get('name', ''),
                                'adGroupName': row.get('adGroup', {}).get('name', ''),
                                'status': aga.get('status', ''),
                                'headlines': headlines,
                                'descriptions': descriptions,
                                'finalUrl': (ad.get('finalUrls') or [''])[0],
                                'clicks': ad_clicks, 'impressions': ad_impr,
                                'spend': round(ad_spend, 2),
                                'ctr': round((ad_clicks / ad_impr) * 100, 2) if ad_impr > 0 else 0,
                            })

                        # Active negative keywords
                        neg_query = f"""
                            SELECT campaign.name, ad_group.name,
                                ad_group_criterion.keyword.text,
                                ad_group_criterion.keyword.match_type,
                                ad_group_criterion.negative
                            FROM ad_group_criterion
                            WHERE ad_group_criterion.negative = TRUE
                                AND ad_group_criterion.status = 'ENABLED'
                                AND campaign.status = 'ENABLED'
                                AND ad_group_criterion.type = 'KEYWORD'
                                {campaign_filter}
                            LIMIT 100
                        """
                        neg_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, neg_query)
                        ads_data['activeNegativeKeywords'] = []
                        for row in neg_results:
                            crit = row.get('adGroupCriterion', {})
                            kw = crit.get('keyword', {})
                            ads_data['activeNegativeKeywords'].append({
                                'keyword': kw.get('text', ''),
                                'matchType': kw.get('matchType', ''),
                                'campaignName': row.get('campaign', {}).get('name', ''),
                                'adGroupName': row.get('adGroup', {}).get('name', ''),
                            })

                        # Campaign-level negative keywords
                        camp_neg_query = f"""
                            SELECT campaign.name,
                                campaign_criterion.keyword.text,
                                campaign_criterion.keyword.match_type
                            FROM campaign_criterion
                            WHERE campaign_criterion.negative = TRUE
                                AND campaign.status = 'ENABLED'
                                AND campaign_criterion.type = 'KEYWORD'
                                {campaign_filter}
                            LIMIT 100
                        """
                        try:
                            camp_neg_results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, camp_neg_query)
                            for row in camp_neg_results:
                                crit = row.get('campaignCriterion', {})
                                kw = crit.get('keyword', {})
                                ads_data['activeNegativeKeywords'].append({
                                    'keyword': kw.get('text', ''),
                                    'matchType': kw.get('matchType', ''),
                                    'campaignName': row.get('campaign', {}).get('name', ''),
                                    'adGroupName': '(campaign-level)',
                                })
                        except Exception:
                            pass  # campaign_criterion may not be available

                except Exception as e:
                    logger.warning(f'Failed to fetch ads data for suggestions: {str(e)}')

                # Compute period insights from real data (not AI-generated)
                total_spend = sum(float(c.get('spend', 0) or 0) for c in ads_data.get('campaigns', []))
                total_clicks = sum(int(c.get('clicks', 0) or 0) for c in ads_data.get('campaigns', []))
                total_impressions = sum(int(c.get('impressions', 0) or 0) for c in ads_data.get('campaigns', []))
                total_conversions = sum(float(c.get('conversions', 0) or 0) for c in ads_data.get('campaigns', []))
                days_analyzed = max(1, (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1)

                # Fetch Stripe revenue for the analysis period
                total_revenue = 0
                paid_conversions = 0
                try:
                    if STRIPE_SECRET_KEY:
                        import stripe
                        stripe.api_key = STRIPE_SECRET_KEY
                        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
                        end_ts = int((datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).timestamp())
                        charges = stripe.Charge.list(created={'gte': start_ts, 'lte': end_ts}, limit=100)
                        successful = [c for c in charges.data if c.status == 'succeeded' and not c.refunded]
                        total_revenue = round(sum(c.amount for c in successful) / 100, 2)
                        paid_conversions = len(successful)
                except Exception as e:
                    logger.warning(f'Failed to fetch Stripe revenue for period insights: {e}')

                net_position = round(total_revenue - total_spend, 2)
                roas_ratio = round(total_revenue / total_spend, 2) if total_spend > 0 else 0
                cost_per_paid = round(total_spend / paid_conversions, 2) if paid_conversions > 0 else 0
                cost_per_case = round(total_spend / total_conversions, 2) if total_conversions > 0 else 0

                period_insights = {
                    'dateRange': f'{start_date} to {end_date}',
                    'daysAnalyzed': days_analyzed,
                    'totalSpend': round(total_spend, 2),
                    'totalRevenue': total_revenue,
                    'netPosition': net_position,
                    'costPerCaseStart': cost_per_case,
                    'costPerPaidConversion': cost_per_paid,
                    'revenueToSpendRatio': roas_ratio,
                    'dataQualityNote': '' if total_revenue > 0 else 'Revenue data not available for this period. Stripe charges may not directly correlate to Google Ads clicks within this date range.'
                }

                # Fetch real email funnel metrics (from email_funnel table, not Klaviyo)
                klaviyo_context_str = ''
                try:
                    funnel_resp = requests.get(
                        f"{SUPABASE_URL}/rest/v1/email_funnel",
                        headers=supabase_headers(),
                        params={'select': '*', 'limit': '500'},
                        timeout=TIMEOUT,
                    )
                    funnel_rows = funnel_resp.json() if funnel_resp.ok else []

                    total_funnel = len(funnel_rows)
                    quick_count = sum(1 for r in funnel_rows if r.get('stage') == 'quick_preview_complete')
                    full_count = sum(1 for r in funnel_rows if r.get('stage') == 'full_preview_viewed')
                    purchased_count = sum(1 for r in funnel_rows if r.get('purchased'))
                    nudges_sent = sum(1 for r in funnel_rows if r.get('nudge_1_sent')) + sum(1 for r in funnel_rows if r.get('nudge_2_sent')) + sum(1 for r in funnel_rows if r.get('nudge_3_sent'))
                    capture_rate = (total_funnel / total_clicks * 100) if total_clicks > 0 else 0

                    klaviyo_context_str = f"""
=== EMAIL FUNNEL METRICS (Resend) ===
Analysis Period: {start_date} to {end_date}
Total Emails Captured: {total_funnel}
Funnel Breakdown:
  Quick Preview Stage: {quick_count}
  Full Preview Stage: {full_count}
  Purchased: {purchased_count}
Nudge Emails Sent: {nudges_sent}
Email Capture Rate: {capture_rate:.1f}% (captures / ad clicks)
Revenue from Funnel: ${purchased_count * 29}
IMPORTANT: These are ACTUAL numbers from the email_funnel database table. Do NOT inflate, estimate, or round up these numbers. Report them exactly as shown.
========================"""
                except Exception as e:
                    logger.warning(f'Failed to fetch email funnel metrics for ad suggestions: {e}')
                    klaviyo_context_str = """
=== EMAIL FUNNEL METRICS ===
Data unavailable for this period
========================"""

                # Build plan context if requested
                plan_context_str = ''
                plan_context_data = {}
                if include_plan_context:
                    try:
                        plan_context_data = _build_plan_context()
                        plan_context_str = f"""
=== 6-MONTH MARKETING PLAN CONTEXT ===
Plan Start: {plan_context_data['plan_start_date']} ({plan_context_data['days_since_start']} days ago)
Current Month: {plan_context_data['current_month_number']} - {plan_context_data['current_month_name']}
Current Theme: {plan_context_data['current_theme']}
Budget Planned This Month: ${plan_context_data['budget_planned']}
Budget Spent (All Campaigns): ${plan_context_data['budget_actual']}
Monthly Revenue: ${plan_context_data['monthly_revenue']}
ROAS: {plan_context_data['roas']}x
Monthly Visitors: {plan_context_data['monthly_visitors']}
Ads CTR: {plan_context_data['ads_ctr_pct']}%
Customer Acquisition Cost: ${plan_context_data['cac']}
Grades: Traffic={plan_context_data['grades']['traffic']}, CTR={plan_context_data['grades']['ctr']}, Revenue={plan_context_data['grades']['revenue']}, ROAS={plan_context_data['grades']['roas']}
Overall Scenario: {plan_context_data['scenario']}

Full 6-Month Plan:
{json.dumps(plan_context_data['all_months'], indent=2)}
"""
                    except Exception as e:
                        logger.warning(f'Failed to build plan context: {str(e)}')

                # Build system prompt
                campaign_label = campaign_name or (f'Campaign ID {campaign_id}' if campaign_id else '')
                if campaign_label:
                    system_prompt = f'''You are a Google Ads optimization expert for DisputeMyHOA, a $29 self-service SaaS tool that helps homeowners respond to HOA violation notices.
You are generating a CAMPAIGN BRIEF for: "{campaign_label}".
This brief should provide a full picture of how this specific campaign is performing, grounded in the 6-month marketing plan context.
Evaluate the campaign's trajectory — is it improving, declining, or stable? Is it on track for the plan goals?
Respond with ONLY valid JSON, no markdown, no explanation. Every array in the response MUST have at least 2-3 items.'''
                else:
                    system_prompt = '''You are a Google Ads optimization expert for DisputeMyHOA, a $29 self-service SaaS tool that helps homeowners respond to HOA violation notices.
The product helps DIY homeowners write responses to HOA fines and violations — NOT for people seeking attorneys.
Respond with ONLY valid JSON, no markdown, no explanation. Every array in the response MUST have at least 2-3 items.'''

                # System prompt v2.2 — 2026-03-01
                # Changes:
                # 1. Negative keyword logic: require spend>0 AND clicks>0
                # 2. Keyword dedup: mandatory cross-reference vs keyword_view
                # 3. Duplicate match type detection: flag phrase+exact conflicts
                # 4. Email capture: surface Klaviyo data as funnel signal
                # 5. Ad copy constraints: char limits + no legal/prevention language
                # 6. Ad scheduling: flag overnight impression waste >10%
                # 7. Citation enforcement: verbatim dataSource required
                # 8. Budget phase awareness: no increase in first 14 days
                # 9. Zero-spend keyword rule: never flag $0 spend keywords

                # Build campaign brief JSON field if in campaign mode
                campaign_brief_field = ''
                if campaign_label:
                    campaign_brief_field = f'"campaignBrief": {{ "campaignName": "{campaign_label}", "trajectory": "improving|declining|stable", "planAlignment": "<how this campaign aligns with current month theme and goals>", "budgetAssessment": "<is spending on track vs plan?>", "gradeImpact": "<how this campaign affects the overall grades>", "strategicRecommendation": "<1-2 sentence high-level recommendation based on plan phase>" }},'

                prompt = f'''Analyze this Google Ads performance data for {start_date} to {end_date}{f' for campaign: "{campaign_label}"' if campaign_label else ''}.
{plan_context_str}
=== CAMPAIGN PERFORMANCE ===
{json.dumps(ads_data.get('campaigns', []), indent=2)}

=== KEYWORD PERFORMANCE (active campaigns only) ===
{json.dumps(ads_data.get('keywords', []), indent=2)}

=== SEARCH TERMS (what users actually searched) ===
{json.dumps(ads_data.get('searchTerms', []), indent=2)}

=== CURRENT AD COPY (headlines & descriptions in use) ===
{json.dumps(ads_data.get('adCopy', []), indent=2)}

=== ACTIVE NEGATIVE KEYWORDS (currently excluded) ===
{json.dumps(ads_data.get('activeNegativeKeywords', []), indent=2)}

{klaviyo_context_str}

Based on ALL the data above, respond with ONLY a JSON object (no markdown code fences). EVERY array MUST contain at least 2-3 items:

{{
  "performanceSummary": "<brief 2-3 sentence analysis of overall performance>",
  {campaign_brief_field}
  "keywordSuggestions": [{{ "keyword": "<text>", "matchType": "PHRASE|EXACT|BROAD", "action": "add|pause|modify", "rationale": "<why this keyword should be added/paused/modified>", "priority": "high|medium|low" }}],
  "negativeKeywordSuggestions": [{{ "keyword": "<search term to exclude>", "rationale": "<why this wastes budget — reference search terms data and active negatives>", "priority": "high|medium|low" }}],
  "adCopySuggestions": [{{ "type": "headline|description", "current": "<current text being replaced, or null if new>", "suggested": "<suggested new text — headlines max 30 chars, descriptions max 90 chars>", "rationale": "<what current ad copy is missing or how this improves CTR>", "priority": "high|medium|low" }}],
  "generalRecommendations": [{{ "recommendation": "<actionable recommendation>", "category": "budget|targeting|bidding|creative|landing_page", "priority": "high|medium|low", "expectedImpact": "<expected outcome if implemented>" }}]
}}

===================================================================
ANALYSIS RULES — follow every rule below precisely.
===================================================================

BUSINESS CONTEXT — READ FIRST:
DisputeMyHOA is a $29 self-serve SaaS tool that generates HOA violation response letters. It is NOT a law firm, NOT legal counsel, NOT a referral service.

Target customer: a homeowner who already has an HOA violation, fine, notice, or collections threat in hand and wants to respond themselves without hiring an attorney.

The funnel has two steps:
  Step 1 — Free preview: user uploads notice, gets plain-English explanation, risk assessment, and response options
  Step 2 — $29 payment: unlocks complete response letters, compliance checklist, statute citations, deadline reminders

Email capture via abandonment flows is the intermediate conversion event between ad click and payment. It is meaningful funnel progress and must be treated as a positive signal, not ignored.

---

WRONG-INTENT WORDS — never recommend targeting these:
lawyer, lawyers, attorney, attorneys, near me, pro bono, free consultation, legal advice, lawsuit, sue, court, mediators, template, templates, free, sample, investigate, investigated, prevention, prevent, stop violations

RIGHT-INTENT SIGNALS — these indicate a buyer:
respond, response, letter, dispute, fight, violation, notice, write, help, fine, appeal, collections, challenge

---

RULE 1 — NEGATIVE KEYWORD LOGIC:
Only recommend EXCLUDE for a search term if ALL of these are true simultaneously:
  1. spend > $0
  2. clicks > 0
  3. conversions = 0
  4. Term contains at least one wrong-intent word from the list above

NEVER recommend excluding:
  - Any term where clicks = 0 AND cost = $0
  - Broad informational words in isolation ("rules", "guidelines", "laws", "regulations") unless paired with wrong-intent words AND have real spend
  - Terms you haven't seen in the provided search_term_view data — do not invent candidates

If no terms meet all 4 criteria, return empty array [].

---

RULE 2 — KEYWORD DEDUPLICATION — EXECUTE BEFORE EVERY SUGGESTION:

Step 1: Copy this exact list from keyword_view into your working memory before generating any suggestions:
[every keyword text value from keyword_view, lowercased, trimmed, with match type noted]

Step 2: For each keyword you are considering suggesting with action=ADD, do this check explicitly:
  - Take the candidate keyword text, lowercase it, trim it
  - Search the list from Step 1 for an exact text match
  - If found in ANY match type: DO NOT include this keyword in keywordSuggestions. Remove it. Full stop.
  - If not found in any match type: include it normally

Step 3: Before writing your final JSON output, read back each keywordSuggestion you are about to output and verify its text does not appear in the keyword_view list.

If after deduplication no valid ADD candidates remain, return an empty keywordSuggestions array [].

An empty array is correct and acceptable output. Populating keywordSuggestions with keywords already in the account is the worst possible output — it is misinformation that wastes the operator's time.

---

RULE 3 — DUPLICATE MATCH TYPE DETECTION:
Before generating any other recommendations, scan keyword_view for keywords appearing in more than one match type simultaneously where BOTH versions have spend > $0.

For each duplicate found, add to generalRecommendations:
{{
  "recommendation": "DUPLICATE KEYWORD — '[keyword]' is running as both [MATCH TYPE A] ($X.XX spent, Y clicks) and [MATCH TYPE B] ($X.XX spent, Y clicks) simultaneously. They compete against each other in the same auctions, splitting bid signal and inflating your own CPC. Pause the PHRASE match. Keep EXACT match for bid control.",
  "category": "bidding",
  "priority": "high",
  "expectedImpact": "Consolidates bid signal, eliminates internal auction competition, reduces wasted spend",
  "dataSource": "keyword_view: [keyword] PHRASE spend=$X.XX clicks=Y | [keyword] EXACT spend=$X.XX clicks=Y"
}}

If no duplicates exist with spend > $0 on both, skip silently.

---

RULE 4 — ZERO SPEND KEYWORD RULE:
Never recommend PAUSE, MODIFY, or any action for a keyword where clicks = 0 AND spend = $0, regardless of impressions.

Keywords with zero spend have zero impact on the account. Impressions alone do not justify action in a CPC campaign. Only flag keywords with actual spend > $0 and no conversion.

---

RULE 5 — EMAIL CAPTURE AS FUNNEL SIGNAL:
The KLAVIYO METRICS section contains email captures from abandonment flows during the analysis period.

If total email captures > 0:
  Calculate email_capture_rate = total_email_captures / total_clicks (if total_clicks > 0)

  Always include this in performanceSummary regardless of payment conversions:
  "Email capture rate of [rate]% — [N] homeowner(s) entered the abandonment flow, confirming the free preview is compelling enough to generate leads. Abandonment flow is active and working."

  In gradeImpact within campaignBrief, list email capture rate as a positive metric alongside CTR.

  NEVER grade a period as purely negative when email captures exist. An email capture is a real person who saw value in the free preview. It is the most important leading indicator of future conversion.

  A period with 1+ email captures and 0 payments is "early funnel working, payment conversion pending" — not "complete failure."

If KLAVIYO METRICS shows data unavailable:
  Note in performanceSummary: "Klaviyo capture data unavailable for this period — email funnel metrics excluded from analysis."

---

RULE 6 — AD COPY CONSTRAINTS:
Users ALREADY HAVE a violation. Never imply prevention.

FORBIDDEN in any headline or description:
  - Violation prevention: "Stop HOA Violations", "Prevent fines", "Avoid violations"
  - Legal representation: "legal help", "legal service", "attorneys", "lawyers", "legal advice"
  - Attorney replacement: "no attorney needed", "without a lawyer", "handle without an attorney", "without an attorney" in any form

ACCEPTABLE framing:
  - "Handle it yourself" / "Fight it yourself"
  - "Skip the attorney fees" (cost comparison, not legal claim)
  - "Respond to your HOA violation"
  - "Free preview first" / "See before you pay"
  - "$29 flat fee" (price clarity)

CHARACTER LIMITS — these are hard limits, not guidelines:
  Headlines: 30 characters maximum (count spaces + punctuation)
  Descriptions: 90 characters maximum
  Callouts: 25 characters maximum

Count every character in every suggestion before including it. If it exceeds the limit, rewrite until it fits. Never output a suggestion that exceeds these limits.

Also check existing ad copy in the RSA data for typos or broken text. Flag any headline or description where the text appears cut off, grammatically broken, or contains obvious errors.

Use type "headline" or "description". Set current to the text being replaced (or null if brand new). Set suggested to the new text.

---

RULE 7 — AD SCHEDULING AUDIT:
When hour_of_day data is present, calculate:

overnight_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
overnight_impressions = sum of impressions in those hours
overnight_pct = overnight_impressions / total_impressions

If overnight_pct > 0.10:
  Add to generalRecommendations:
  {{
    "recommendation": "Set ad schedule to 7AM-9PM only. [pct]% of impressions ([overnight_impressions] of [total]) occur overnight (10PM-6AM) when homeowners making $29 decisions are asleep. These accumulate low-engagement signals that reduce Quality Score without driving conversions. Peak hours are [top 3 hours by impression volume] — concentrate budget there.",
    "category": "targeting",
    "priority": "high",
    "expectedImpact": "Redirect [pct]% of overnight budget to peak hours, improve Quality Score",
    "dataSource": "hour_of_day: [overnight_impressions] overnight impressions of [total] total"
  }}

---

RULE 8 — BUDGET PHASE AWARENESS:
The campaign uses a deliberate phased budget approach:
  Days 1-14:  $5/day — traffic quality validation phase
  Days 15-28: $10/day — scale only if traffic is clean
  Reserve:    held back for negative keyword adjustments

To determine campaign age, use the earliest date in the data where spend > $0 vs the analysis end date.

If campaign_age_days < 14:
  DO NOT recommend increasing daily budget. Underspend in the first 14 days is intentional strategy.

  If a budget recommendation would otherwise be triggered, replace it with:
  {{
    "recommendation": "Campaign is in validation phase (day [campaign_age_days] of 14). Hold budget at $5/day. Budget increase trigger: day 14 reached AND search terms show <20% wrong-intent clicks AND at least 1 confirmed case start.",
    "category": "budget",
    "priority": "low",
    "expectedImpact": "Prevents premature scaling before traffic quality is confirmed",
    "dataSource": "campaign time_series: first spend date [first_date], analysis end [end_date], age [N] days"
  }}

If campaign_age_days >= 14 AND traffic is clean:
  Then and only then evaluate whether budget increase is warranted based on conversion data.

---

RULE 9 — CITATION ENFORCEMENT (non-negotiable):
Every item in negativeKeywordSuggestions, keywordSuggestions, adCopySuggestions, and generalRecommendations MUST have a dataSource field containing specific verbatim values from the provided data.

Required format examples:
  "search_term_view: term='hoa lawyers near me', clicks=3, spend=$8.15, conversions=0"
  "keyword_view: 'fighting hoa violations' PHRASE spend=$8.15 clicks=3 | EXACT spend=$2.29 clicks=1"
  "hour_of_day: overnight impressions 21 of 138 total (15.2%)"
  "campaign metrics: CTR=5.71%, CPC=$2.87, clicks=8, impressions=140"

NEVER use vague sources like:
  "campaign data", "performance analysis", "account trends", "historical data"

If you cannot cite a specific number from the provided data, remove the recommendation entirely. Zero hallucinated recommendations is the goal. Fewer accurate items always beats more invented ones.

---

GENERAL RECOMMENDATIONS:
Provide strategic advice on budget, bidding, targeting, or landing page based on the performance data. Must have at least 3 items. Each must have a category.'''


                response_text = call_claude_api(prompt, system_prompt, 8192, model='claude-opus-4-6')
                # Strip markdown fences if present, then find the JSON object
                cleaned = response_text.strip()
                if cleaned.startswith('```'):
                    cleaned = cleaned.split('\n', 1)[-1]
                if cleaned.endswith('```'):
                    cleaned = cleaned.rsplit('```', 1)[0]
                cleaned = cleaned.strip()
                json_start = cleaned.find('{')
                json_end = cleaned.rfind('}')
                if json_start >= 0 and json_end > json_start:
                    cleaned = cleaned[json_start:json_end + 1]
                suggestions = json.loads(cleaned)

                result = {
                    **suggestions,
                    'periodInsights': period_insights,
                    'generatedAt': datetime.now().isoformat(),
                    'dateRange': {'startDate': start_date, 'endDate': end_date, 'daysAnalyzed': days_analyzed},
                }
                if plan_context_data:
                    result['planContext'] = plan_context_data
                if campaign_label:
                    result['campaignFilter'] = campaign_label

                ad_suggestion_jobs[job_id] = {'status': 'complete', 'result': result}

                if SUPABASE_URL:
                    requests.patch(
                        f"{SUPABASE_URL}/rest/v1/ad_suggestion_jobs",
                        params={'job_id': f'eq.{job_id}'},
                        headers=supabase_headers(),
                        json={
                            'status': 'complete',
                            'result': result,
                            'updated_at': datetime.now().isoformat(),
                        },
                        timeout=TIMEOUT
                    )

            except Exception as e:
                logger.error(f'Ad suggestions processing error: {str(e)}')
                ad_suggestion_jobs[job_id] = {'status': 'error', 'error': str(e)}

                if SUPABASE_URL:
                    requests.patch(
                        f"{SUPABASE_URL}/rest/v1/ad_suggestion_jobs",
                        params={'job_id': f'eq.{job_id}'},
                        headers=supabase_headers(),
                        json={
                            'status': 'error',
                            'error': str(e),
                            'updated_at': datetime.now().isoformat(),
                        },
                        timeout=TIMEOUT
                    )

        thread = threading.Thread(target=process_analysis)
        thread.start()

        return jsonify({
            'status': 'processing',
            'jobId': job_id,
            'message': f'Analysis started. Poll GET /api/dashboard/ad-suggestions?jobId={job_id}',
        }), 202

    except Exception as e:
        logger.error(f'Ad suggestions error: {str(e)}')
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


# ============================================================================
# CHECKLISTS ENDPOINTS
# ============================================================================

def call_claude_haiku(prompt, system_prompt='', max_retries=3):
    """Call Claude Haiku with retry logic and exponential backoff."""
    # Auto-append human voice rules to every system prompt
    effective_system = (system_prompt or '') + HUMAN_VOICE_RULES
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': ANTHROPIC_API_KEY,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-haiku-4-5-20251001',
                    'max_tokens': 4096,
                    'system': effective_system,
                    'messages': [{'role': 'user', 'content': prompt}]
                },
                timeout=(10, 120)
            )

            if response.status_code in [429, 529, 503]:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(f'Claude Haiku returned {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})')
                    time.sleep(delay)
                    continue
                raise Exception(f'Claude Haiku returned {response.status_code} after {max_retries} attempts')

            if not response.ok:
                raise Exception(f'Claude Haiku error: {response.status_code} {response.text[:200]}')

            result = response.json()
            usage = result.get('usage', {})
            _log_claude_usage(
                model='claude-haiku-4-5-20251001',
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                endpoint='haiku'
            )
            content = result.get('content', [])
            if content and content[0].get('type') == 'text':
                return content[0]['text']
            raise Exception('Empty response from Claude Haiku')

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning(f'Claude Haiku timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})')
                time.sleep(delay)
                continue
            raise
    raise Exception('Claude Haiku failed after all retries')


@dashboard_bp.route('/api/dashboard/checklists', methods=['GET', 'OPTIONS'])
def get_checklists():
    """Get checklist items with optional filters."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        params = {'select': '*', 'order': 'created_at.asc'}

        month = request.args.get('month')
        if month:
            params['month'] = f'eq.{month}'

        source_doc = request.args.get('source_doc')
        if source_doc:
            params['source_doc'] = f'eq.{source_doc}'

        status = request.args.get('status')
        if status:
            params['status'] = f'eq.{status}'

        category = request.args.get('category')
        if category:
            params['category'] = f'eq.{category}'

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_checklists",
            params=params,
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to fetch checklists: {response.text}')

        items = response.json()

        # Sort by priority: high first, then medium, then low
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        items.sort(key=lambda x: (priority_order.get(x.get('priority', 'low'), 2), x.get('created_at', '')))

        return jsonify(items)

    except Exception as e:
        logger.error(f'Checklists GET error: {str(e)}')
        return jsonify({'error': 'Failed to fetch checklists'}), 500


@dashboard_bp.route('/api/dashboard/checklists/<item_id>', methods=['PATCH', 'OPTIONS'])
def update_checklist(item_id):
    """Update a checklist item."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No update data provided'}), 400

        # Build update payload with only allowed fields
        allowed_fields = {'status', 'notes', 'priority', 'due_date', 'title', 'description', 'category'}
        update = {k: v for k, v in data.items() if k in allowed_fields}

        if not update:
            return jsonify({'error': 'No valid fields to update'}), 400

        # Handle completed_at logic
        if 'status' in update:
            if update['status'] == 'done':
                update['completed_at'] = datetime.now().isoformat()
            else:
                update['completed_at'] = None

        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/dmhoa_checklists?id=eq.{item_id}",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json=update,
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to update checklist: {response.text}')

        items = response.json()
        if not items:
            return jsonify({'error': 'Checklist item not found'}), 404

        return jsonify(items[0])

    except Exception as e:
        logger.error(f'Checklists PATCH error: {str(e)}')
        return jsonify({'error': 'Failed to update checklist'}), 500


@dashboard_bp.route('/api/dashboard/checklists/seed', methods=['POST', 'OPTIONS'])
def seed_checklists():
    """Seed checklists from strategic documents using Claude Haiku."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Anthropic API key not configured'}), 500

    try:
        data = request.get_json() or {}
        documents = data.get('documents')

        # If no documents provided, auto-fetch from doc_references
        if not documents:
            doc_refs_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_doc_references",
                params={'select': 'doc_key,doc_name,summary_text,key_points'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if doc_refs_resp.ok:
                documents = {}
                for ref in doc_refs_resp.json():
                    content_parts = [f"Document: {ref.get('doc_name', '')}"]
                    if ref.get('summary_text'):
                        content_parts.append(f"Summary: {ref['summary_text']}")
                    if ref.get('key_points'):
                        content_parts.append("Key Points:\n" + "\n".join(f"- {p}" for p in ref['key_points']))
                    documents[ref['doc_key']] = "\n\n".join(content_parts)

        if not documents:
            return jsonify({'error': 'No document references found.'}), 400

        # Check which docs already have checklists
        existing_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_checklists?select=source_doc",
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not existing_response.ok:
            raise Exception(f'Failed to check existing checklists: {existing_response.text}')

        existing_docs = set(item['source_doc'] for item in existing_response.json())

        seeded = {}
        skipped = []
        total = 0

        for doc_key, doc_content in documents.items():
            if doc_key in existing_docs:
                skipped.append(doc_key)
                continue

            if not doc_content or not doc_content.strip():
                skipped.append(doc_key)
                continue

            # Call Claude Haiku to extract tasks
            prompt = f"""You are generating actionable checklist tasks for DisputeMyHOA.com, a SaaS product that helps homeowners fight HOA violations using AI-powered legal analysis. The product has a funnel: quick preview → full preview → paid analysis.

Based on the document information below, generate 5-8 specific, actionable tasks. For each task, return a JSON object with:
- "title": Short task name (max 80 chars)
- "description": What specifically needs to be done (1-2 sentences)
- "category": One of ["google_ads", "content_seo", "social", "media", "product", "email", "ops", "finance", "legal"]
- "month": Integer 1-6 if the task is time-bound to a specific month in a 6-month plan, null if it's evergreen
- "priority": "high", "medium", or "low" based on business impact
- "due_date": null

Return ONLY a valid JSON array. No markdown, no commentary, no code fences.

Document:
{doc_content[:12000]}"""

            raw_response = call_claude_haiku(prompt)

            # Parse JSON — strip any accidental markdown fences
            cleaned = raw_response.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            tasks = json.loads(cleaned)

            if not isinstance(tasks, list):
                logger.warning(f'Claude returned non-list for {doc_key}, skipping')
                skipped.append(doc_key)
                continue

            # Insert tasks into Supabase
            rows = []
            valid_categories = {'google_ads', 'content_seo', 'social', 'media', 'product', 'email', 'ops', 'finance', 'legal'}
            valid_priorities = {'high', 'medium', 'low'}

            for task in tasks:
                row = {
                    'source_doc': doc_key,
                    'title': str(task.get('title', ''))[:80],
                    'description': str(task.get('description', '')),
                    'category': task.get('category', 'ops') if task.get('category') in valid_categories else 'ops',
                    'month': task.get('month') if isinstance(task.get('month'), int) and 1 <= task.get('month', 0) <= 6 else None,
                    'priority': task.get('priority', 'medium') if task.get('priority') in valid_priorities else 'medium',
                    'due_date': task.get('due_date') if task.get('due_date') else None,
                    'status': 'pending',
                }
                rows.append(row)

            if rows:
                insert_response = requests.post(
                    f"{SUPABASE_URL}/rest/v1/dmhoa_checklists",
                    headers={**supabase_headers(), 'Prefer': 'return=representation'},
                    json=rows,
                    timeout=TIMEOUT
                )

                if not insert_response.ok:
                    logger.error(f'Failed to insert checklists for {doc_key}: {insert_response.text}')
                    skipped.append(doc_key)
                    continue

                seeded[doc_key] = len(rows)
                total += len(rows)
            else:
                skipped.append(doc_key)

        return jsonify({
            'seeded': seeded,
            'skipped': skipped,
            'total': total,
        })

    except json.JSONDecodeError as e:
        logger.error(f'Checklists seed JSON parse error: {str(e)}')
        return jsonify({'error': 'Failed to parse AI response as JSON'}), 500
    except Exception as e:
        logger.error(f'Checklists seed error: {str(e)}')
        return jsonify({'error': 'Failed to seed checklists'}), 500


# ============================================================================
# DOCUMENT REFERENCES ENDPOINTS
# ============================================================================

DOC_URLS = {
    '6month_plan': 'https://docs.google.com/document/d/1edSWxbDRH6NvaXgTZFcV0XIFJk9XzBv_/edit',
    'scenario': 'https://docs.google.com/document/d/1midUedXwq4Dc6cXZspx37vuAFI5kH62r/edit',
    'media_plan': 'https://docs.google.com/document/d/1fpdvSdnmegUcEXP9ch7ULMD8cliNac2Y/edit',
    'persona': 'https://docs.google.com/document/d/1E-s4qfKkyJugXndEq8qg0-iRqp6AT2WL/edit',
    'identity': 'https://docs.google.com/document/d/1734OW_IZuzhrIzZ4eTghqbK3KPhQnjAp/edit',
    'dev_system': 'https://docs.google.com/document/d/1qmdYKzCw4Jc4T5AfX1BX2c4FYagBt41g/edit',
}

@dashboard_bp.route('/api/dashboard/doc-references', methods=['GET', 'OPTIONS'])
def get_doc_references():
    """Get all document references with summaries and links."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_doc_references",
            params={'select': '*', 'order': 'doc_name.asc'},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to fetch doc references: {response.text}')

        docs = response.json()
        for doc in docs:
            doc['doc_url'] = DOC_URLS.get(doc.get('doc_key'))
        return jsonify(docs)

    except Exception as e:
        logger.error(f'Doc references GET error: {str(e)}')
        return jsonify({'error': 'Failed to fetch document references'}), 500


@dashboard_bp.route('/api/dashboard/doc-references/refresh', methods=['POST', 'OPTIONS'])
def refresh_doc_reference():
    """Regenerate AI summary for a specific document."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Anthropic API key not configured'}), 500

    try:
        data = request.get_json()
        if not data or not data.get('doc_key') or not data.get('doc_content'):
            return jsonify({'error': 'Request body must include "doc_key" and "doc_content"'}), 400

        doc_key = data['doc_key']
        doc_content = data['doc_content']

        prompt = f"""Summarize this business document for a solo founder/developer who needs a quick reference.

Provide:
1. "summary_text": A 3-4 sentence executive summary
2. "key_points": A JSON array of 5-7 key takeaway strings (max 100 chars each)

Return as a JSON object with those two fields. No markdown, no code fences.

Document:
{doc_content[:12000]}"""

        raw_response = call_claude_haiku(prompt)

        # Parse JSON
        cleaned = raw_response.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        summary_text = parsed.get('summary_text', '')
        key_points = parsed.get('key_points', [])

        # Update the doc reference in Supabase
        update_response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/dmhoa_doc_references?doc_key=eq.{doc_key}",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json={
                'summary_text': summary_text,
                'key_points': key_points,
                'last_refreshed': datetime.now().isoformat(),
            },
            timeout=TIMEOUT
        )

        if not update_response.ok:
            raise Exception(f'Failed to update doc reference: {update_response.text}')

        items = update_response.json()
        if not items:
            return jsonify({'error': f'Document reference with key "{doc_key}" not found'}), 404

        return jsonify(items[0])

    except json.JSONDecodeError as e:
        logger.error(f'Doc reference refresh JSON parse error: {str(e)}')
        return jsonify({'error': 'Failed to parse AI summary response'}), 500
    except Exception as e:
        logger.error(f'Doc reference refresh error: {str(e)}')
        return jsonify({'error': 'Failed to refresh document reference'}), 500


# ============================================================================
# ALERTS SYSTEM — HELPERS
# ============================================================================

def send_alert_email(title, message):
    """Send a critical alert email to the admin."""
    try:
        if not ADMIN_EMAIL:
            logger.warning('ADMIN_EMAIL not set, skipping alert email')
            return

        if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
            logger.warning('SMTP not configured, skipping alert email')
            return

        msg = MIMEMultipart()
        msg['From'] = SMTP_FROM
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f'DMHOA Alert: {title}'

        body = f"""DMHOA Dashboard Alert
=====================
Severity: CRITICAL
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

{title}

{message}

---
View all alerts in your dashboard."""

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [ADMIN_EMAIL], msg.as_string())

        logger.info(f'Alert email sent to {ADMIN_EMAIL}: {title}')
    except Exception as e:
        logger.error(f'Failed to send alert email: {str(e)}')


def _fetch_google_ads_metrics(period: str = 'today') -> Optional[Dict]:
    """Fetch Google Ads performance metrics. Only includes campaigns with ad spend. Returns dict or None."""
    has_credentials = all([
        GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CUSTOMER_ID,
        GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN
    ])
    if not has_credentials:
        return None

    try:
        date_range = get_google_ads_date_range(period)
        access_token = get_google_ads_access_token()
        if not access_token:
            return None

        query = f"""
            SELECT
                metrics.impressions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.conversions
            FROM campaign
            WHERE segments.date BETWEEN '{date_range["startDate"]}' AND '{date_range["endDate"]}'
                AND campaign.status != 'REMOVED'
                AND metrics.cost_micros > 0
        """
        results = query_google_ads(GOOGLE_ADS_CUSTOMER_ID, access_token, query)

        total_spend = 0
        total_clicks = 0
        total_impressions = 0
        total_conversions = 0

        for row in results:
            metrics = row.get('metrics', {})
            total_spend += int(metrics.get('costMicros', 0) or 0) / 1_000_000
            total_clicks += int(metrics.get('clicks', 0) or 0)
            total_impressions += int(metrics.get('impressions', 0) or 0)
            total_conversions += float(metrics.get('conversions', 0) or 0)

        ctr = (total_clicks / total_impressions) if total_impressions > 0 else 0
        cpa = (total_spend / total_conversions) if total_conversions > 0 else 0

        return {
            'spend': round(total_spend, 2),
            'clicks': total_clicks,
            'impressions': total_impressions,
            'conversions': round(total_conversions, 2),
            'ctr': ctr,
            'cpa': round(cpa, 2),
        }
    except Exception as e:
        logger.error(f'Alert scan - Google Ads fetch failed: {str(e)}')
        return None


def _fetch_openai_usage_metrics() -> Optional[Dict]:
    """Fetch OpenAI usage metrics for the alert scan. Returns dict or None."""
    if not OPENAI_ADMIN_KEY:
        return None

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        today = end_date.strftime('%Y-%m-%d')

        response = requests.get(
            'https://api.openai.com/v1/organization/usage/completions',
            params={
                'start_time': int(start_date.timestamp()),
                'end_time': int(end_date.timestamp()),
                'bucket_width': '1d',
            },
            headers={
                'Authorization': f'Bearer {OPENAI_ADMIN_KEY}',
                'Content-Type': 'application/json',
            },
            timeout=TIMEOUT
        )

        if not response.ok:
            return None

        usage_data = response.json()
        buckets = usage_data.get('data', []) or usage_data.get('buckets', [])

        today_cost = 0
        daily_costs = []

        for bucket in buckets:
            bucket_date = datetime.fromtimestamp(bucket.get('start_time', 0)).strftime('%Y-%m-%d') if bucket.get('start_time') else 'unknown'
            day_cost = 0

            results = bucket.get('results', [bucket])
            for item in results:
                input_tokens = item.get('input_tokens', 0) or item.get('prompt_tokens', 0) or 0
                output_tokens = item.get('output_tokens', 0) or item.get('completion_tokens', 0) or 0
                model = item.get('model', 'default')
                pricing = OPENAI_PRICING.get(model, OPENAI_PRICING['default'])
                day_cost += (input_tokens / 1000) * pricing['input'] + (output_tokens / 1000) * pricing['output']

            if bucket_date == today:
                today_cost = day_cost
            daily_costs.append(day_cost)

        # 7-day average (exclude today)
        recent_costs = daily_costs[-8:-1] if len(daily_costs) > 1 else []
        avg_7d = sum(recent_costs) / len(recent_costs) if recent_costs else 0

        return {
            'today_cost': round(today_cost, 2),
            'avg_daily_7d': round(avg_7d, 2),
        }
    except Exception as e:
        logger.error(f'Alert scan - OpenAI usage fetch failed: {str(e)}')
        return None


def _fetch_claude_usage_metrics(period: str = 'month') -> Optional[Dict]:
    """Fetch Claude usage from Supabase dmhoa_claude_usage table for a given period."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None

    try:
        now = datetime.now()
        if period == 'yesterday':
            yesterday = now - timedelta(days=1)
            period_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'today':
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            period_start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # month
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        start_str = period_start.strftime('%Y-%m-%dT%H:%M:%S')

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_claude_usage",
            params={
                'select': 'model,input_tokens,output_tokens,cost,created_at',
                'created_at': f'gte.{start_str}',
                'order': 'created_at.desc',
                'limit': '5000',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            logger.warning(f'Claude usage table query failed: {response.status_code}')
            return None

        rows = response.json()
        total_cost = sum(r.get('cost', 0) for r in rows)
        total_input = sum(r.get('input_tokens', 0) for r in rows)
        total_output = sum(r.get('output_tokens', 0) for r in rows)
        calls = len(rows)

        # Today's cost
        today_str = now.strftime('%Y-%m-%d')
        today_cost = sum(r.get('cost', 0) for r in rows
                         if r.get('created_at', '').startswith(today_str))

        # Average daily
        if period in ('today', 'yesterday'):
            num_days = 1
        elif period == 'week':
            num_days = 7
        else:
            num_days = max(now.day, 1)
        avg_daily = total_cost / num_days if num_days > 0 else 0

        return {
            'mtd_cost': round(total_cost, 4),
            'today_cost': round(today_cost, 4),
            'avg_daily': round(avg_daily, 4),
            'total_calls': calls,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
        }
    except Exception as e:
        logger.error(f'Claude usage metrics fetch failed: {str(e)}')
        return None


# ============================================================================
# ALERTS ENDPOINTS
# ============================================================================

@dashboard_bp.route('/api/dashboard/alerts', methods=['GET', 'OPTIONS'])
def get_alerts():
    """Get alerts with optional filters."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        limit = request.args.get('limit', '50')
        offset = request.args.get('offset', '0')

        params = {
            'select': '*',
            'order': 'created_at.desc',
            'limit': limit,
            'offset': offset,
        }

        severity = request.args.get('severity')
        if severity:
            params['severity'] = f'eq.{severity}'

        acknowledged = request.args.get('acknowledged')
        if acknowledged is not None and acknowledged in ('true', 'false'):
            params['acknowledged'] = f'eq.{acknowledged}'

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_alerts",
            params=params,
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to fetch alerts: {response.text}')

        alerts = response.json()

        # Fetch unacknowledged counts by severity
        counts_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_alerts",
            params={
                'select': 'severity',
                'acknowledged': 'eq.false',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        unack_counts = {'critical': 0, 'warning': 0, 'info': 0}
        if counts_response.ok:
            for alert in counts_response.json():
                sev = alert.get('severity', 'info')
                if sev in unack_counts:
                    unack_counts[sev] += 1

        return jsonify({
            'alerts': alerts,
            'unacknowledged_counts': unack_counts,
        })

    except Exception as e:
        logger.error(f'Alerts GET error: {str(e)}')
        return jsonify({'error': 'Failed to fetch alerts'}), 500


@dashboard_bp.route('/api/dashboard/alerts/<alert_id>/ack', methods=['PATCH', 'OPTIONS'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/dmhoa_alerts?id=eq.{alert_id}",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json={
                'acknowledged': True,
                'acknowledged_at': datetime.now().isoformat(),
            },
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to acknowledge alert: {response.text}')

        items = response.json()
        if not items:
            return jsonify({'error': 'Alert not found'}), 404

        return jsonify(items[0])

    except Exception as e:
        logger.error(f'Alert ACK error: {str(e)}')
        return jsonify({'error': 'Failed to acknowledge alert'}), 500


def _execute_alert_scan():
    """Core alert scan logic — can be called from route handler or scheduler."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return {'error': 'Supabase not configured', 'alerts_created': 0}

    alerts_created = []

    def create_alert_if_new(alert_type, severity, title, message, data=None):
        """Only create an alert if no duplicate unacknowledged alert exists within the last 2 hours."""
        try:
            two_hours_ago = (datetime.utcnow() - timedelta(hours=2)).isoformat()

            existing_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_alerts",
                params={
                    'alert_type': f'eq.{alert_type}',
                    'title': f'eq.{title}',
                    'acknowledged': 'eq.false',
                    'created_at': f'gte.{two_hours_ago}',
                    'select': 'id',
                    'limit': '1',
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )

            if existing_response.ok and existing_response.json():
                return None  # Duplicate — skip

            alert = {
                'alert_type': alert_type,
                'severity': severity,
                'title': title,
                'message': message,
                'data': data or {},
            }

            insert_response = requests.post(
                f"{SUPABASE_URL}/rest/v1/dmhoa_alerts",
                headers={**supabase_headers(), 'Prefer': 'return=representation'},
                json=alert,
                timeout=TIMEOUT
            )

            if insert_response.ok:
                result = insert_response.json()
                created = result[0] if isinstance(result, list) and result else result
                alerts_created.append(created)

                if severity == 'critical':
                    send_alert_email(title, message)

                return created
            else:
                logger.error(f'Failed to insert alert: {insert_response.text}')
                return None
        except Exception as e:
            logger.error(f'create_alert_if_new error: {str(e)}')
            return None

    # === CHECK 1: SITE HEALTH ===
    try:
        resp = requests.get('https://disputemyhoa.com', timeout=5)
        if resp.status_code != 200:
            create_alert_if_new(
                'site_down', 'critical',
                'Site returning non-200 status',
                f'disputemyhoa.com returned HTTP {resp.status_code}',
                {'status_code': resp.status_code}
            )
    except requests.exceptions.Timeout:
        create_alert_if_new(
            'site_down', 'critical',
            'Site timeout',
            'disputemyhoa.com did not respond within 5 seconds',
            {'timeout': 5}
        )
    except requests.exceptions.ConnectionError:
        create_alert_if_new(
            'site_down', 'critical',
            'Site unreachable',
            'Could not connect to disputemyhoa.com',
            {}
        )
    except Exception as e:
        logger.error(f'Alert scan - Site health check failed: {str(e)}')

    # === CHECK 2: STRIPE PAYMENT FAILURES ===
    try:
        if STRIPE_SECRET_KEY:
            import stripe
            stripe.api_key = STRIPE_SECRET_KEY
            one_hour_ago = int((datetime.utcnow() - timedelta(hours=1)).timestamp())

            failed_charges = stripe.Charge.list(
                created={'gte': one_hour_ago},
                limit=10
            )
            failed = [c for c in failed_charges.data if c.status == 'failed']
            if failed:
                create_alert_if_new(
                    'payment_failure', 'warning',
                    f'{len(failed)} failed payment(s) in last hour',
                    f'Stripe recorded {len(failed)} failed charge(s). Check Stripe dashboard.',
                    {'failed_count': len(failed), 'charge_ids': [c.id for c in failed]}
                )

            refunds = stripe.Refund.list(created={'gte': one_hour_ago}, limit=10)
            if refunds.data:
                create_alert_if_new(
                    'payment_failure', 'warning',
                    f'{len(refunds.data)} refund(s) issued in last hour',
                    'Check Stripe dashboard for refund details.',
                    {'refund_count': len(refunds.data)}
                )
    except Exception as e:
        logger.error(f'Alert scan - Stripe check failed: {str(e)}')

    # === CHECK 3: GOOGLE ADS PERFORMANCE ===
    try:
        ads_data = _fetch_google_ads_metrics()
        if ads_data:
            if ads_data['cpa'] > 100:
                create_alert_if_new(
                    'ad_performance', 'warning',
                    'Google Ads CPA spike',
                    f"Current CPA: ${ads_data['cpa']:.2f} (threshold: $100). Review keyword bids.",
                    {'cpa': ads_data['cpa'], 'threshold': 100}
                )
            if ads_data['impressions'] > 0 and ads_data['ctr'] < 0.015:
                create_alert_if_new(
                    'ad_performance', 'warning',
                    'Google Ads CTR below threshold',
                    f"Current CTR: {ads_data['ctr']*100:.1f}% (threshold: 1.5%). Review ad copy.",
                    {'ctr': ads_data['ctr'], 'threshold': 0.015}
                )
    except Exception as e:
        logger.error(f'Alert scan - Google Ads check failed: {str(e)}')

    # === CHECK 4: REVENUE ANOMALY (no revenue for 72+ hours) ===
    try:
        if STRIPE_SECRET_KEY:
            import stripe
            stripe.api_key = STRIPE_SECRET_KEY
            three_days_ago = int((datetime.utcnow() - timedelta(days=3)).timestamp())
            thirty_days_ago = int((datetime.utcnow() - timedelta(days=30)).timestamp())

            recent_charges = stripe.Charge.list(
                created={'gte': three_days_ago},
                status='succeeded',
                limit=1
            )
            monthly_charges = stripe.Charge.list(
                created={'gte': thirty_days_ago},
                status='succeeded',
                limit=100
            )

            has_monthly_revenue = len(monthly_charges.data) > 0
            has_recent_revenue = len(recent_charges.data) > 0

            if has_monthly_revenue and not has_recent_revenue:
                create_alert_if_new(
                    'revenue', 'warning',
                    'No revenue in 72+ hours',
                    'No successful charges in the last 3 days. Monthly average suggests this is unusual.',
                    {'days_since_last': 3}
                )
    except Exception as e:
        logger.error(f'Alert scan - Revenue check failed: {str(e)}')

    # === CHECK 5: API COST SPIKE ===
    try:
        openai_data = _fetch_openai_usage_metrics()
        if openai_data:
            daily = openai_data['today_cost']
            avg_7d = openai_data['avg_daily_7d']
            if avg_7d > 0 and daily > avg_7d * 2:
                create_alert_if_new(
                    'cost_spike', 'warning',
                    'OpenAI API cost spike',
                    f"Today: ${daily:.2f} vs 7-day avg: ${avg_7d:.2f}. Check for unusual usage.",
                    {'today': daily, 'avg_7d': avg_7d}
                )
    except Exception as e:
        logger.error(f'Alert scan - OpenAI cost check failed: {str(e)}')

    # === CHECK 6: STUCK CASE ANALYSES ===
    try:
        ten_min_ago = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        one_hour_ago_iso = (datetime.utcnow() - timedelta(hours=1)).isoformat()

        # Find paid cases from the last hour that should have outputs by now
        # Use PostgREST 'and' filter to combine two conditions on created_at
        paid_cases_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
            params={
                'select': 'id,token,created_at',
                'status': 'eq.paid',
                'and': f'(created_at.gte.{one_hour_ago_iso},created_at.lte.{ten_min_ago})',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if paid_cases_response.ok:
            paid_cases = paid_cases_response.json()
            if paid_cases:
                # Check which have outputs
                tokens = [c.get('token') for c in paid_cases if c.get('token')]
                stuck_count = 0

                for token in tokens:
                    output_response = requests.get(
                        f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs",
                        params={
                            'select': 'id',
                            'case_token': f'eq.{token}',
                            'limit': '1',
                        },
                        headers=supabase_headers(),
                        timeout=TIMEOUT
                    )
                    if output_response.ok and not output_response.json():
                        stuck_count += 1

                if stuck_count > 0:
                    create_alert_if_new(
                        'case_failure', 'critical',
                        f'{stuck_count} stuck case analysis(es)',
                        f'{stuck_count} paid case(s) from the last hour have no generated output. Check case pipeline.',
                        {'stuck_count': stuck_count}
                    )
    except Exception as e:
        logger.error(f'Alert scan - Case analysis check failed: {str(e)}')

    # === CHECK 7: CONVERSION RATE ===
    try:
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()

        previews_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
            params={
                'select': 'id',
                'created_at': f'gte.{seven_days_ago}',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        paid_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
            params={
                'select': 'id',
                'status': 'eq.paid',
                'created_at': f'gte.{seven_days_ago}',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if previews_response.ok and paid_response.ok:
            previews = previews_response.json()
            paid_cases = paid_response.json()

            if len(previews) > 10:
                rate = len(paid_cases) / len(previews)
                if rate < 0.02:
                    create_alert_if_new(
                        'anomaly', 'warning',
                        'Low conversion rate',
                        f'Preview-to-paid: {rate*100:.1f}% (last 7 days). Threshold: 2%.',
                        {'rate': rate, 'previews': len(previews), 'paid': len(paid_cases)}
                    )
    except Exception as e:
        logger.error(f'Alert scan - Conversion check failed: {str(e)}')

    # === CHECK 8: GITHUB DEPENDABOT SECURITY ALERTS ===
    try:
        if GITHUB_TOKEN:
            github_headers = {
                'Authorization': f'Bearer {GITHUB_TOKEN}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            }
            for repo in GITHUB_REPOS:
                try:
                    gh_resp = requests.get(
                        f'https://api.github.com/repos/{repo}/dependabot/alerts',
                        headers=github_headers,
                        params={'state': 'open', 'severity': 'high,critical', 'per_page': '100'},
                        timeout=10
                    )
                    if gh_resp.ok:
                        alerts = gh_resp.json()
                        if alerts:
                            critical = [a for a in alerts if a.get('security_vulnerability', {}).get('severity') == 'critical']
                            high = [a for a in alerts if a.get('security_vulnerability', {}).get('severity') == 'high']
                            repo_short = repo.split('/')[-1]
                            packages = ', '.join(set(
                                a.get('dependency', {}).get('package', {}).get('name', '?')
                                for a in alerts[:5]
                            ))
                            severity = 'critical' if critical else 'warning'
                            create_alert_if_new(
                                'dependabot', severity,
                                f'{len(alerts)} Dependabot alert(s) in {repo_short}',
                                f'{len(critical)} critical, {len(high)} high severity. '
                                f'Affected packages: {packages}.',
                                {
                                    'repo': repo,
                                    'total': len(alerts),
                                    'critical': len(critical),
                                    'high': len(high),
                                    'packages': [
                                        {
                                            'name': a.get('dependency', {}).get('package', {}).get('name'),
                                            'severity': a.get('security_vulnerability', {}).get('severity'),
                                            'advisory': a.get('security_advisory', {}).get('summary'),
                                            'patched_version': (a.get('security_vulnerability', {})
                                                                .get('first_patched_version', {}) or {}).get('identifier'),
                                            'url': a.get('html_url'),
                                        }
                                        for a in alerts
                                    ],
                                }
                            )
                    elif gh_resp.status_code == 403:
                        logger.warning(f'Dependabot check - insufficient permissions for {repo}')
                    elif gh_resp.status_code == 404:
                        logger.warning(f'Dependabot check - alerts not enabled for {repo}')
                    else:
                        logger.warning(f'Dependabot check - {repo} returned {gh_resp.status_code}')
                except Exception as e:
                    logger.error(f'Alert scan - Dependabot check failed for {repo}: {str(e)}')
    except Exception as e:
        logger.error(f'Alert scan - Dependabot check failed: {str(e)}')

    # === CHECK 9: SUPABASE SECURITY ADVISOR ===
    try:
        if SUPABASE_ACCESS_TOKEN and SUPABASE_PROJECT_REF:
            sb_resp = requests.get(
                f'https://api.supabase.com/v1/projects/{SUPABASE_PROJECT_REF}/advisors/security',
                headers={
                    'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}',
                    'Content-Type': 'application/json',
                },
                timeout=15
            )
            if sb_resp.ok:
                sb_data = sb_resp.json()
                lints = sb_data.get('lints', []) if isinstance(sb_data, dict) else sb_data
                # Filter to ERROR and WARN level issues
                errors = [l for l in lints if l.get('level') == 'ERROR']
                warnings = [l for l in lints if l.get('level') == 'WARN']

                if errors:
                    issue_list = ', '.join(set(l.get('title', l.get('name', '?')) for l in errors[:5]))
                    create_alert_if_new(
                        'supabase_security', 'critical',
                        f'{len(errors)} Supabase security error(s)',
                        f'Security advisor found {len(errors)} error(s) and {len(warnings)} warning(s). '
                        f'Issues: {issue_list}.',
                        {
                            'errors': len(errors),
                            'warnings': len(warnings),
                            'issues': [
                                {
                                    'name': l.get('name'),
                                    'title': l.get('title'),
                                    'level': l.get('level'),
                                    'detail': l.get('detail'),
                                    'remediation': l.get('remediation'),
                                    'metadata': l.get('metadata'),
                                }
                                for l in errors
                            ],
                        }
                    )
                elif warnings:
                    issue_list = ', '.join(set(l.get('title', l.get('name', '?')) for l in warnings[:5]))
                    create_alert_if_new(
                        'supabase_security', 'warning',
                        f'{len(warnings)} Supabase security warning(s)',
                        f'Security advisor found {len(warnings)} warning(s). Issues: {issue_list}.',
                        {
                            'errors': 0,
                            'warnings': len(warnings),
                            'issues': [
                                {
                                    'name': l.get('name'),
                                    'title': l.get('title'),
                                    'level': l.get('level'),
                                    'detail': l.get('detail'),
                                    'remediation': l.get('remediation'),
                                    'metadata': l.get('metadata'),
                                }
                                for l in warnings
                            ],
                        }
                    )
            elif sb_resp.status_code == 401:
                logger.warning('Supabase security check - invalid access token')
            else:
                logger.warning(f'Supabase security check returned {sb_resp.status_code}: {sb_resp.text[:200]}')
    except Exception as e:
        logger.error(f'Alert scan - Supabase security check failed: {str(e)}')

    return {
        'scan_completed': True,
        'alerts_created': len(alerts_created),
        'timestamp': datetime.utcnow().isoformat(),
    }


@dashboard_bp.route('/api/dashboard/alerts/scan', methods=['POST', 'OPTIONS'])
def run_alert_scan():
    """Run all health and performance checks, create alerts for triggered conditions."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    result = _execute_alert_scan()
    if 'error' in result:
        return jsonify(result), 500
    return jsonify(result)


# ============================================================================
# PHASE 3 — SHARED DATA-FETCHING HELPERS
# ============================================================================

# In-memory cache for command center (60-second TTL)
_command_center_cache = {'data': None, 'ts': 0}


def _fetch_stripe_metrics(period='today') -> Dict:
    """Fetch Stripe revenue metrics internally (no Flask request needed)."""
    result = {'revenue': 0, 'transactions': 0, 'refunds': 0, 'refund_amount': 0, 'mrr': 0}
    if not STRIPE_SECRET_KEY:
        return result
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        ts_range = get_timestamp_range(period)

        charges = stripe.Charge.list(created={'gte': ts_range['gte'], 'lte': ts_range['lte']}, limit=100)
        successful = [c for c in charges.data if c.status == 'succeeded' and not c.refunded]
        result['revenue'] = round(sum(c.amount for c in successful) / 100, 2)
        result['transactions'] = len(successful)

        refunds = stripe.Refund.list(created={'gte': ts_range['gte'], 'lte': ts_range['lte']}, limit=100)
        result['refunds'] = len(refunds.data)
        result['refund_amount'] = round(sum(r.amount for r in refunds.data) / 100, 2)
    except Exception as e:
        logger.error(f'_fetch_stripe_metrics error: {str(e)}')
    return result


def _fetch_supabase_case_metrics(period='today') -> Dict:
    """Fetch case analytics from Supabase internally."""
    result = {'new_cases': 0, 'paid_cases': 0, 'total_cases': 0, 'conversion_rate': 0}
    if not SUPABASE_URL:
        return result
    try:
        date_range = get_date_range(period)
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
            params={
                'select': 'id,status,email,payload',
                'created_at': f'gte.{date_range["start"]}',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        if response.ok:
            cases = response.json()
            filtered = [c for c in cases if not any(
                exc.lower() == ((c.get('payload') or {}).get('email') or c.get('email') or '').lower()
                for exc in EXCLUDED_EMAILS
            )]
            result['total_cases'] = len(filtered)
            result['new_cases'] = len(filtered)
            paid = [c for c in filtered if c.get('status') == 'paid']
            result['paid_cases'] = len(paid)
            result['conversion_rate'] = round(len(paid) / len(filtered) * 100, 1) if filtered else 0
    except Exception as e:
        logger.error(f'_fetch_supabase_case_metrics error: {str(e)}')
    return result


def _fetch_klaviyo_metrics() -> Dict:
    """Fetch Klaviyo metrics internally — total profiles and new subscribers today."""
    result = {'total_profiles': 0, 'new_today': 0}
    if not KLAVIYO_API_KEY:
        return result
    try:
        # Count total profiles by paginating (meta.page_info.total not available in 2024 API)
        total = 0
        next_url = '/profiles/?page[size]=100'
        while next_url:
            response = requests.get(
                f'https://a.klaviyo.com/api{next_url}',
                headers=klaviyo_headers(),
                timeout=TIMEOUT
            )
            if not response.ok:
                break
            page_data = response.json()
            total += len(page_data.get('data', []))
            next_link = page_data.get('links', {}).get('next')
            if next_link:
                from urllib.parse import urlparse
                parsed = urlparse(next_link)
                next_url = parsed.path + ('?' + parsed.query if parsed.query else '')
            else:
                next_url = None
        result['total_profiles'] = total

        since = (datetime.now() - timedelta(hours=24)).isoformat()
        new_response = requests.get(
            f'https://a.klaviyo.com/api/profiles/?filter=greater-or-equal(created,{since})&page[size]=100',
            headers=klaviyo_headers(),
            timeout=TIMEOUT
        )
        if new_response.ok:
            result['new_today'] = len(new_response.json().get('data', []))
    except Exception as e:
        logger.error(f'_fetch_klaviyo_metrics error: {str(e)}')
    return result


def _fetch_alert_counts() -> Dict:
    """Fetch unacknowledged alert counts by severity."""
    counts = {'critical': 0, 'warning': 0, 'info': 0, 'total': 0}
    if not SUPABASE_URL:
        return counts
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_alerts",
            params={'select': 'severity', 'acknowledged': 'eq.false'},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        if response.ok:
            for alert in response.json():
                sev = alert.get('severity', 'info')
                if sev in counts:
                    counts[sev] += 1
            counts['total'] = counts['critical'] + counts['warning'] + counts['info']
    except Exception as e:
        logger.error(f'_fetch_alert_counts error: {str(e)}')
    return counts


def _fetch_checklist_progress() -> Dict:
    """Fetch checklist completion stats."""
    result = {'done': 0, 'total': 0, 'pct': 0, 'top_pending': []}
    if not SUPABASE_URL:
        return result
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_checklists",
            params={'select': 'id,title,status,priority,category'},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        if response.ok:
            items = response.json()
            result['total'] = len(items)
            result['done'] = sum(1 for i in items if i.get('status') == 'done')
            result['pct'] = round(result['done'] / result['total'] * 100, 1) if result['total'] > 0 else 0

            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            pending = [i for i in items if i.get('status') in ('pending', 'in_progress')]
            pending.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
            result['top_pending'] = pending[:5]
    except Exception as e:
        logger.error(f'_fetch_checklist_progress error: {str(e)}')
    return result


def _fetch_doc_summaries() -> Dict[str, str]:
    """Fetch document summaries from dmhoa_doc_references."""
    summaries = {}
    if not SUPABASE_URL:
        return summaries
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_doc_references",
            params={'select': 'doc_key,summary_text,key_points'},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        if response.ok:
            for doc in response.json():
                key = doc.get('doc_key', '')
                summary = doc.get('summary_text') or ''
                points = doc.get('key_points') or []
                if isinstance(points, list) and points:
                    summary += '\nKey points: ' + '; '.join(str(p) for p in points)
                summaries[key] = summary
    except Exception as e:
        logger.error(f'_fetch_doc_summaries error: {str(e)}')
    return summaries


def _read_api_cache(cache_key: str) -> Optional[Dict]:
    """Read cached data from api_cache table. Returns data dict or None."""
    if not SUPABASE_URL:
        return None
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/api_cache",
            params={'cache_key': f'eq.{cache_key}', 'select': 'data,updated_at'},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        if response.ok:
            rows = response.json()
            if rows:
                return rows[0].get('data')
    except Exception as e:
        logger.error(f'_read_api_cache({cache_key}) error: {str(e)}')
    return None


def _build_live_data_snapshot() -> Dict:
    """Build aggregated live data for the chatbot and daily summary. Tolerates individual source failures."""
    sources_loaded = []
    sources_failed = []
    data = {}

    # Stripe — 'yesterday' is the completed day (digest fires at 6am ET so
    # 'today' would always be ~empty); MTD stays as-is.
    try:
        stripe_yest = _fetch_stripe_metrics('yesterday')
        stripe_week = _fetch_stripe_metrics('week')
        stripe_month = _fetch_stripe_metrics('month')
        data['stripe'] = {
            'revenue_yesterday': stripe_yest['revenue'],
            'revenue_week': stripe_week['revenue'],
            'revenue_month': stripe_month['revenue'],
            'transactions_yesterday': stripe_yest['transactions'],
            'transactions_month': stripe_month['transactions'],
        }
        sources_loaded.append('stripe')
    except Exception as e:
        logger.error(f'Snapshot - Stripe failed: {str(e)}')
        sources_failed.append('stripe')
        data['stripe'] = {}

    # Supabase cases — yesterday for activity, month for the trend
    try:
        cases_yest = _fetch_supabase_case_metrics('yesterday')
        cases_month = _fetch_supabase_case_metrics('month')
        data['cases'] = {
            'new_yesterday': cases_yest['new_cases'],
            'paid_yesterday': cases_yest['paid_cases'],
            'paid_month': cases_month['paid_cases'],
            'conversion_rate': cases_month['conversion_rate'],
        }
        sources_loaded.append('supabase')
    except Exception as e:
        logger.error(f'Snapshot - Supabase failed: {str(e)}')
        sources_failed.append('supabase')
        data['cases'] = {}

    # Google Ads — yesterday's completed day
    try:
        ads = _fetch_google_ads_metrics('yesterday')
        if ads:
            ads['ctr_pct'] = round(ads.get('ctr', 0) * 100, 2)
            data['google_ads'] = ads
            sources_loaded.append('google_ads')
        else:
            sources_failed.append('google_ads')
            data['google_ads'] = {}
    except Exception as e:
        logger.error(f'Snapshot - Google Ads failed: {str(e)}')
        sources_failed.append('google_ads')
        data['google_ads'] = {}

    # Email — Resend (sends) + email_funnel (capture/stages). Replaces
    # the old Klaviyo block. Klaviyo profile sync still runs in save_case
    # for legacy flows but is no longer the source of truth for metrics.
    try:
        data['email'] = _fetch_email_metrics()
        sources_loaded.append('email')
    except Exception as e:
        logger.error(f'Snapshot - Email metrics failed: {str(e)}')
        sources_failed.append('email')
        data['email'] = {}

    # OpenAI costs
    try:
        openai = _fetch_openai_usage_metrics()
        data['openai'] = openai or {}
        if openai:
            sources_loaded.append('openai')
        else:
            sources_failed.append('openai')
    except Exception as e:
        logger.error(f'Snapshot - OpenAI failed: {str(e)}')
        sources_failed.append('openai')
        data['openai'] = {}

    # Alerts
    try:
        data['alerts'] = _fetch_alert_counts()
        sources_loaded.append('alerts')
    except Exception:
        data['alerts'] = {'critical': 0, 'warning': 0, 'info': 0, 'total': 0}

    # Checklists
    try:
        data['checklists'] = _fetch_checklist_progress()
        sources_loaded.append('checklists')
    except Exception:
        data['checklists'] = {'done': 0, 'total': 0, 'pct': 0}

    # Extra ops metrics: legal referrals, ad-analyzer proposals, previews
    try:
        data['ops'] = _fetch_ops_metrics()
        sources_loaded.append('ops')
    except Exception as e:
        logger.error(f'Snapshot - ops metrics failed: {e}')
        data['ops'] = {}

    data['_sources_loaded'] = sources_loaded
    data['_sources_failed'] = sources_failed
    return data


def _fetch_ops_metrics() -> Dict:
    """Counts for legal referrals, pending ad-analyzer proposals, and
    previews generated YESTERDAY (completed day). pending_proposals is
    current state since the digest fires at 6am ET when nothing has
    happened today yet."""
    out = {
        'legal_referrals_yesterday': 0,
        'legal_referrals_7d': 0,
        'pending_proposals': 0,
        'previews_yesterday': 0,
    }
    if not SUPABASE_URL:
        return out

    now_et = datetime.now(ZoneInfo('America/New_York'))
    today_et_date = now_et.date()
    yest_start = (now_et - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    yest_end = (now_et - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
    seven_days_ago = (now_et - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _count(url, params):
        try:
            r = requests.head(
                url,
                params=params,
                headers={**supabase_headers(), 'Prefer': 'count=exact'},
                timeout=TIMEOUT,
            )
            cr = r.headers.get('content-range', '')
            if '/' in cr:
                tail = cr.split('/')[-1]
                return int(tail) if tail.isdigit() else 0
        except Exception:
            return 0
        return 0

    try:
        out['legal_referrals_yesterday'] = _count(
            f"{SUPABASE_URL}/rest/v1/email_funnel",
            {
                'stage': 'eq.legal_referral_requested',
                'created_at': f'gte.{yest_start.isoformat()}',
                'and': f'(created_at.lte.{yest_end.isoformat()})',
            },
        )
        out['legal_referrals_7d'] = _count(
            f"{SUPABASE_URL}/rest/v1/email_funnel",
            {'stage': 'eq.legal_referral_requested', 'created_at': f'gte.{seven_days_ago.isoformat()}'},
        )
    except Exception as e:
        logger.warning(f'legal referral count failed: {e}')

    try:
        out['pending_proposals'] = _count(
            f"{SUPABASE_URL}/rest/v1/ad_proposals",
            {'status': 'eq.pending'},
        )
    except Exception as e:
        logger.warning(f'pending proposals count failed: {e}')

    try:
        out['previews_yesterday'] = _count(
            f"{SUPABASE_URL}/rest/v1/dmhoa_case_previews",
            {
                'created_at': f'gte.{yest_start.isoformat()}',
                'and': f'(created_at.lte.{yest_end.isoformat()})',
            },
        )
    except Exception as e:
        logger.warning(f'previews count failed: {e}')

    return out


def call_claude_sonnet(prompt, system_prompt='', max_retries=3):
    """Call Claude Sonnet with retry logic."""
    effective_system = (system_prompt or '') + HUMAN_VOICE_RULES
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': ANTHROPIC_API_KEY,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-sonnet-4-5-20250929',
                    'max_tokens': 2048,
                    'system': effective_system,
                    'messages': [{'role': 'user', 'content': prompt}]
                },
                timeout=(10, 60)
            )

            if response.status_code in [429, 529, 503]:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(f'Claude Sonnet {response.status_code}, retrying in {delay}s')
                    time.sleep(delay)
                    continue
                raise Exception(f'Claude Sonnet returned {response.status_code} after {max_retries} attempts')

            if not response.ok:
                raise Exception(f'Claude Sonnet error: {response.status_code} {response.text[:200]}')

            result = response.json()
            content = result.get('content', [])
            usage = result.get('usage', {})
            text = content[0]['text'] if content and content[0].get('type') == 'text' else ''
            _log_claude_usage('claude-sonnet-4-5-20250929', usage.get('input_tokens', 0), usage.get('output_tokens', 0), 'call_claude_sonnet')
            return text, usage

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise Exception('Claude Sonnet failed after all retries')


def call_claude_sonnet_chat(messages, system_prompt='', max_retries=3):
    """Call Claude Sonnet for multi-turn chat with full message history."""
    effective_system = (system_prompt or '') + HUMAN_VOICE_RULES
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': ANTHROPIC_API_KEY,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-sonnet-4-5-20250929',
                    'max_tokens': 2048,
                    'system': effective_system,
                    'messages': messages
                },
                timeout=(10, 60)
            )

            if response.status_code in [429, 529, 503]:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f'Claude Sonnet returned {response.status_code} after {max_retries} attempts')

            if not response.ok:
                raise Exception(f'Claude Sonnet error: {response.status_code} {response.text[:200]}')

            result = response.json()
            content = result.get('content', [])
            usage = result.get('usage', {})
            text = content[0]['text'] if content and content[0].get('type') == 'text' else ''
            _log_claude_usage('claude-sonnet-4-5-20250929', usage.get('input_tokens', 0), usage.get('output_tokens', 0), 'call_claude_sonnet_chat')
            return text, usage

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise Exception('Claude Sonnet chat failed after all retries')


# ============================================================================
# CHATBOT ENDPOINTS
# ============================================================================

@dashboard_bp.route('/api/dashboard/chat', methods=['POST', 'OPTIONS'])
def dashboard_chat():
    """In-dashboard Claude advisor chatbot."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Anthropic API key not configured'}), 500

    try:
        body = request.get_json()
        if not body or not body.get('message'):
            return jsonify({'error': 'Request body must include "message"'}), 400

        user_message = body['message']

        # 1. Load conversation history (last 20 messages)
        chat_history = []
        if SUPABASE_URL:
            hist_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_chat_history",
                params={'select': 'role,content', 'order': 'created_at.asc', 'limit': '20', 'offset': '0'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if hist_response.ok:
                chat_history = hist_response.json()

        # 2. Load document summaries
        doc_summaries = _fetch_doc_summaries()

        # 3. Build live data snapshot
        snapshot = _build_live_data_snapshot()

        # 4. Build system prompt
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        month_num = max(1, min(6, datetime.now().month - 2))  # March=1..August=6

        s = snapshot
        stripe_data = s.get('stripe', {})
        cases_data = s.get('cases', {})
        ads_data = s.get('google_ads', {})
        klaviyo_data = s.get('klaviyo', {})
        openai_data = s.get('openai', {})
        alerts_data = s.get('alerts', {})
        checklists_data = s.get('checklists', {})

        docs_section = ''
        doc_labels = {
            '6month_plan': '6-Month Operations Plan',
            'media_plan': 'Media Plan',
            'scenario': 'Scenario Planning (Good/Bad/Ugly)',
            'persona': 'User Persona Guide',
            'identity': 'Company Identity',
            'dev_system': 'Dev System Documentation',
        }
        for key, label in doc_labels.items():
            text = doc_summaries.get(key, 'Not available')
            docs_section += f'\n{label}:\n{text}\n'

        unavailable_note = ''
        if s.get('_sources_failed'):
            unavailable_note = f"\nNote: The following data sources were unavailable: {', '.join(s['_sources_failed'])}. Do not reference data from these sources.\n"

        system_prompt = f"""You are the DMHOA Business Advisor — a senior co-founder-level strategic advisor for Dispute My HOA, a $29 one-time-purchase educational platform that helps homeowners navigate HOA disputes.

Your role: Be direct, data-driven, and action-oriented. Reference specific numbers. Recommend concrete next steps. Challenge weak assumptions. Think like a technical founder running a bootstrapped SaaS with a $600/month marketing budget.

COMPANY DOCUMENTS:
{docs_section}

LIVE BUSINESS DATA (as of {current_time}):

Revenue:
- Today: ${stripe_data.get('revenue_today', 0)}
- This week: ${stripe_data.get('revenue_week', 0)}
- This month: ${stripe_data.get('revenue_month', 0)}
- Transactions this month: {stripe_data.get('transactions_month', 0)}

Cases:
- New today: {cases_data.get('new_today', 0)}
- Paid today: {cases_data.get('paid_today', 0)}
- Paid this month: {cases_data.get('paid_month', 0)}
- Conversion rate: {cases_data.get('conversion_rate', 0)}%

Google Ads:
- Spend today: ${ads_data.get('spend', 0)}
- Clicks today: {ads_data.get('clicks', 0)}
- CPA: ${ads_data.get('cpa', 0)}
- CTR: {round(ads_data.get('ctr', 0) * 100, 2) if ads_data.get('ctr') else 0}%

Email (Klaviyo):
- List size: {klaviyo_data.get('total_profiles', 0)}
- New today: {klaviyo_data.get('new_today', 0)}

Costs:
- OpenAI API today: ${openai_data.get('today_cost', 0)}
- OpenAI 7-day avg: ${openai_data.get('avg_daily_7d', 0)}

System Status:
- Active alerts: {alerts_data.get('critical', 0)} critical, {alerts_data.get('warning', 0)} warning, {alerts_data.get('info', 0)} info
- Checklist: {checklists_data.get('done', 0)}/{checklists_data.get('total', 0)} complete ({checklists_data.get('pct', 0)}%)
- Current month in plan: Month {month_num}
{unavailable_note}
IMPORTANT RULES:
- Always reference specific data points from the live data above
- If asked about strategy, reference the relevant strategic document
- If you recommend an action, make it specific and executable
- Be honest about what's working and what's not
- You are NOT a legal advisor — if asked about legal matters, note that DMHOA is an educational platform, not a law firm
- Keep responses concise but thorough. No fluff."""

        # 5. Build messages array
        messages = []
        for msg in chat_history:
            messages.append({'role': msg['role'], 'content': msg['content']})
        messages.append({'role': 'user', 'content': user_message})

        # 6. Call Claude Sonnet
        assistant_response, usage = call_claude_sonnet_chat(messages, system_prompt=system_prompt)

        # 7. Store both messages in history
        context_used = s.get('_sources_loaded', [])
        if SUPABASE_URL:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/dmhoa_chat_history",
                headers=supabase_headers(),
                json=[
                    {'role': 'user', 'content': user_message, 'context_used': {'sources': context_used}},
                    {'role': 'assistant', 'content': assistant_response},
                ],
                timeout=TIMEOUT
            )

        return jsonify({
            'response': assistant_response,
            'context_used': context_used,
            'tokens_used': {
                'input': usage.get('input_tokens', 0),
                'output': usage.get('output_tokens', 0),
            },
        })

    except requests.exceptions.Timeout:
        return jsonify({'error': 'The AI advisor took too long to respond. Please try again.'}), 504
    except Exception as e:
        logger.error(f'Chat endpoint error: {str(e)}')
        return jsonify({'error': 'Failed to process chat message'}), 500


@dashboard_bp.route('/api/dashboard/chat/history', methods=['GET', 'OPTIONS'])
def get_chat_history():
    """Get chat history."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        limit = min(int(request.args.get('limit', '50')), 100)
        offset = int(request.args.get('offset', '0'))

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_chat_history",
            params={'select': '*', 'order': 'created_at.desc', 'limit': str(limit), 'offset': str(offset)},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to fetch chat history: {response.text}')

        return jsonify(response.json())

    except Exception as e:
        logger.error(f'Chat history error: {str(e)}')
        return jsonify({'error': 'Failed to fetch chat history'}), 500


# ============================================================================
# DAILY SUMMARY ENDPOINTS
# ============================================================================

def _generate_daily_summary() -> Dict:
    """Generate today's daily summary using Claude Haiku."""
    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    snapshot = _build_live_data_snapshot()
    checklists = snapshot.get('checklists', {})

    data_text = json.dumps(snapshot, indent=2, default=str)

    # Generate structured JSON summary
    json_prompt = f"""You are a business operations assistant for DisputeMyHOA (DMHOA), a bootstrapped $29 one-time-purchase educational platform.

CONTEXT:
- The 6-month growth plan starts May 1, 2026. Phases: Month 1 May (Foundation & Launch), Month 2 June (Optimize & Amplify), Month 3 July (Content Engine), Month 4 August (Scale What Works), Month 5 September (Authority & Media), Month 6 October (Evaluate & Decide). Until May 1 we are in Pre-plan Ramp-Up (April 2026).
- This digest fires at 6am ET daily. All activity metrics in DATA reflect YESTERDAY (the completed day), not the current day. Do NOT report 0s as "today is slow" — they reflect a day that has not started yet.
- Active ad campaigns: "DMHOA-Search-M1-Disputes" with ad groups Appeal Intent, Long-Tail Specifics, Fight Intent (Dispute Process Intent is intentionally PAUSED). Only campaigns with actual ad spend are included in the ads data below.
- The "ctr_pct" field is CTR as a percentage (e.g., 5.65 means 5.65%). Use this for grading, NOT the raw "ctr" decimal.
- The "total_profiles" in Klaviyo represents email list size.
- "pending_proposals" in ops is current state — these are ad-analyzer proposals waiting on Eric's review.
- "email" data has two halves: "resend" = actual sends to inboxes (delivered/bounced/etc.) and "funnel" = our internal funnel stages from email_funnel table. Use these for email metrics. Klaviyo data is no longer reported.

Generate a daily business summary for {today} (reporting on yesterday's activity). Be concise, direct, and action-oriented.

DATA:
{data_text}

FORMAT YOUR RESPONSE AS A JSON OBJECT (all "yesterday" fields reflect the completed day, NOT today):
{{
  "executive_summary": "2-3 sentence overview framed as 'yesterday Eric got X / spent Y / etc.'",
  "revenue": {{
    "yesterday": number,
    "mtd": number,
    "target": 1000,
    "pace": "on_track" or "behind" or "ahead"
  }},
  "cases": {{
    "new_yesterday": number,
    "previews_yesterday": number,
    "paid_yesterday": number,
    "mtd_paid": number,
    "conversion_rate": number
  }},
  "ads": {{
    "spend_yesterday": number,
    "clicks_yesterday": number,
    "cpa": number,
    "ctr": number,
    "grade": "A" or "C" or "F"
  }},
  "email": {{
    "captured_yesterday": number (unique emails entering funnel yesterday),
    "delivered_yesterday": number (Resend delivered events yesterday),
    "bounce_rate_yesterday_pct": number,
    "nudges_sent_yesterday": number,
    "weekly_capture_total": number
  }},
  "ops": {{
    "legal_referrals_yesterday": number,
    "legal_referrals_7d": number,
    "pending_proposals": number
  }},
  "costs": {{
    "api_yesterday": number
  }},
  "alerts_active": number,
  "checklist_progress": {{
    "done": {checklists.get('done', 0)},
    "total": {checklists.get('total', 0)},
    "pct": {checklists.get('pct', 0)}
  }},
  "action_list": [
    "Concrete things Eric should do or check today, ordered by impact. Include any pending ad-analyzer proposals to review, legal-referral requests to follow up on, anomalies in numbers, or specific keywords/ads to inspect."
  ],
  "positive_notes": [
    "What is working — concrete, specific. Examples: ad group X improved CPA, conversion lift, milestone hit, new paying customer."
  ],
  "negative_notes": [
    "What is not working or worth concern — specific, with the exact metric or item. Examples: keyword X has 0 conv on $Y spent, conversion rate dropping, ad group impressions stalled."
  ]
}}

GRADING REFERENCE (use ctr_pct for CTR grading):
- Traffic A: 1,000+ visitors/mo | C: 250-999 | F: <250
- Ads CTR A: >3.5% | C: 2-3.5% | F: <2%  (compare against ctr_pct, NOT raw ctr decimal)
- Conversion A: >3% | C: 1-3% | F: <1%
- Revenue A: >$1,000/mo | C: $300-1,000 | F: <$300

Return ONLY the JSON object. No markdown, no code fences, no commentary."""

    raw_json = call_claude_haiku(json_prompt)
    cleaned = raw_json.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    summary_json = json.loads(cleaned.strip())

    # Generate markdown text summary
    text_prompt = f"""Generate a plain-English daily summary from this data. Format as markdown.

DATA: {json.dumps(summary_json, indent=2)}

Use these sections in this order. All daily figures are YESTERDAY's (completed day) — phrase them that way. Do NOT call yesterday's numbers "today's".
## Daily Summary — {today}
### Yesterday's Numbers
(Revenue yesterday + MTD, ad spend + clicks + CPA + CTR, new cases, new previews, paid cases, legal referrals — keep tight, one line each. Then on a separate line note current state: "pending ad proposals: N" since that's right-now state, not yesterday.)
### What's Working (positive notes)
(bulleted list of positive_notes verbatim, with one-line context if useful)
### What's Not Working (negative notes)
(bulleted list of negative_notes verbatim)
### Action List
(bulleted list of action_list verbatim — these are concrete things to do or check today)

Keep it under 350 words. Direct, no fluff. Written for a solo founder checking their phone first thing in the morning. No section headers other than the four above."""

    summary_text = call_claude_haiku(text_prompt)

    return {'date': today, 'summary_json': summary_json, 'summary_text': summary_text}


@dashboard_bp.route('/api/dashboard/daily-summary', methods=['GET', 'OPTIONS'])
def get_daily_summary():
    """Get or generate today's daily business summary."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

        # Check if today's summary already exists
        existing = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_daily_summaries",
            params={'select': '*', 'date': f'eq.{today}', 'limit': '1'},
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if existing.ok and existing.json():
            row = existing.json()[0]
            return jsonify({
                'date': today,
                'summary': row.get('summary_json'),
                'summary_text': row.get('summary_text'),
                'generated_at': row.get('created_at'),
            })

        # Generate new summary
        if not ANTHROPIC_API_KEY:
            return jsonify({'error': 'Anthropic API key not configured — cannot generate summary'}), 500

        result = _generate_daily_summary()

        # Save to Supabase
        requests.post(
            f"{SUPABASE_URL}/rest/v1/dmhoa_daily_summaries",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json={
                'date': today,
                'summary_json': result['summary_json'],
                'summary_text': result['summary_text'],
            },
            timeout=TIMEOUT
        )

        return jsonify({
            'date': today,
            'summary': result['summary_json'],
            'summary_text': result['summary_text'],
        })

    except json.JSONDecodeError as e:
        logger.error(f'Daily summary JSON parse error: {str(e)}')
        return jsonify({'error': 'Failed to parse AI-generated summary'}), 500
    except Exception as e:
        logger.error(f'Daily summary error: {str(e)}')
        return jsonify({'error': 'Failed to generate daily summary'}), 500


@dashboard_bp.route('/api/dashboard/daily-summary/send', methods=['POST', 'OPTIONS'])
def send_daily_summary():
    """Send today's daily summary via email."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not ADMIN_EMAIL:
        return jsonify({'error': 'ADMIN_EMAIL not configured'}), 500

    try:
        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

        # Get or generate today's summary
        summary_text = None
        if SUPABASE_URL:
            existing = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_daily_summaries",
                params={'select': 'summary_text,id', 'date': f'eq.{today}', 'limit': '1'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if existing.ok and existing.json():
                row = existing.json()[0]
                summary_text = row.get('summary_text')

        if not summary_text:
            if not ANTHROPIC_API_KEY:
                return jsonify({'error': 'No summary available and Anthropic API key not configured'}), 500
            result = _generate_daily_summary()
            summary_text = result['summary_text']

            if SUPABASE_URL:
                requests.post(
                    f"{SUPABASE_URL}/rest/v1/dmhoa_daily_summaries",
                    headers={**supabase_headers(), 'Prefer': 'return=representation'},
                    json={'date': today, 'summary_json': result['summary_json'], 'summary_text': summary_text},
                    timeout=TIMEOUT
                )

        # Send email
        if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
            return jsonify({'error': 'SMTP not configured'}), 500

        msg = MIMEMultipart()
        msg['From'] = SMTP_FROM
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f'DMHOA Daily Summary — {today}'
        msg.attach(MIMEText(summary_text, 'plain'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_FROM, [ADMIN_EMAIL], msg.as_string())

        # Mark as sent
        if SUPABASE_URL:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/dmhoa_daily_summaries?date=eq.{today}",
                headers=supabase_headers(),
                json={'sent': True, 'sent_at': datetime.now().isoformat()},
                timeout=TIMEOUT
            )

        return jsonify({'sent': True, 'to': ADMIN_EMAIL})

    except Exception as e:
        logger.error(f'Send daily summary error: {str(e)}')
        return jsonify({'error': 'Failed to send daily summary'}), 500


# ============================================================================
# SIX-MONTH PLAN ENDPOINT
# ============================================================================

PLAN_MONTHS = [
    {'month': 1, 'name': 'May 2026', 'theme': 'Foundation & Launch', 'budget_planned': 400},
    {'month': 2, 'name': 'June 2026', 'theme': 'Optimize & Amplify', 'budget_planned': 500},
    {'month': 3, 'name': 'July 2026', 'theme': 'Content Engine', 'budget_planned': 600},
    {'month': 4, 'name': 'August 2026', 'theme': 'Scale What Works', 'budget_planned': 750},
    {'month': 5, 'name': 'September 2026', 'theme': 'Authority & Media', 'budget_planned': 850},
    {'month': 6, 'name': 'October 2026', 'theme': 'Evaluate & Decide', 'budget_planned': 900},
]

# Pre-plan ramp-up data (April 2026) — kept for historical reference
RAMP_UP_PERIOD = {
    'name': 'April 2026',
    'theme': 'Pre-plan Ramp-Up',
    'budget_planned': 600,
}


def _fetch_posthog_metrics_for_plan() -> Dict:
    """Fetch PostHog visitor metrics since plan start date for grading."""
    result = {'unique_visitors': 0, 'total_sessions': 0, 'total_pageviews': 0}
    if not POSTHOG_PERSONAL_API_KEY or not POSTHOG_PROJECT_ID:
        return result

    try:
        posthog_headers = {
            'Authorization': f'Bearer {POSTHOG_PERSONAL_API_KEY}',
            'Content-Type': 'application/json',
        }

        # Query unique visitors and sessions since plan start
        query = {
            'query': {
                'kind': 'HogQLQuery',
                'query': f"""
                    SELECT
                        count(DISTINCT person_id) as unique_visitors,
                        count(DISTINCT $session_id) as total_sessions,
                        count(*) as total_pageviews
                    FROM events
                    WHERE event = '$pageview'
                      AND timestamp >= toDateTime('{PLAN_START_DATE}')
                """
            }
        }

        response = requests.post(
            f'{POSTHOG_API_URL}/api/projects/{POSTHOG_PROJECT_ID}/query/',
            headers=posthog_headers,
            json=query,
            timeout=TIMEOUT
        )

        if response.ok:
            rows = response.json().get('results', [])
            if rows and len(rows) > 0:
                row = rows[0]
                result = {
                    'unique_visitors': int(row[0] or 0),
                    'total_sessions': int(row[1] or 0),
                    'total_pageviews': int(row[2] or 0),
                }
        else:
            logger.warning(f'PostHog plan metrics query failed: {response.status_code}')
    except Exception as e:
        logger.error(f'_fetch_posthog_metrics_for_plan error: {str(e)}')

    return result

GRADING = {
    'traffic': {
        'monthly_visitors': {'A': 500, 'C': 100},
        'google_ads_ctr': {'A': 0.025, 'C': 0.015},
        'organic_share': {'A': 0.20, 'C': 0.05},
    },
    'conversion': {
        'site_conversion': {'A': 0.02, 'C': 0.005},
        'email_open_rate': {'A': 0.25, 'C': 0.15},
        'preview_to_paid': {'A': 0.03, 'C': 0.01},
    },
    'revenue': {
        'monthly_revenue': {'A': 200, 'C': 50},
        'cac_ltv_ratio': {'A': 3, 'C': 1},
        'roas': {'A': 2, 'C': 0.5},
    },
}


def _grade(value, thresholds):
    if value >= thresholds['A']:
        return 'A'
    elif value >= thresholds['C']:
        return 'C'
    return 'F'


@dashboard_bp.route('/api/dashboard/six-month', methods=['GET', 'OPTIONS'])
def get_six_month_plan():
    """Return the 6-month plan execution status with live grades."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        now = datetime.now()
        # Official plan starts May 2026: May=Month 1, June=Month 2, ..., October=Month 6
        if now.strftime('%Y-%m-%d') < PLAN_START_DATE:
            current_month = 1  # Pre-plan, still in ramp-up
        else:
            current_month = max(1, min(6, now.month - 4))  # May=1..October=6

        # Fetch live data — use plan_start date range for the full picture
        stripe_month = _fetch_stripe_metrics('month')
        cases_month = _fetch_supabase_case_metrics('month')
        ads = _fetch_google_ads_metrics('plan_start') or {}
        posthog = _fetch_posthog_metrics_for_plan()
        klaviyo = _fetch_klaviyo_metrics()

        monthly_revenue = stripe_month.get('revenue', 0)
        conversion_rate = cases_month.get('conversion_rate', 0) / 100  # convert % to decimal
        ads_ctr = ads.get('ctr', 0)
        ads_spend = ads.get('spend', 0)
        ads_conversions = ads.get('conversions', 0)

        # PostHog visitor data
        monthly_visitors = posthog.get('unique_visitors', 0)
        total_sessions = posthog.get('total_sessions', 0)
        total_pageviews = posthog.get('total_pageviews', 0)
        # Estimate organic share: sessions without paid ads / total sessions
        # For now, if we have visitors but low ad spend, organic is higher
        paid_sessions = ads.get('clicks', 0)
        organic_share = ((total_sessions - paid_sessions) / total_sessions) if total_sessions > 0 else 0
        organic_share = max(0, min(1, organic_share))  # clamp 0-1

        # Klaviyo email metrics
        email_subscribers = klaviyo.get('total_profiles', 0)
        # Estimate open rate from subscriber engagement (Klaviyo basic API doesn't expose open rates)
        # Use subscriber count as a proxy metric for the email_open_rate grade
        # If we have 50+ subscribers in the first month, that's a good sign
        # Map subscriber count to an estimated open rate for grading
        # Industry avg HOA email open rate ~25-35%
        estimated_open_rate = 0.25 if email_subscribers >= 50 else (0.18 if email_subscribers >= 20 else 0.10)

        roas = (monthly_revenue / ads_spend) if ads_spend > 0 else 0
        cac = (ads_spend / ads_conversions) if ads_conversions > 0 else 0
        ltv = 49  # single purchase product
        cac_ltv_ratio = (ltv / cac) if cac > 0 else 0

        # Compute grades with live data
        current_grades = {
            'traffic': {
                'monthly_visitors': {'value': monthly_visitors, 'grade': _grade(monthly_visitors, GRADING['traffic']['monthly_visitors'])},
                'google_ads_ctr': {'value': round(ads_ctr * 100, 2), 'grade': _grade(ads_ctr, GRADING['traffic']['google_ads_ctr'])},
                'organic_share': {'value': round(organic_share, 4), 'grade': _grade(organic_share, GRADING['traffic']['organic_share'])},
            },
            'conversion': {
                'site_conversion': {'value': round(conversion_rate, 4), 'grade': _grade(conversion_rate, GRADING['conversion']['site_conversion'])},
                'email_open_rate': {'value': round(estimated_open_rate, 4), 'grade': _grade(estimated_open_rate, GRADING['conversion']['email_open_rate'])},
                'preview_to_paid': {'value': round(conversion_rate, 4), 'grade': _grade(conversion_rate, GRADING['conversion']['preview_to_paid'])},
            },
            'revenue': {
                'monthly_revenue': {'value': monthly_revenue, 'grade': _grade(monthly_revenue, GRADING['revenue']['monthly_revenue'])},
                'cac_ltv_ratio': {'value': round(cac_ltv_ratio, 2), 'grade': _grade(cac_ltv_ratio, GRADING['revenue']['cac_ltv_ratio'])},
                'roas': {'value': round(roas, 2), 'grade': _grade(roas, GRADING['revenue']['roas'])},
            },
        }

        # Determine scenario
        all_grades = []
        f_by_category = {}
        for category, metrics in current_grades.items():
            f_count = 0
            for metric_name, metric_data in metrics.items():
                all_grades.append(metric_data['grade'])
                if metric_data['grade'] == 'F':
                    f_count += 1
            f_by_category[category] = f_count

        a_count = all_grades.count('A')
        c_count = all_grades.count('C')
        f_count_total = all_grades.count('F')

        if a_count > c_count and a_count > f_count_total:
            scenario = 'good'
            scenario_label = 'The Good — Profitable & Growing'
        elif f_count_total > a_count and f_count_total > c_count:
            scenario = 'ugly'
            scenario_label = 'The Ugly — No Revenue After Months of Effort'
        else:
            scenario = 'bad'
            scenario_label = 'The Bad — Break-Even but Stuck'

        pivot_triggered = any(v >= 2 for v in f_by_category.values())
        pivot_reason = ''
        if pivot_triggered:
            for cat, count in f_by_category.items():
                if count >= 2:
                    pivot_reason = f'2+ F grades in {cat.capitalize()} category'
                    break

        # Fetch checklist progress by month
        checklists_all = _fetch_checklist_progress()
        checklist_by_month = {}
        if SUPABASE_URL:
            try:
                cl_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/dmhoa_checklists",
                    params={'select': 'month,status'},
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )
                if cl_response.ok:
                    for item in cl_response.json():
                        m = item.get('month')
                        if m:
                            if m not in checklist_by_month:
                                checklist_by_month[m] = {'done': 0, 'total': 0}
                            checklist_by_month[m]['total'] += 1
                            if item.get('status') == 'done':
                                checklist_by_month[m]['done'] += 1
            except Exception:
                pass

        # --- Content actuals ---
        # Auto-count blog posts by month from blog_posts table
        # Plan months: 1=May 2026, 2=June 2026, ..., 6=October 2026
        # April 2026 = ramp-up period (kept separately)
        blog_by_month = {}
        ramp_up_blog_count = 0
        try:
            blog_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/blog_posts",
                params={
                    'select': 'published_at',
                    'published_at': 'gte.2026-04-01',
                    'order': 'published_at.asc',
                    'limit': '500'
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if blog_resp.ok:
                for bp in blog_resp.json():
                    pub = bp.get('published_at', '')
                    if pub:
                        try:
                            pub_date = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                            if pub_date.month == 4 and pub_date.year == 2026:
                                ramp_up_blog_count += 1
                            else:
                                # May=month1, June=month2, ..., October=month6
                                blog_month = pub_date.month - 4
                                if 1 <= blog_month <= 6:
                                    blog_by_month[blog_month] = blog_by_month.get(blog_month, 0) + 1
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f'Six-month plan - blog count failed: {e}')

        # Manual content counts from api_cache
        manual_content = _read_api_cache('plan_content_actuals') or {}

        months = []
        for pm in PLAN_MONTHS:
            m = pm['month']
            cl = checklist_by_month.get(m, {'done': 0, 'total': 0})
            status = 'active' if m == current_month else ('completed' if m < current_month else 'upcoming')
            m_manual = manual_content.get(str(m), {})
            months.append({
                **pm,
                'budget_actual': ads_spend if m == current_month else 0,
                'checklist_done': cl['done'],
                'checklist_total': cl['total'],
                'status': status,
                'content_actuals': {
                    'blog': blog_by_month.get(m, 0),
                    'video': m_manual.get('video', 0),
                    'newsletter': m_manual.get('newsletter', 0),
                    'social': m_manual.get('social', 0),
                },
            })

        total_done = checklists_all.get('done', 0)
        total_all = checklists_all.get('total', 0)
        overall_progress = round(total_done / total_all, 2) if total_all > 0 else 0

        # Pre-plan ramp-up summary (April 2026 — historical)
        ramp_up_manual = _read_api_cache('plan_ramp_up_actuals') or {}
        ramp_up = {
            **RAMP_UP_PERIOD,
            'budget_actual': 55.98,  # historical: ad spend during ramp-up
            'content_actuals': {
                'blog': ramp_up_blog_count,
                'video': ramp_up_manual.get('video', 0),
                'newsletter': ramp_up_manual.get('newsletter', 0),
                'social': ramp_up_manual.get('social', 0),
            },
        }

        return jsonify({
            'current_month': current_month,
            'months': months,
            'ramp_up': ramp_up,
            'current_grades': current_grades,
            'scenario': {
                'current': scenario,
                'label': scenario_label,
                'f_count_by_category': f_by_category,
                'pivot_triggered': pivot_triggered,
                'pivot_reason': pivot_reason,
            },
            'overall_progress': overall_progress,
        })

    except Exception as e:
        logger.error(f'Six-month plan error: {str(e)}')
        return jsonify({'error': 'Failed to generate six-month plan status'}), 500


@dashboard_bp.route('/api/dashboard/six-month/content', methods=['PATCH', 'OPTIONS'])
def update_content_actual():
    """Update manual content objective counts (video, newsletter, social)."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    body = request.get_json(silent=True) or {}
    month = body.get('month')
    content_type = body.get('type')
    count = body.get('count', 0)

    if not month or month not in range(1, 7):
        return jsonify({'error': 'Month must be 1-6'}), 400
    if content_type not in ('video', 'newsletter', 'social'):
        return jsonify({'error': 'Type must be video, newsletter, or social'}), 400
    if not isinstance(count, int) or count < 0:
        return jsonify({'error': 'Count must be a non-negative integer'}), 400

    try:
        # Read existing
        existing = _read_api_cache('plan_content_actuals') or {}
        month_key = str(month)
        if month_key not in existing:
            existing[month_key] = {}
        existing[month_key][content_type] = count

        # Write back
        requests.post(
            f"{SUPABASE_URL}/rest/v1/api_cache",
            headers={**supabase_headers(), 'Prefer': 'resolution=merge-duplicates'},
            json={
                'cache_key': 'plan_content_actuals',
                'data': existing,
                'updated_at': datetime.now().isoformat(),
            },
            timeout=TIMEOUT
        )

        return jsonify({'ok': True, 'month': month, 'type': content_type, 'count': count})

    except Exception as e:
        logger.error(f'Update content actual error: {str(e)}')
        return jsonify({'error': str(e)}), 500


# ============================================================================
# COSTS ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/costs', methods=['GET', 'OPTIONS'])
def get_costs():
    """Aggregated cost tracking across all services. Accepts ?period=today|week|month (default: month)."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        period = request.args.get('period', 'month')
        if period not in ('today', 'yesterday', 'week', 'month'):
            period = 'month'

        now = datetime.now()
        if period == 'yesterday':
            num_days = 1
            yesterday = now - timedelta(days=1)
            period_label = yesterday.strftime('%Y-%m-%d')
        elif period == 'today':
            num_days = 1
            period_label = now.strftime('%Y-%m-%d')
        elif period == 'week':
            num_days = 7
            start = now - timedelta(days=7)
            period_label = f'{start.strftime("%m/%d")} - {now.strftime("%m/%d")}'
        else:
            num_days = max(now.day, 1)
            period_label = now.strftime('%Y-%m')

        # Stripe revenue + fees
        stripe_data = _fetch_stripe_metrics(period)
        gross_revenue = stripe_data.get('revenue', 0)
        txn_count = stripe_data.get('transactions', 0)
        stripe_fees = round(txn_count * 0.30 + gross_revenue * 0.029, 2)
        net_revenue = round(gross_revenue - stripe_fees, 2)

        # Google Ads
        ads = _fetch_google_ads_metrics(period) or {}
        ads_spend = ads.get('spend', 0)

        # OpenAI
        openai = _fetch_openai_usage_metrics() or {}
        openai_today = openai.get('today_cost', 0)
        openai_avg = openai.get('avg_daily_7d', 0)
        if period in ('today', 'yesterday'):
            openai_spend = openai_today
        elif period == 'week':
            openai_spend = round(openai_avg * 7, 2)
        else:
            openai_spend = round(openai_avg * num_days, 2)

        # Claude / Anthropic
        claude = _fetch_claude_usage_metrics(period) or {}
        claude_spend = claude.get('mtd_cost', 0)
        claude_today = claude.get('today_cost', 0)

        # Fixed costs — pro-rate by period
        heroku_monthly = 7
        supabase_monthly = 0
        tools_monthly = 42  # domain, email, misc tools
        daily_fixed = (heroku_monthly + supabase_monthly + tools_monthly) / 30
        heroku_spend = round(heroku_monthly * num_days / 30, 2)
        supabase_spend = round(supabase_monthly * num_days / 30, 2)
        tools_spend = round(tools_monthly * num_days / 30, 2)

        total_costs = round(ads_spend + openai_spend + claude_spend + heroku_spend + supabase_spend + tools_spend + stripe_fees, 2)
        profit = round(net_revenue - total_costs + stripe_fees, 2)  # stripe fees already deducted from net
        margin_pct = round(profit / gross_revenue, 3) if gross_revenue > 0 else 0
        burn_rate_daily = round(total_costs / max(num_days, 1), 2)

        return jsonify({
            'period': period_label,
            'periodType': period,
            'revenue': {
                'gross': gross_revenue,
                'stripe_fees': stripe_fees,
                'net': net_revenue,
            },
            'costs': {
                'google_ads': {'mtd': ads_spend, 'daily_avg': round(ads_spend / max(num_days, 1), 2)},
                'openai_api': {'mtd': openai_spend, 'today': openai_today},
                'claude_api': {'mtd': claude_spend, 'today': claude_today, 'total_calls': claude.get('total_calls', 0)},
                'heroku': {'mtd': heroku_spend},
                'supabase': {'mtd': supabase_spend},
                'tools': {'mtd': tools_spend},
                'total': total_costs,
            },
            'margin': {
                'net_revenue': net_revenue,
                'total_costs': total_costs,
                'profit': profit,
                'margin_pct': margin_pct,
            },
            'burn_rate_daily': burn_rate_daily,
            'break_even': profit >= 0,
        })

    except Exception as e:
        logger.error(f'Costs endpoint error: {str(e)}')
        return jsonify({'error': 'Failed to calculate costs'}), 500


# ============================================================================
# FEATURE REQUESTS ENDPOINTS
# ============================================================================

@dashboard_bp.route('/api/dashboard/features', methods=['GET', 'POST', 'OPTIONS'])
def handle_features():
    """List or create feature requests."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL:
        return jsonify({'error': 'Supabase not configured'}), 500

    # GET
    if request.method == 'GET':
        try:
            params = {'select': '*', 'order': 'created_at.desc'}
            status = request.args.get('status')
            if status:
                params['status'] = f'eq.{status}'

            response = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_feature_requests",
                params=params,
                headers=supabase_headers(),
                timeout=TIMEOUT
            )

            if not response.ok:
                raise Exception(f'Failed to fetch features: {response.text}')

            items = response.json()
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            items.sort(key=lambda x: (priority_order.get(x.get('priority', 'low'), 2), x.get('created_at', '')))

            return jsonify(items)

        except Exception as e:
            logger.error(f'Features GET error: {str(e)}')
            return jsonify({'error': 'Failed to fetch feature requests'}), 500

    # POST
    try:
        body = request.get_json()
        if not body or not body.get('title'):
            return jsonify({'error': 'title is required'}), 400

        allowed = {'title', 'description', 'source', 'target_repo', 'estimated_effort', 'priority'}
        row = {k: v for k, v in body.items() if k in allowed}
        row['status'] = 'proposed'

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/dmhoa_feature_requests",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json=row,
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to create feature: {response.text}')

        items = response.json()
        return jsonify(items[0] if isinstance(items, list) and items else items), 201

    except Exception as e:
        logger.error(f'Features POST error: {str(e)}')
        return jsonify({'error': 'Failed to create feature request'}), 500


@dashboard_bp.route('/api/dashboard/features/<feature_id>', methods=['PATCH', 'OPTIONS'])
def update_feature(feature_id):
    """Update a feature request."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL:
        return jsonify({'error': 'Supabase not configured'}), 500

    try:
        body = request.get_json()
        if not body:
            return jsonify({'error': 'No update data provided'}), 400

        allowed = {'status', 'priority', 'title', 'description', 'target_repo', 'estimated_effort', 'implementation_prompt'}
        update = {k: v for k, v in body.items() if k in allowed}

        if 'status' in update:
            if update['status'] == 'accepted':
                update['accepted_at'] = datetime.now().isoformat()
            elif update['status'] == 'done':
                update['completed_at'] = datetime.now().isoformat()

        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/dmhoa_feature_requests?id=eq.{feature_id}",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json=update,
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Failed to update feature: {response.text}')

        items = response.json()
        if not items:
            return jsonify({'error': 'Feature request not found'}), 404

        return jsonify(items[0])

    except Exception as e:
        logger.error(f'Features PATCH error: {str(e)}')
        return jsonify({'error': 'Failed to update feature request'}), 500


@dashboard_bp.route('/api/dashboard/features/<feature_id>/prompt', methods=['POST', 'OPTIONS'])
def generate_feature_prompt(feature_id):
    """Generate a Claude Code implementation prompt for a feature request."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Supabase or Anthropic API not configured'}), 500

    try:
        # Load feature
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_feature_requests?id=eq.{feature_id}&select=*",
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok or not response.json():
            return jsonify({'error': 'Feature request not found'}), 404

        feature = response.json()[0]

        # Fetch technical documentation for richer context
        doc_summaries = _fetch_doc_summaries()
        docs_context = ''
        if doc_summaries:
            docs_context = '\n\nTECHNICAL DOCUMENTATION:\n'
            for key, summary in doc_summaries.items():
                docs_context += f'\n[{key}]\n{summary}\n'

        prompt = f"""You are a senior full-stack engineer writing implementation prompts for Claude Code.

DMHOA TECH STACK:
- Frontend: Vite 6.3.1 + Tailwind CSS 4.1.4 + Vanilla JS (no framework). Key files: index.html, start-case.html, case-workspace.html. JS classes: SimpleCaseWizard (simple-wizard.htm), CaseWorkspace (case-workspace.htm, ~40 methods), BlogAPI (services/blog-api.js). Hosted on Netlify.
- Backend: Python 3.11.7 + Flask 2.3+ + Gunicorn. Single app.py (~5,900 lines) + dashboard_routes.py Blueprint + statute_lookup.py. Supabase (PostgreSQL) via REST API. Hosted on Heroku.
- Dashboard: Angular 17.3.0 (standalone components) + Tailwind CSS 3.4.1 + TypeScript. 13 services, each calling /api/dashboard/*. Hosted on Netlify.
- Database: Supabase — tables: dmhoa_cases, dmhoa_documents, dmhoa_case_previews, dmhoa_case_outputs, dmhoa_messages, dmhoa_events, hoa_statutes
- AI: Claude Sonnet 4.6 (case analysis), Claude Haiku 4.5 (lightweight tasks), GPT-4o-mini (chat, previews, blog)
{docs_context}
FEATURE TO IMPLEMENT:
Title: {feature.get('title', '')}
Description: {feature.get('description', '')}
Target Repo: {feature.get('target_repo', 'unknown')}
Estimated Effort: {feature.get('estimated_effort', 'medium')}

Generate a complete, copy-pasteable Claude Code prompt that covers:
1. Exact files to create or modify (full paths)
2. Database changes (CREATE TABLE / ALTER TABLE SQL)
3. New API endpoints (method, path, request body, response shape)
4. Frontend/component changes
5. Step-by-step implementation order
6. What to test when done

Write it as a single prompt that starts with the instruction and includes all context. It should be immediately executable in Claude Code without any additional context."""

        result_text, usage = call_claude_sonnet(prompt)

        # Save prompt to feature
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/dmhoa_feature_requests?id=eq.{feature_id}",
            headers=supabase_headers(),
            json={'implementation_prompt': result_text},
            timeout=TIMEOUT
        )

        return jsonify({
            'prompt': result_text,
            'feature_id': feature_id,
            'tokens_used': {
                'input': usage.get('input_tokens', 0),
                'output': usage.get('output_tokens', 0),
            },
        })

    except Exception as e:
        logger.error(f'Feature prompt generation error: {str(e)}')
        return jsonify({'error': 'Failed to generate implementation prompt'}), 500


@dashboard_bp.route('/api/dashboard/features/suggestions', methods=['POST', 'OPTIONS'])
def generate_feature_suggestions():
    """Generate AI-powered feature suggestions from aggregated dashboard data."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Supabase or Anthropic API not configured'}), 500

    result_text = ''
    try:
        # 1. Gather live data snapshot (Stripe, Cases, Ads, Klaviyo, Alerts, Checklists)
        snapshot = _build_live_data_snapshot()

        # 2. Read PostHog cached data
        posthog_data = _read_api_cache('posthog_data') or {}

        # 3. Read Lighthouse cached data
        lighthouse_data = _read_api_cache('lighthouse_data') or {}

        # 4. Read latest legality scorecard
        scorecard_data = {}
        try:
            sc_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/legality_scorecard",
                params={
                    'status': 'eq.completed',
                    'order': 'analysis_date.desc',
                    'limit': '1',
                    'select': 'summary,categories,cases_analyzed'
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if sc_resp.ok and sc_resp.json():
                scorecard_data = sc_resp.json()[0]
        except Exception as e:
            logger.warning(f'Feature suggestions - scorecard fetch failed: {e}')

        # 5. Read recent HOA news
        news_data = []
        try:
            news_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/hoa_news_articles",
                params={
                    'select': 'title,category,priority',
                    'order': 'pub_date.desc',
                    'limit': '15'
                },
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if news_resp.ok:
                news_data = news_resp.json()
        except Exception as e:
            logger.warning(f'Feature suggestions - news fetch failed: {e}')

        # 6. Fetch existing features to avoid duplicates
        existing_features = []
        try:
            feat_resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_feature_requests",
                params={'select': 'title,status', 'order': 'created_at.desc'},
                headers=supabase_headers(),
                timeout=TIMEOUT
            )
            if feat_resp.ok:
                existing_features = feat_resp.json()
        except Exception as e:
            logger.warning(f'Feature suggestions - existing features fetch failed: {e}')

        # 7. Build aggregated context for Claude
        context = {
            'revenue': snapshot.get('stripe', {}),
            'cases': snapshot.get('cases', {}),
            'google_ads': snapshot.get('google_ads', {}),
            'email': snapshot.get('klaviyo', {}),
            'posthog': {
                'bounceRate': posthog_data.get('bounceRate', 0),
                'avgTimeOnPage': posthog_data.get('avgTimeOnPage', 0),
                'pagesPerSession': posthog_data.get('pagesPerSession', 0),
                'totalSessions': posthog_data.get('totalSessions', 0),
                'rageClicks': posthog_data.get('rageClicks', 0),
                'deadClicks': posthog_data.get('deadClicks', 0),
                'jsErrors': posthog_data.get('jsErrors', 0),
                'slowPageLoads': posthog_data.get('slowPageLoads', 0),
                'topPages': posthog_data.get('topPages', []),
                'pagesWithIssues': posthog_data.get('pagesWithIssues', []),
                'webVitals': posthog_data.get('webVitals', {}),
                'funnel': posthog_data.get('funnel', {}),
                'compositeGrade': posthog_data.get('compositeGrade', {}),
            },
            'lighthouse': {
                'performanceScore': lighthouse_data.get('performanceScore', 0),
                'seoScore': lighthouse_data.get('seoScore', 0),
                'accessibilityScore': lighthouse_data.get('accessibilityScore', 0),
                'bestPracticesScore': lighthouse_data.get('bestPracticesScore', 0),
                'lcp': lighthouse_data.get('lcp', {}),
                'fcp': lighthouse_data.get('fcp', {}),
                'cls': lighthouse_data.get('cls', {}),
                'tbt': lighthouse_data.get('tbt', {}),
            },
            'legality_scorecard': scorecard_data,
            'recent_news': [{'title': n.get('title', ''), 'category': n.get('category', ''), 'priority': n.get('priority', '')} for n in news_data[:15]],
        }

        existing_titles = [f.get('title', '') for f in existing_features]

        # 8. Build and send Claude prompt
        system_prompt = 'You are a product strategist for DisputeMyHOA, a $29 self-service SaaS that helps homeowners respond to HOA violation notices. You analyze real business data and suggest actionable feature improvements. Respond with valid JSON only, no markdown code fences.'

        prompt = f"""Analyze the following real-time business data for DisputeMyHOA and suggest 5-8 feature improvements.

BUSINESS DATA:
{json.dumps(context, indent=2, default=str)}

EXISTING FEATURES (do NOT duplicate these):
{json.dumps(existing_titles, indent=2)}

For each suggestion, provide:
- "title": concise feature title (max 80 chars)
- "description": 2-3 sentence explanation of what to build and why
- "category": one of "homepage", "conversion", "performance", "content", "marketing", "ux", "product"
- "target_repo": one of "frontend", "backend", "dashboard"
- "estimated_effort": one of "small", "medium", "large"
- "priority": one of "high", "medium", "low" (based on potential impact)
- "data_basis": brief explanation of which specific metrics informed this suggestion (e.g. "bounce rate 72%, avg time on page 15s")

Focus areas (prioritize top to bottom):
1. HOMEPAGE ENGAGEMENT: If bounce rate > 50% or avg time on page < 30s, suggest specific homepage changes to increase visitor stay time (hero copy, social proof, interactive elements, above-the-fold optimization)
2. CONVERSION FUNNEL: If there are significant drop-offs between landing -> preview -> purchase, suggest fixes for the weakest stage
3. PERFORMANCE: If any Lighthouse score < 80 or web vitals exceed thresholds (LCP > 2500ms, CLS > 0.1, FCP > 1800ms), suggest specific technical fixes
4. MARKETING: Based on ad CPA, email list growth, and keyword performance, suggest campaign improvements
5. PRODUCT: Based on violation type patterns from the legality scorecard, suggest new features or content for the most common case types
6. CONTENT: Based on HOA news trends, suggest blog topics or educational content that could drive organic traffic

Respond with this exact JSON structure:
{{"suggestions": [...]}}"""

        result_text, usage = call_claude_sonnet(prompt, system_prompt)

        # Parse the response
        cleaned = result_text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(cleaned)
        suggestions = parsed.get('suggestions', [])

        # Track data sources that were available
        data_sources = snapshot.get('_sources_loaded', [])
        if posthog_data:
            data_sources.append('posthog')
        if lighthouse_data:
            data_sources.append('lighthouse')
        if scorecard_data:
            data_sources.append('legality_scorecard')
        if news_data:
            data_sources.append('hoa_news')

        return jsonify({
            'suggestions': suggestions,
            'data_sources': data_sources,
            'tokens_used': {
                'input': usage.get('input_tokens', 0),
                'output': usage.get('output_tokens', 0),
            },
        })

    except json.JSONDecodeError as e:
        logger.error(f'Feature suggestions JSON parse error: {str(e)}')
        return jsonify({'error': 'Failed to parse AI suggestions'}), 500
    except Exception as e:
        logger.error(f'Feature suggestions error: {str(e)}')
        return jsonify({'error': 'Failed to generate feature suggestions'}), 500


# ============================================================================
# COMMAND CENTER ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/command-center', methods=['GET', 'OPTIONS'])
def get_command_center():
    """Aggregated overview for the hub page. Cached for 60 seconds."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        now = time.time()

        # Return cached data if fresh (< 60 seconds)
        if _command_center_cache['data'] and (now - _command_center_cache['ts']) < 60:
            return jsonify(_command_center_cache['data'])

        # Build fresh data
        stripe_today = _fetch_stripe_metrics('today')
        stripe_week = _fetch_stripe_metrics('week')
        stripe_month = _fetch_stripe_metrics('month')

        cases_today = _fetch_supabase_case_metrics('today')
        cases_7d = _fetch_supabase_case_metrics('week')

        alerts = _fetch_alert_counts()
        checklists = _fetch_checklist_progress()
        ads = _fetch_google_ads_metrics() or {}
        # Replaced klaviyo with funnel-derived weekly capture count
        funnel_week = _fetch_email_funnel_metrics('week')

        # Quick site health check
        site_up = True
        site_ms = 0
        try:
            start_t = time.time()
            site_resp = requests.get('https://disputemyhoa.com', timeout=5)
            site_ms = round((time.time() - start_t) * 1000)
            site_up = site_resp.status_code == 200
        except Exception:
            site_up = False

        # Get daily summary preview
        summary_preview = ''
        if SUPABASE_URL:
            try:
                today_str = datetime.now().strftime('%Y-%m-%d')
                sum_resp = requests.get(
                    f"{SUPABASE_URL}/rest/v1/dmhoa_daily_summaries",
                    params={'select': 'summary_json', 'date': f'eq.{today_str}', 'limit': '1'},
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )
                if sum_resp.ok and sum_resp.json():
                    sj = sum_resp.json()[0].get('summary_json') or {}
                    summary_preview = sj.get('executive_summary', '')
            except Exception:
                pass

        # Get last scan time
        last_scan = ''
        if SUPABASE_URL:
            try:
                scan_resp = requests.get(
                    f"{SUPABASE_URL}/rest/v1/dmhoa_alerts",
                    params={'select': 'created_at', 'order': 'created_at.desc', 'limit': '1'},
                    headers=supabase_headers(),
                    timeout=TIMEOUT
                )
                if scan_resp.ok and scan_resp.json():
                    last_scan = scan_resp.json()[0].get('created_at', '')
            except Exception:
                pass

        data = {
            'revenue': {
                'today': stripe_today.get('revenue', 0),
                'week': stripe_week.get('revenue', 0),
                'month': stripe_month.get('revenue', 0),
            },
            'cases': {
                'new_today': cases_today.get('new_cases', 0),
                'paid_today': cases_today.get('paid_cases', 0),
                'pending_analysis': 0,
                'conversion_rate_7d': cases_7d.get('conversion_rate', 0),
            },
            'alerts': {
                'critical': alerts.get('critical', 0),
                'warning': alerts.get('warning', 0),
                'info': alerts.get('info', 0),
                'total_unacknowledged': alerts.get('total', 0),
            },
            'checklists': {
                'done': checklists.get('done', 0),
                'total': checklists.get('total', 0),
                'pct': checklists.get('pct', 0) / 100 if checklists.get('pct', 0) > 1 else checklists.get('pct', 0),
                'top_pending': checklists.get('top_pending', []),
            },
            'health': {
                'site_up': site_up,
                'site_response_ms': site_ms,
                'last_scan': last_scan,
            },
            'quick_stats': {
                # 7-day unique-email capture count from the email_funnel table.
                # Field name preserved to avoid breaking the existing dashboard
                # binding; semantics changed from "Klaviyo profile count" to
                # "weekly captured emails" (a more directly actionable number).
                'klaviyo_list_size': funnel_week.get('total_unique', 0),
                'ads_spend_today': ads.get('spend', 0),
                'ads_cpa_today': ads.get('cpa', 0),
            },
            'daily_summary_preview': summary_preview,
        }

        # Update cache
        _command_center_cache['data'] = data
        _command_center_cache['ts'] = now

        return jsonify(data)

    except Exception as e:
        logger.error(f'Command center error: {str(e)}')
        return jsonify({'error': 'Failed to load command center data'}), 500


# ============================================================================
# DAILY GARDEN — "Watering the Garden" recurring daily growth tasks
# ============================================================================

@dashboard_bp.route('/api/dashboard/email-funnel-metrics', methods=['GET', 'OPTIONS'])
def email_funnel_metrics():
    """Return aggregate funnel metrics from the email_funnel table.
    Shows stage counts, conversion rates, and nudge effectiveness."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        # Fetch all funnel rows
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/email_funnel",
            headers=supabase_headers(),
            params={'select': '*', 'limit': '5000'},
            timeout=TIMEOUT,
        )
        rows = resp.json() if resp.ok else []

        total = len(rows)
        if total == 0:
            return jsonify({
                'total': 0,
                'stages': {'quick_preview_complete': 0, 'full_preview_viewed': 0, 'purchased': 0},
                'nudges': {'nudge_1_sent': 0, 'nudge_2_sent': 0, 'nudge_3_sent': 0},
                'conversion_rates': {'quick_to_full': 0, 'full_to_purchased': 0, 'overall': 0},
                'purchased_count': 0,
                'revenue_estimate': 0,
            })

        # Stage counts (current stage, not cumulative)
        stage_counts = {'quick_preview_complete': 0, 'full_preview_viewed': 0, 'purchased': 0}
        nudge_counts = {'nudge_1_sent': 0, 'nudge_2_sent': 0, 'nudge_3_sent': 0}
        purchased_count = 0

        # "Reached" counts (cumulative: if you're at purchased, you also reached full and quick)
        reached_quick = 0
        reached_full = 0
        reached_purchased = 0

        stage_rank = {'quick_preview_complete': 1, 'full_preview_viewed': 2, 'purchased': 3}

        for row in rows:
            stage = row.get('stage', '')
            rank = stage_rank.get(stage, 0)

            # Current stage count
            if stage in stage_counts:
                stage_counts[stage] += 1

            # Cumulative "reached" counts
            if rank >= 1:
                reached_quick += 1
            if rank >= 2:
                reached_full += 1
            if rank >= 3:
                reached_purchased += 1

            # Nudge counts
            if row.get('nudge_1_sent'):
                nudge_counts['nudge_1_sent'] += 1
            if row.get('nudge_2_sent'):
                nudge_counts['nudge_2_sent'] += 1
            if row.get('nudge_3_sent'):
                nudge_counts['nudge_3_sent'] += 1

            if row.get('purchased'):
                purchased_count += 1

        # Conversion rates (percentage of people who reached the next stage)
        quick_to_full = round((reached_full / reached_quick) * 100, 1) if reached_quick > 0 else 0
        full_to_purchased = round((reached_purchased / reached_full) * 100, 1) if reached_full > 0 else 0
        overall = round((reached_purchased / reached_quick) * 100, 1) if reached_quick > 0 else 0

        return jsonify({
            'total': total,
            'stages': stage_counts,
            'reached': {
                'quick_preview': reached_quick,
                'full_preview': reached_full,
                'purchased': reached_purchased,
            },
            'nudges': nudge_counts,
            'conversion_rates': {
                'quick_to_full': quick_to_full,
                'full_to_purchased': full_to_purchased,
                'overall': overall,
            },
            'purchased_count': purchased_count,
            'revenue_estimate': purchased_count * 29,
        })

    except Exception as e:
        logger.error(f'email_funnel_metrics error: {e}')
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/api/dashboard/daily-garden', methods=['GET', 'OPTIONS'])
def get_daily_garden():
    """Return today's garden task completion state."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/daily_garden",
            headers=supabase_headers(),
            params={
                'task_date': f'eq.{today}',
                'select': '*',
            },
            timeout=TIMEOUT,
        )
        rows = resp.json() if resp.ok else []
        completions = {r['task_key']: r.get('completed', False) for r in rows}
        return jsonify({'date': today, 'completions': completions})
    except Exception as e:
        logger.error(f'daily_garden GET error: {e}')
        return jsonify({'date': today, 'completions': {}, 'error': str(e)})


@dashboard_bp.route('/api/dashboard/daily-garden/<task_key>', methods=['PATCH', 'OPTIONS'])
def toggle_daily_garden(task_key):
    """Toggle a garden task's completion for today."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    body = request.get_json(silent=True) or {}
    completed = bool(body.get('completed', False))
    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    now_iso = datetime.now().isoformat()

    payload = {
        'task_date': today,
        'task_key': task_key,
        'completed': completed,
        'completed_at': now_iso if completed else None,
    }

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/daily_garden",
            headers={**supabase_headers(), 'Prefer': 'resolution=merge-duplicates,return=representation'},
            json=payload,
            timeout=TIMEOUT,
        )
        if resp.ok:
            rows = resp.json() or []
            return jsonify({'ok': True, 'row': rows[0] if rows else payload})
        return jsonify({'ok': False, 'error': resp.text[:300]}), 500
    except Exception as e:
        logger.error(f'daily_garden PATCH error: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500
