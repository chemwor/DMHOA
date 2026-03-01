# Dashboard API Routes - Migrated from Netlify Functions
# Flask routes for the DMHOA Dashboard analytics and management endpoints

import os
import json
import logging
import math
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps

import requests
from flask import Blueprint, request, jsonify

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
dashboard_bp = Blueprint('dashboard', __name__)

# Configuration from environment
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Klaviyo Configuration
KLAVIYO_API_KEY = os.environ.get('KLAVIYO_API_KEY')
KLAVIYO_FULL_PREVIEW_LIST_ID = os.environ.get('KLAVIYO_FULL_PREVIEW_LIST_ID', 'T6LY99')
KLAVIYO_QUICK_PREVIEW_LIST_ID = os.environ.get('KLAVIYO_QUICK_PREVIEW_LIST_ID', 'QS6zfC')

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

    if period == 'today':
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

    if period == 'today':
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


PLAN_START_DATE = '2026-02-25'

def get_google_ads_date_range(period: str) -> Dict[str, str]:
    """Get date range in YYYY-MM-DD format for Google Ads API."""
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')

    if period == 'today':
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
# KLAVIYO ENDPOINT
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

        flow_stats = [
            {
                'name': f.get('attributes', {}).get('name', 'Unknown Flow'),
                'status': f.get('attributes', {}).get('status', 'unknown'),
            }
            for f in flows[:5]
        ]

        total_emails_in_flow = full_preview_count + quick_preview_count

        return jsonify({
            'totalProfiles': total_profiles,
            'totalEmailsInFlow': total_emails_in_flow,
            'fullPreviewEmails': full_preview_count,
            'quickPreviewEmails': quick_preview_count,
            'emailsCollectedToday': emails_collected_today,
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

        return jsonify({
            'dailySpend': round(total_spend, 2),
            'clicks': total_clicks,
            'impressions': total_impressions,
            'cpc': round(total_spend / total_clicks, 2) if total_clicks > 0 else 0,
            'ctr': round((total_clicks / total_impressions) * 100, 2) if total_impressions > 0 else 0,
            'conversions': round(total_conversions, 2),
            'costPerConversion': round(total_spend / total_conversions, 2) if total_conversions > 0 else 0,
            'campaigns': campaigns,
            'keywords': [],  # Can be expanded
            'searchTerms': [],  # Can be expanded
            'ads': [],  # Can be expanded
            'targetCampaign': 'DMHOA - DIY Response - Phrase - March',
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
            'totalSessions': 0,
            'totalPageViews': 0,
            'pagesPerSession': 0,
            'avgScrollDepth': 0,
            'avgTimeOnPage': 0,
            'bounceRate': 0,
            'totalVisits': 0,
            'uniqueVisitors': 0,
            'returningVisitors': 0,
            'rageClicks': 0,
            'deadClicks': 0,
            'quickBacks': 0,
            'excessiveScrolling': 0,
            'jsErrors': 0,
            'slowPageLoads': 0,
            'isMockData': True,
            'message': 'PostHog not configured.',
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

        # Query 1: Sessions — total sessions, unique users, avg duration, pageviews, pages/session, bounce rate
        sessions_query = {
            'query': {
                'kind': 'HogQLQuery',
                'query': """
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
                """
            }
        }

        sessions_response = requests.post(
            f'{POSTHOG_API_URL}/api/projects/{POSTHOG_PROJECT_ID}/query/',
            headers=posthog_headers,
            json=sessions_query,
            timeout=TIMEOUT
        )

        sessions_data = {'total_sessions': 0, 'unique_users': 0, 'avg_session_duration': 0,
                         'total_pageviews': 0, 'pages_per_session': 0, 'bounce_rate': 0}

        if sessions_response.ok:
            result = sessions_response.json()
            rows = result.get('results', [])
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
        else:
            logger.warning(f'PostHog sessions query failed: {sessions_response.status_code} {sessions_response.text}')

        # Query 2: Rage clicks (last 3 days)
        rage_query = {
            'query': {
                'kind': 'HogQLQuery',
                'query': """
                    SELECT count(*) as rage_clicks
                    FROM events
                    WHERE event = '$rageclick'
                      AND timestamp > now() - INTERVAL 3 DAY
                """
            }
        }

        rage_response = requests.post(
            f'{POSTHOG_API_URL}/api/projects/{POSTHOG_PROJECT_ID}/query/',
            headers=posthog_headers,
            json=rage_query,
            timeout=TIMEOUT
        )

        rage_clicks = 0
        if rage_response.ok:
            result = rage_response.json()
            rows = result.get('results', [])
            if rows and len(rows) > 0:
                rage_clicks = int(rows[0][0] or 0)

        # Query 3: Dead clicks (last 3 days)
        dead_query = {
            'query': {
                'kind': 'HogQLQuery',
                'query': """
                    SELECT count(*) as dead_clicks
                    FROM events
                    WHERE event = '$dead_click'
                      AND timestamp > now() - INTERVAL 3 DAY
                """
            }
        }

        dead_response = requests.post(
            f'{POSTHOG_API_URL}/api/projects/{POSTHOG_PROJECT_ID}/query/',
            headers=posthog_headers,
            json=dead_query,
            timeout=TIMEOUT
        )

        dead_clicks = 0
        if dead_response.ok:
            result = dead_response.json()
            rows = result.get('results', [])
            if rows and len(rows) > 0:
                dead_clicks = int(rows[0][0] or 0)

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
            'jsErrors': 0,
            'slowPageLoads': 0,
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

        response = requests.get(PAGESPEED_API_URL, params=params, timeout=(10, 60))

        if not response.ok:
            raise Exception(f'PageSpeed API error: {response.status_code}')

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
            'error': 'Failed to fetch Lighthouse data',
            'message': str(e),
            'isMockData': True,
        }), 500


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

HOA_QUERIES = ['HOA homeowners association news', 'HOA legislation law']


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


@dashboard_bp.route('/api/dashboard/hoa-news', methods=['GET', 'PATCH', 'OPTIONS'])
def handle_hoa_news():
    """Get or update HOA news articles."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    # Handle PATCH for updating article status
    if request.method == 'PATCH':
        try:
            data = request.get_json() or {}
            article_id = data.get('articleId')
            action = data.get('action')

            if not article_id or not action:
                return jsonify({'error': 'Missing articleId or action'}), 400

            update_data = {}
            if action == 'bookmark':
                update_data = {'bookmarked': True}
            elif action == 'unbookmark':
                update_data = {'bookmarked': False}
            elif action == 'dismiss':
                update_data = {'dismissed': True}
            elif action == 'undismiss':
                update_data = {'dismissed': False}
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

        # Fetch articles from database
        params = {
            'select': '*',
            'order': 'pub_date.desc.nullsfirst',
            'limit': '100'
        }
        if not include_dismissed:
            params['dismissed'] = 'eq.false'

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
                'timestamp': a.get('pub_date') or a.get('created_at'),
                'bookmarked': a.get('bookmarked', False),
                'usedForContent': a.get('used_for_content', False),
                'dismissed': a.get('dismissed', False),
                'firstSeenAt': a.get('first_seen_at'),
                'lastSeenAt': a.get('last_seen_at'),
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


def call_claude_api(prompt: str, system_prompt: str, max_tokens: int = 4096) -> str:
    """Call Claude API."""
    response = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
        },
        json={
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': max_tokens,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=(10, 120)
    )

    if not response.ok:
        raise Exception(f'Claude API error: {response.status_code} - {response.text}')

    data = response.json()
    usage = data.get('usage', {})
    _log_claude_usage(
        model='claude-sonnet-4-20250514',
        input_tokens=usage.get('input_tokens', 0),
        output_tokens=usage.get('output_tokens', 0),
        endpoint='sonnet'
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

                # Generate ideas using Claude
                article_summaries = '\n'.join([
                    f'- "{a["title"]}" [{a["category"]}]: {a.get("description", "No description")}'
                    for a in articles
                ])

                system_prompt = 'You are an expert content strategist for an HOA dispute resolution platform. Generate unique blog post ideas that help homeowners understand their rights and navigate HOA issues. Respond with valid JSON only.'

                prompt = f'''Based on these HOA news articles, generate 3-5 unique blog post ideas:

ARTICLES:
{article_summaries}

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
            state = payload.get('state') or payload.get('hoaState') or 'Unknown'

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

        system_prompt = 'You are an HOA legal tech analyst. Respond with valid JSON only, no markdown.'

        prompt = f'''Analyze this HOA dispute platform data and provide insights.

VIOLATIONS (top 8): {json.dumps([{"type": t, **d} for t, d in top_types])}

CONVERSION: Total={len(cases)}, Paid={total_paid}, Rate={round(total_paid/len(cases)*100, 1) if cases else 0}%

TOP STATES: {", ".join([f"{s}({d['count']})" for s, d in top_states])}

Respond with this JSON:
{{"trends_summary":{{"most_common_violations":[],"highest_converting_cases":[]}},"conversion_analysis":{{"overall_rate":0,"best_performing_segment":"","improvement_opportunities":[]}},"feature_suggestions":[],"risk_assessment":{{"categories":[],"highest_risk_category":"","most_profitable_category":""}},"strategic_recommendations":[],"executive_summary":""}}

Fill in real values based on the data. Be concise.'''

        try:
            analysis_text = call_claude_api(prompt, system_prompt, 4096)
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
                'overallConversionRate': round(total_paid / len(cases) * 100, 1) if cases else 0,
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
        logger.error(f'Legality scorecard error: {str(e)}')
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


# ============================================================================
# AD SUGGESTIONS ENDPOINT
# ============================================================================

# In-memory job store for ad suggestions
ad_suggestion_jobs = {}


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

        if not start_date or not end_date:
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
                # Fetch data and generate suggestions (simplified)
                system_prompt = '''You are a Google Ads optimization expert for DisputeMyHOA, a $49 self-service SaaS tool.
Focus on DIY homeowners, not attorney-seekers. Provide actionable suggestions in JSON format.'''

                prompt = f'''Analyze Google Ads performance for {start_date} to {end_date}.
Provide optimization suggestions in JSON format with these fields:
- periodInsights: date range, spend, revenue
- performanceSummary: brief analysis
- keywordSuggestions: array of keyword recommendations
- negativeKeywordSuggestions: array of keywords to exclude
- adCopySuggestions: array of ad copy improvements
- generalRecommendations: array of strategic recommendations'''

                response_text = call_claude_api(prompt, system_prompt, 4096)
                cleaned = response_text.replace('```json', '').replace('```', '').strip()
                suggestions = json.loads(cleaned)

                result = {
                    **suggestions,
                    'generatedAt': datetime.now().isoformat(),
                    'dateRange': {'startDate': start_date, 'endDate': end_date},
                }

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
                    'system': system_prompt,
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
        data = request.get_json()
        if not data or not data.get('documents'):
            return jsonify({'error': 'Request body must include "documents" mapping doc_key to document text'}), 400

        documents = data['documents']

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
            prompt = f"""Extract every actionable task from this document. For each task, return a JSON object with:
- "title": Short task name (max 80 chars)
- "description": What specifically needs to be done (1-2 sentences)
- "category": One of ["google_ads", "content_seo", "social", "media", "product", "email", "ops", "finance", "legal"]
- "month": Integer 1-6 if the task is time-bound to a specific month in the plan, null if it's evergreen
- "priority": "high", "medium", or "low" based on how much the document emphasizes it
- "due_date": "YYYY-MM-DD" if a specific date is mentioned or can be inferred, null otherwise

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

        return jsonify(response.json())

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


def _fetch_claude_usage_metrics() -> Optional[Dict]:
    """Fetch MTD Claude usage from Supabase dmhoa_claude_usage table."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None

    try:
        # Get first day of current month
        now = datetime.now()
        month_start = now.replace(day=1).strftime('%Y-%m-%dT00:00:00')

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_claude_usage",
            params={
                'select': 'model,input_tokens,output_tokens,cost',
                'created_at': f'gte.{month_start}',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )

        if not response.ok:
            logger.warning(f'Claude usage table query failed: {response.status_code}')
            return None

        rows = response.json()
        total_cost = 0
        total_input = 0
        total_output = 0
        calls = len(rows)
        today_cost = 0
        today_str = now.strftime('%Y-%m-%d')

        for row in rows:
            total_cost += row.get('cost', 0)
            total_input += row.get('input_tokens', 0)
            total_output += row.get('output_tokens', 0)

        # Get today's cost separately
        today_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_claude_usage",
            params={
                'select': 'cost',
                'created_at': f'gte.{today_str}T00:00:00',
            },
            headers=supabase_headers(),
            timeout=TIMEOUT
        )
        if today_response.ok:
            today_rows = today_response.json()
            today_cost = sum(r.get('cost', 0) for r in today_rows)

        days_in_month = max(now.day, 1)
        avg_daily = total_cost / days_in_month if days_in_month > 0 else 0

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


@dashboard_bp.route('/api/dashboard/alerts/scan', methods=['POST', 'OPTIONS'])
def run_alert_scan():
    """Run all health and performance checks, create alerts for triggered conditions."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

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
        paid_cases_response = requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_cases",
            params={
                'select': 'id,token,created_at',
                'status': 'eq.paid',
                'created_at': f'gte.{one_hour_ago_iso}',
                'created_at': f'lte.{ten_min_ago}',
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

    return jsonify({
        'scan_completed': True,
        'alerts_created': len(alerts_created),
        'timestamp': datetime.utcnow().isoformat(),
    })


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


def _build_live_data_snapshot() -> Dict:
    """Build aggregated live data for the chatbot and daily summary. Tolerates individual source failures."""
    sources_loaded = []
    sources_failed = []
    data = {}

    # Stripe
    try:
        stripe_today = _fetch_stripe_metrics('today')
        stripe_week = _fetch_stripe_metrics('week')
        stripe_month = _fetch_stripe_metrics('month')
        data['stripe'] = {
            'revenue_today': stripe_today['revenue'],
            'revenue_week': stripe_week['revenue'],
            'revenue_month': stripe_month['revenue'],
            'transactions_today': stripe_today['transactions'],
            'transactions_month': stripe_month['transactions'],
        }
        sources_loaded.append('stripe')
    except Exception as e:
        logger.error(f'Snapshot - Stripe failed: {str(e)}')
        sources_failed.append('stripe')
        data['stripe'] = {}

    # Supabase cases
    try:
        cases_today = _fetch_supabase_case_metrics('today')
        cases_month = _fetch_supabase_case_metrics('month')
        data['cases'] = {
            'new_today': cases_today['new_cases'],
            'paid_today': cases_today['paid_cases'],
            'paid_month': cases_month['paid_cases'],
            'conversion_rate': cases_month['conversion_rate'],
        }
        sources_loaded.append('supabase')
    except Exception as e:
        logger.error(f'Snapshot - Supabase failed: {str(e)}')
        sources_failed.append('supabase')
        data['cases'] = {}

    # Google Ads — today's data, only campaigns with spend
    try:
        ads = _fetch_google_ads_metrics('today')
        if ads:
            # Convert CTR to percentage for clarity (raw is decimal like 0.0565)
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

    # Klaviyo
    try:
        klaviyo = _fetch_klaviyo_metrics()
        data['klaviyo'] = klaviyo
        sources_loaded.append('klaviyo')
    except Exception as e:
        logger.error(f'Snapshot - Klaviyo failed: {str(e)}')
        sources_failed.append('klaviyo')
        data['klaviyo'] = {}

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

    data['_sources_loaded'] = sources_loaded
    data['_sources_failed'] = sources_failed
    return data


def call_claude_sonnet(prompt, system_prompt='', max_retries=3):
    """Call Claude Sonnet with retry logic."""
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
                    'system': system_prompt,
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
            return text, usage

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise Exception('Claude Sonnet failed after all retries')


def call_claude_sonnet_chat(messages, system_prompt='', max_retries=3):
    """Call Claude Sonnet for multi-turn chat with full message history."""
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
                    'system': system_prompt,
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

        system_prompt = f"""You are the DMHOA Business Advisor — a senior co-founder-level strategic advisor for Dispute My HOA, a $49 one-time-purchase educational platform that helps homeowners navigate HOA disputes.

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
    today = datetime.now().strftime('%Y-%m-%d')
    snapshot = _build_live_data_snapshot()
    checklists = snapshot.get('checklists', {})

    data_text = json.dumps(snapshot, indent=2, default=str)

    # Generate structured JSON summary
    json_prompt = f"""You are a business operations assistant for DisputeMyHOA (DMHOA), a bootstrapped $49 one-time-purchase educational platform.

CONTEXT:
- The 6-month growth plan started February 25, 2026. We are currently in Month 1 (Foundation & Launch).
- Target campaign: "DMHOA - DIY Response - Phrase - March"
- Only campaigns with actual ad spend are included in the ads data below.
- The "ctr_pct" field is CTR as a percentage (e.g., 5.65 means 5.65%). Use this for grading, NOT the raw "ctr" decimal.
- The "total_profiles" in Klaviyo represents email list size.

Generate a daily business summary for {today}. Be concise, direct, and action-oriented.

DATA:
{data_text}

FORMAT YOUR RESPONSE AS A JSON OBJECT:
{{
  "executive_summary": "2-3 sentence overview of today's business state",
  "revenue": {{
    "today": number,
    "mtd": number,
    "target": 1000,
    "pace": "on_track" or "behind" or "ahead"
  }},
  "cases": {{
    "new_today": number,
    "paid_today": number,
    "mtd_paid": number,
    "conversion_rate": number
  }},
  "ads": {{
    "spend_today": number,
    "clicks": number,
    "cpa": number,
    "ctr": number,
    "grade": "A" or "C" or "F"
  }},
  "email": {{
    "list_size": number,
    "new_subscribers": number
  }},
  "costs": {{
    "api_today": number
  }},
  "alerts_active": number,
  "checklist_progress": {{
    "done": {checklists.get('done', 0)},
    "total": {checklists.get('total', 0)},
    "pct": {checklists.get('pct', 0)}
  }},
  "top_3_actions": [
    "Most impactful thing to do today",
    "Second most impactful",
    "Third most impactful"
  ],
  "risks": ["Any risks or concerns worth noting"],
  "wins": ["Anything positive to highlight"]
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

Use these sections:
## Daily Summary — {today}
### Revenue
### Cases
### Ads
### Email
### Costs
### Top 3 Actions Today
### Risks
### Wins

Keep it to ~300 words. Direct, no fluff. Written for a solo founder checking their phone in the morning."""

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
        today = datetime.now().strftime('%Y-%m-%d')

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
        today = datetime.now().strftime('%Y-%m-%d')

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
    {'month': 1, 'name': 'March 2026', 'theme': 'Foundation & Launch', 'budget_planned': 600},
    {'month': 2, 'name': 'April 2026', 'theme': 'Growth & Optimization', 'budget_planned': 600},
    {'month': 3, 'name': 'May 2026', 'theme': 'Content & SEO Push', 'budget_planned': 600},
    {'month': 4, 'name': 'June 2026', 'theme': 'Scale What Works', 'budget_planned': 750},
    {'month': 5, 'name': 'July 2026', 'theme': 'Expansion & Partnerships', 'budget_planned': 800},
    {'month': 6, 'name': 'August 2026', 'theme': 'Sustainability & Review', 'budget_planned': 900},
]


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
        'monthly_visitors': {'A': 1000, 'C': 250},
        'google_ads_ctr': {'A': 0.035, 'C': 0.02},
        'organic_share': {'A': 0.30, 'C': 0.10},
    },
    'conversion': {
        'site_conversion': {'A': 0.03, 'C': 0.01},
        'email_open_rate': {'A': 0.25, 'C': 0.15},
        'preview_to_paid': {'A': 0.05, 'C': 0.02},
    },
    'revenue': {
        'monthly_revenue': {'A': 1000, 'C': 300},
        'cac_ltv_ratio': {'A': 3, 'C': 1},
        'roas': {'A': 3, 'C': 1},
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
        # Plan started Feb 25, 2026: Feb=Month 1, Mar=Month 1, Apr=Month 2, etc.
        if now.strftime('%Y-%m-%d') < '2026-03-01':
            current_month = 1  # Still in the ramp-up period of Month 1
        else:
            current_month = max(1, min(6, now.month - 2))  # March=1..August=6

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

        months = []
        for pm in PLAN_MONTHS:
            m = pm['month']
            cl = checklist_by_month.get(m, {'done': 0, 'total': 0})
            status = 'active' if m == current_month else ('completed' if m < current_month else 'upcoming')
            months.append({
                **pm,
                'budget_actual': ads_spend if m == current_month else 0,
                'checklist_done': cl['done'],
                'checklist_total': cl['total'],
                'status': status,
            })

        total_done = checklists_all.get('done', 0)
        total_all = checklists_all.get('total', 0)
        overall_progress = round(total_done / total_all, 2) if total_all > 0 else 0

        return jsonify({
            'current_month': current_month,
            'months': months,
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


# ============================================================================
# COSTS ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/costs', methods=['GET', 'OPTIONS'])
def get_costs():
    """Aggregated cost tracking across all services."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        period = datetime.now().strftime('%Y-%m')

        # Stripe revenue + fees
        stripe_month = _fetch_stripe_metrics('month')
        gross_revenue = stripe_month.get('revenue', 0)
        txn_count = stripe_month.get('transactions', 0)
        stripe_fees = round(txn_count * 0.30 + gross_revenue * 0.029, 2)
        net_revenue = round(gross_revenue - stripe_fees, 2)

        # Google Ads — month to date
        ads = _fetch_google_ads_metrics('month') or {}
        ads_mtd = ads.get('spend', 0)

        # OpenAI
        openai = _fetch_openai_usage_metrics() or {}
        openai_today = openai.get('today_cost', 0)
        openai_avg = openai.get('avg_daily_7d', 0)
        days_in_month = datetime.now().day
        openai_mtd = round(openai_avg * days_in_month, 2)

        # Claude / Anthropic
        claude = _fetch_claude_usage_metrics() or {}
        claude_mtd = claude.get('mtd_cost', 0)
        claude_today = claude.get('today_cost', 0)

        # Fixed costs (estimates)
        heroku_mtd = 7
        supabase_mtd = 0
        tools_mtd = 42  # domain, email, misc tools

        total_costs = round(ads_mtd + openai_mtd + claude_mtd + heroku_mtd + supabase_mtd + tools_mtd + stripe_fees, 2)
        profit = round(net_revenue - total_costs + stripe_fees, 2)  # stripe fees already deducted from net
        margin_pct = round(profit / gross_revenue, 3) if gross_revenue > 0 else 0
        burn_rate_daily = round(total_costs / max(days_in_month, 1), 2)

        return jsonify({
            'period': period,
            'revenue': {
                'gross': gross_revenue,
                'stripe_fees': stripe_fees,
                'net': net_revenue,
            },
            'costs': {
                'google_ads': {'mtd': ads_mtd, 'daily_avg': round(ads_mtd / max(days_in_month, 1), 2)},
                'openai_api': {'mtd': openai_mtd, 'today': openai_today},
                'claude_api': {'mtd': claude_mtd, 'today': claude_today, 'total_calls': claude.get('total_calls', 0)},
                'heroku': {'mtd': heroku_mtd},
                'supabase': {'mtd': supabase_mtd},
                'tools': {'mtd': tools_mtd},
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

        prompt = f"""You are a senior full-stack engineer writing implementation prompts for Claude Code.

DMHOA TECH STACK:
- Frontend: Vite 6.3.1 + Tailwind CSS 4.1.4 + Vanilla JS (no framework). Key files: index.html, start-case.html, case-workspace.html. JS classes: SimpleCaseWizard (simple-wizard.htm), CaseWorkspace (case-workspace.htm, ~40 methods), BlogAPI (services/blog-api.js). Hosted on Netlify.
- Backend: Python 3.11.7 + Flask 2.3+ + Gunicorn. Single app.py (~5,900 lines) + dashboard_routes.py Blueprint + statute_lookup.py. Supabase (PostgreSQL) via REST API. Hosted on Heroku.
- Dashboard: Angular 17.3.0 (standalone components) + Tailwind CSS 3.4.1 + TypeScript. 13 services, each calling /api/dashboard/*. Hosted on Netlify.
- Database: Supabase — tables: dmhoa_cases, dmhoa_documents, dmhoa_case_previews, dmhoa_case_outputs, dmhoa_messages, dmhoa_events, hoa_statutes
- AI: Claude Sonnet 4.6 (case analysis), Claude Haiku 4.5 (lightweight tasks), GPT-4o-mini (chat, previews, blog)

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
        klaviyo = _fetch_klaviyo_metrics()

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
                'klaviyo_list_size': klaviyo.get('total_profiles', 0),
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
