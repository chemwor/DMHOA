# Dashboard API Routes - Migrated from Netlify Functions
# Flask routes for the DMHOA Dashboard analytics and management endpoints

import os
import json
import logging
import math
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

# Clarity Configuration
CLARITY_API_TOKEN = os.environ.get('CLARITY_API_TOKEN')
CLARITY_API_URL = 'https://www.clarity.ms/export-data/api/v1/project-live-insights'

# Lighthouse Configuration
GOOGLE_PAGESPEED_API_KEY = os.environ.get('GOOGLE_PAGESPEED_API_KEY')
PAGESPEED_API_URL = 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed'
TARGET_URL = 'https://disputemyhoa.com/'

# OpenAI Configuration
OPENAI_ADMIN_KEY = os.environ.get('OPENAI_ADMIN_KEY') or os.environ.get('OPENAI_API_KEY')
OPENAI_MONTHLY_BUDGET = float(os.environ.get('OPENAI_MONTHLY_BUDGET', 0))

# Unsplash Configuration
UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY')

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

            is_quick_preview = payload.get('completionPhase') == 'simple'
            is_full_preview = not is_quick_preview and payload.get('pastedText')
            is_purchase = case.get('unlocked') or case.get('stripe_payment_intent_id')

            case_output = outputs_map.get(case.get('token'))
            has_output = case_output is not None

            if is_purchase:
                purchases += 1
                if case.get('amount_total'):
                    total_revenue += case['amount_total'] / 100

            if is_quick_preview:
                quick_previews += 1
            elif is_full_preview or payload.get('issueText'):
                full_previews += 1

            if len(recent_cases) < 10:
                recent_cases.append({
                    'id': case.get('id'),
                    'token': case.get('token'),
                    'email': payload.get('email') or case.get('email'),
                    'created_at': case.get('created_at'),
                    'status': case.get('status'),
                    'type': 'quick' if is_quick_preview else 'full',
                    'unlocked': case.get('unlocked', False),
                    'noticeType': payload.get('noticeType'),
                    'issueText': (payload.get('issueText') or '')[:100] + '...' if payload.get('issueText') else None,
                    'amount': case.get('amount_total', 0) / 100 if case.get('amount_total') else None,
                    'hasOutput': has_output,
                    'outputStatus': case_output.get('status') if case_output else None,
                })

            if is_purchase and len(completed_cases) < 20:
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

        # Get total profiles
        response = requests.get(
            'https://a.klaviyo.com/api/profiles/?page[size]=1',
            headers=klaviyo_headers(),
            timeout=TIMEOUT
        )
        total_profiles = 0
        if response.ok:
            data = response.json()
            total_profiles = data.get('meta', {}).get('page_info', {}).get('total', 0)

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
            'targetCampaign': 'DMHOA Initial Test',
            'isMockData': True,
            'message': 'Google Ads not configured.',
        })

    try:
        period = request.args.get('period', 'today')
        date_range = get_google_ads_date_range(period)

        access_token = get_google_ads_access_token()
        if not access_token:
            raise Exception('Failed to get access token')

        # Query campaign performance
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

            spend = (metrics.get('costMicros', 0) or 0) / 1_000_000
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
            'targetCampaign': 'DMHOA Initial Test',
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
# CLARITY ENDPOINT
# ============================================================================

@dashboard_bp.route('/api/dashboard/clarity', methods=['GET', 'OPTIONS'])
def get_clarity_data():
    """Get Microsoft Clarity analytics data for dashboard."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not CLARITY_API_TOKEN:
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
            'message': 'Clarity not configured.',
        })

    try:
        # Check cache first
        force_refresh = request.args.get('refresh') == 'true'
        cache_key = 'clarity_data'

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
                    if cache_age < 4 * 60 * 60:  # 4 hours
                        return jsonify({
                            **cache_data[0]['data'],
                            'fromCache': True,
                            'cacheAge': f'{int(cache_age / 60)} minutes',
                        })

        # Fetch from Clarity API
        response = requests.get(
            f'{CLARITY_API_URL}?numOfDays=3',
            headers={
                'Authorization': f'Bearer {CLARITY_API_TOKEN}',
                'Content-Type': 'application/json',
            },
            timeout=TIMEOUT
        )

        if not response.ok:
            raise Exception(f'Clarity API error: {response.status_code}')

        api_data = response.json()

        # Parse metrics
        metrics = {
            'totalSessions': 0,
            'uniqueVisitors': 0,
            'pagesPerSession': 0,
            'avgScrollDepth': 0,
            'avgTimeOnPage': 0,
            'rageClicks': 0,
            'deadClicks': 0,
            'quickBacks': 0,
            'excessiveScrolling': 0,
            'jsErrors': 0,
        }

        for item in api_data if isinstance(api_data, list) else []:
            metric_name = item.get('metricName')
            info = item.get('information', [])

            if metric_name == 'Traffic':
                for entry in info:
                    metrics['totalSessions'] += int(entry.get('totalSessionCount', 0))
                    metrics['uniqueVisitors'] = max(metrics['uniqueVisitors'], int(entry.get('distantUserCount', 0)))
            elif metric_name == 'Scroll Depth':
                scroll_values = [float(e.get('avgScrollDepth', 0)) for e in info if e.get('avgScrollDepth')]
                if scroll_values:
                    metrics['avgScrollDepth'] = round(sum(scroll_values) / len(scroll_values))
            elif metric_name == 'Dead Click Count':
                metrics['deadClicks'] = sum(int(e.get('count', 0)) for e in info)
            elif metric_name == 'Rage Click Count':
                metrics['rageClicks'] = sum(int(e.get('count', 0)) for e in info)

        data = {
            **metrics,
            'totalPageViews': round(metrics['totalSessions'] * max(metrics['pagesPerSession'], 1)),
            'bounceRate': 0,
            'totalVisits': metrics['totalSessions'],
            'returningVisitors': 0,
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
        logger.error(f'Clarity API error: {str(e)}')
        return jsonify({
            'error': 'Failed to fetch Clarity data',
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
                    'unlocked': 'eq.true',
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
        total_unlocked = 0
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
                by_type[notice_type] = {'count': 0, 'unlocked': 0, 'revenue': 0}
            by_type[notice_type]['count'] += 1

            if state not in by_state:
                by_state[state] = {'count': 0, 'unlocked': 0, 'revenue': 0}
            by_state[state]['count'] += 1

            if case.get('unlocked'):
                total_unlocked += 1
                by_type[notice_type]['unlocked'] += 1
                by_state[state]['unlocked'] += 1
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

CONVERSION: Total={len(cases)}, Unlocked={total_unlocked}, Rate={round(total_unlocked/len(cases)*100, 1) if cases else 0}%

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
                'unlockedCases': total_unlocked,
                'overallConversionRate': round(total_unlocked / len(cases) * 100, 1) if cases else 0,
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
