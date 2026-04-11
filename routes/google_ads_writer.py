"""
Google Ads write/automation routes.

Provides:
- POST /api/dashboard/google-ads/launch-m1     create the full M1 search campaign
- POST /api/dashboard/google-ads/analyze       run analyzer, generate proposals
- GET  /api/dashboard/google-ads/proposals     list proposals
- POST /api/dashboard/google-ads/proposals/<id>/approve  apply a proposal
- POST /api/dashboard/google-ads/proposals/<id>/reject   dismiss a proposal

The mutate functions use the same OAuth/refresh-token credentials as the
existing read endpoints in dashboard_routes.py.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any

import requests
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

google_ads_writer_bp = Blueprint('google_ads_writer', __name__)

# --- Credentials & config (same env vars as the read endpoints) ---

GOOGLE_ADS_DEVELOPER_TOKEN = os.environ.get('GOOGLE_ADS_DEVELOPER_TOKEN')
GOOGLE_ADS_CUSTOMER_ID = os.environ.get('GOOGLE_ADS_CUSTOMER_ID')
GOOGLE_ADS_LOGIN_CUSTOMER_ID = os.environ.get('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
GOOGLE_ADS_CLIENT_ID = os.environ.get('GOOGLE_ADS_CLIENT_ID')
GOOGLE_ADS_CLIENT_SECRET = os.environ.get('GOOGLE_ADS_CLIENT_SECRET')
GOOGLE_ADS_REFRESH_TOKEN = os.environ.get('GOOGLE_ADS_REFRESH_TOKEN')
GOOGLE_ADS_API_VERSION = 'v21'
GOOGLE_ADS_API_BASE = f'https://googleads.googleapis.com/{GOOGLE_ADS_API_VERSION}'

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
TIMEOUT = (5, 60)

# Site host used for landing page URLs
SITE_URL = os.environ.get('SITE_URL', 'https://disputemyhoa.com')

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

# Same writing style rules as the rest of the system
HUMAN_VOICE_RULES = """

WRITING STYLE RULES (critical, must follow):
- Never use em-dashes (—) or en-dashes (–). Use periods, commas, colons, or parentheses instead.
- Never use these words/phrases: delve, leverage, robust, seamlessly, comprehensive, holistic, empower, streamline, cutting-edge, state-of-the-art, embark, harness, tapestry, vibrant, transformative, paramount, pivotal, moreover, furthermore, in essence, it is worth noting, in conclusion, ultimately, navigate the complexities, in today's, in the realm of.
- Do not start sentences with "Indeed", "Notably", "Importantly", or "However,".
- Write plain, direct, conversational English. Short sentences. No throat-clearing.
- Sound like a real person wrote this, not like a press release."""


def supabase_headers():
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json',
    }


def _get_access_token() -> Optional[str]:
    if not all([GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN]):
        return None
    try:
        r = requests.post(
            'https://oauth2.googleapis.com/token',
            data={
                'client_id': GOOGLE_ADS_CLIENT_ID,
                'client_secret': GOOGLE_ADS_CLIENT_SECRET,
                'refresh_token': GOOGLE_ADS_REFRESH_TOKEN,
                'grant_type': 'refresh_token',
            },
            timeout=TIMEOUT,
        )
        if r.ok:
            return r.json().get('access_token')
        return None
    except Exception as e:
        logger.error(f'Google Ads token refresh failed: {e}')
        return None


def _ads_headers(token: str) -> Dict[str, str]:
    h = {
        'Authorization': f'Bearer {token}',
        'developer-token': GOOGLE_ADS_DEVELOPER_TOKEN,
        'Content-Type': 'application/json',
    }
    if GOOGLE_ADS_LOGIN_CUSTOMER_ID:
        h['login-customer-id'] = GOOGLE_ADS_LOGIN_CUSTOMER_ID
    return h


def _query(token: str, query: str) -> List[Dict]:
    """Execute a GAQL query against the operating customer."""
    url = f'{GOOGLE_ADS_API_BASE}/customers/{GOOGLE_ADS_CUSTOMER_ID}/googleAds:search'
    r = requests.post(url, headers=_ads_headers(token), json={'query': query}, timeout=TIMEOUT)
    if not r.ok:
        raise Exception(f'GAQL query failed: {r.status_code} - {r.text}')
    return r.json().get('results', [])


def _mutate(token: str, resource: str, operations: List[Dict]) -> Dict:
    """Generic mutate helper. resource is e.g. 'campaigns', 'adGroups', 'campaignBudgets'."""
    url = f'{GOOGLE_ADS_API_BASE}/customers/{GOOGLE_ADS_CUSTOMER_ID}/{resource}:mutate'
    r = requests.post(url, headers=_ads_headers(token), json={'operations': operations}, timeout=TIMEOUT)
    if not r.ok:
        raise Exception(f'Mutate {resource} failed: {r.status_code} - {r.text}')
    return r.json()


# --- M1 campaign blueprint ---

M1_CAMPAIGN_NAME = 'DMHOA-Search-M1-Disputes'
M1_BUDGET_DAILY_USD = 25  # daily cap
M1_DEFAULT_CPC_USD = 5  # default max cpc; can override per ad group

# Convert dollars to micros (Google Ads uses micros: 1 USD = 1_000_000)
def _to_micros(usd: float) -> int:
    return int(round(usd * 1_000_000))


# All ad copy is UPL-compliant: no claims of legal advice, representation,
# guaranteed outcomes, or "winning". Frames the product as a self-help
# document preparation tool, not legal counsel.
M1_AD_GROUPS = [
    {
        'name': 'Appeal Intent',
        'final_url_path': '/appeal-hoa-fine',
        'max_cpc_usd': 7,
        'keywords': [
            'appeal hoa fine',
            'appeal hoa violation',
            'hoa appeal letter',
            'how to appeal hoa',
        ],
        'rsa': {
            'headlines': [
                'HOA Appeal Letter Tool',
                'Draft Your HOA Appeal',
                'The Dispute Engine',
                'Self-Help HOA Disputes',
                'Free HOA Notice Review',
                'Prepare Your Appeal',
                'HOA Letter Drafting Tool',
                'Understand Your HOA Notice',
            ],
            'descriptions': [
                'The Dispute Engine drafts your HOA appeal letter. Free notice review. Not a law firm.',
                'Self-help document prep for HOA disputes. Free preview, full letter from $49.',
                'Educational HOA info and Engine drafting. Not a law firm. Not legal advice.',
                'Get help organizing and responding to your HOA notice. Free review available.',
            ],
        },
    },
    {
        'name': 'Fight Intent',
        'final_url_path': '/fight-hoa-violation',
        'max_cpc_usd': 7,
        'keywords': [
            'fight hoa fine',
            'fight hoa violation',
            'unfair hoa fine',
            'hoa fine unfair',
            'refuse to pay hoa fine',
        ],
        'rsa': {
            'headlines': [
                'Respond to HOA Violations',
                'Unfair HOA Notice?',
                'HOA Response Letter Tool',
                "Don't Just Pay It",
                'The Dispute Engine',
                'Draft Your Own Response',
                'Free HOA Letter Review',
                'HOA Violation Help',
            ],
            'descriptions': [
                'Self-help tool to draft your own response to an HOA violation notice. Free review.',
                'Engine-assisted document preparation for HOA disputes. Not a law firm. From $49.',
                'Get organized and informed before you reply to your HOA. Free notice analysis.',
                'Educational tool that helps you understand and respond to HOA notices. From $49.',
            ],
        },
    },
    {
        'name': 'Dispute Process Intent',
        'final_url_path': '/dispute-hoa',
        'max_cpc_usd': 5,
        'keywords': [
            'how to dispute hoa fine',
            'how to dispute hoa',
            'hoa dispute resolution',
            'is my hoa fine legal',
            'can hoa fine me for',
        ],
        'rsa': {
            'headlines': [
                'How to Dispute an HOA Fine',
                'Understand HOA Notices',
                'HOA Dispute Process',
                '3-Step HOA Response Tool',
                'Free HOA Notice Review',
                'The Dispute Engine',
                'Know Your HOA Notice',
                'Disputing HOA Fines',
            ],
            'descriptions': [
                'Step-by-step educational guide to HOA disputes. Free notice review.',
                'Learn the HOA dispute process. The Dispute Engine is a self-help tool. Not a law firm.',
                'Educational resource and document drafting tool for HOA disputes. Free preview.',
                'Understand your HOA notice before responding. Free preview, full report from $49.',
            ],
        },
    },
]

NEGATIVE_KEYWORDS = [
    'hoa management',
    'hoa software',
    'hoa job',
    'hoa career',
    'hoa accounting',
    'hoa insurance',
    'free template',
    'reddit',
    'what is hoa',
    'hoa president',
]


# --- M1 launch logic ---

def _create_m1_campaign(token: str) -> Dict:
    """Create the full M1 campaign structure. Returns a summary dict."""
    summary: Dict[str, Any] = {
        'budget_resource': None,
        'campaign_resource': None,
        'ad_groups': [],
        'keywords_added': 0,
        'ads_added': 0,
        'negatives_added': 0,
    }

    # 1. Create the budget
    budget_op = {
        'create': {
            'name': f'{M1_CAMPAIGN_NAME} Budget',
            'amountMicros': str(_to_micros(M1_BUDGET_DAILY_USD)),
            'deliveryMethod': 'STANDARD',
            'explicitlyShared': False,
        }
    }
    budget_resp = _mutate(token, 'campaignBudgets', [budget_op])
    budget_resource = budget_resp['results'][0]['resourceName']
    summary['budget_resource'] = budget_resource

    # 2. Create the campaign (PAUSED initially for safety)
    campaign_op = {
        'create': {
            'name': M1_CAMPAIGN_NAME,
            'advertisingChannelType': 'SEARCH',
            'status': 'PAUSED',
            'manualCpc': {'enhancedCpcEnabled': False},
            'campaignBudget': budget_resource,
            'networkSettings': {
                'targetGoogleSearch': True,
                'targetSearchNetwork': False,
                'targetContentNetwork': False,
                'targetPartnerSearchNetwork': False,
            },
            # Required by Google Ads API v21 for EU compliance.
            # DMHOA does not run political ads, so this is always false.
            'containsEuPoliticalAdvertising': 'DOES_NOT_CONTAIN_EU_POLITICAL_ADVERTISING',
        }
    }
    campaign_resp = _mutate(token, 'campaigns', [campaign_op])
    campaign_resource = campaign_resp['results'][0]['resourceName']
    summary['campaign_resource'] = campaign_resource

    # 3. Target US only via campaign criterion (geo target ID 2840 = United States)
    geo_op = {
        'create': {
            'campaign': campaign_resource,
            'location': {'geoTargetConstant': 'geoTargetConstants/2840'},
        }
    }
    try:
        _mutate(token, 'campaignCriteria', [geo_op])
    except Exception as e:
        logger.warning(f'Geo targeting setup failed (non-fatal): {e}')

    # 4. Add negative keywords at the campaign level
    neg_ops = []
    for neg in NEGATIVE_KEYWORDS:
        neg_ops.append({
            'create': {
                'campaign': campaign_resource,
                'negative': True,
                'keyword': {'text': neg, 'matchType': 'BROAD'},
            }
        })
    if neg_ops:
        try:
            _mutate(token, 'campaignCriteria', neg_ops)
            summary['negatives_added'] = len(neg_ops)
        except Exception as e:
            logger.warning(f'Negative keywords setup failed: {e}')

    # 5. Create each ad group, its keywords, and an RSA
    for ag in M1_AD_GROUPS:
        ag_op = {
            'create': {
                'name': ag['name'],
                'campaign': campaign_resource,
                'status': 'ENABLED',
                'type': 'SEARCH_STANDARD',
                'cpcBidMicros': str(_to_micros(ag['max_cpc_usd'])),
            }
        }
        ag_resp = _mutate(token, 'adGroups', [ag_op])
        ag_resource = ag_resp['results'][0]['resourceName']

        # Keywords as ad group criteria
        kw_ops = []
        for kw in ag['keywords']:
            kw_ops.append({
                'create': {
                    'adGroup': ag_resource,
                    'status': 'ENABLED',
                    'keyword': {'text': kw, 'matchType': 'EXACT'},
                }
            })
        if kw_ops:
            _mutate(token, 'adGroupCriteria', kw_ops)
            summary['keywords_added'] += len(kw_ops)

        # Responsive search ad - validate character limits before sending
        final_url = f"{SITE_URL}{ag['final_url_path']}"
        for h in ag['rsa']['headlines']:
            if len(h) > 30:
                logger.error(f"Headline too long ({len(h)} > 30): {h!r}")
                summary.setdefault('errors', []).append(f"{ag['name']}: headline too long ({len(h)} chars): {h!r}")
        for d in ag['rsa']['descriptions']:
            if len(d) > 90:
                logger.error(f"Description too long ({len(d)} > 90): {d!r}")
                summary.setdefault('errors', []).append(f"{ag['name']}: description too long ({len(d)} chars): {d!r}")

        rsa_op = {
            'create': {
                'adGroup': ag_resource,
                'status': 'ENABLED',
                'ad': {
                    'finalUrls': [final_url],
                    'responsiveSearchAd': {
                        'headlines': [{'text': h[:30]} for h in ag['rsa']['headlines']],
                        'descriptions': [{'text': d[:90]} for d in ag['rsa']['descriptions']],
                    },
                },
            }
        }
        try:
            _mutate(token, 'adGroupAds', [rsa_op])
            summary['ads_added'] += 1
        except Exception as e:
            logger.error(f"RSA creation failed for {ag['name']}: {e}")
            summary.setdefault('errors', []).append(f"{ag['name']}: RSA failed: {str(e)[:200]}")

        summary['ad_groups'].append({
            'name': ag['name'],
            'resource': ag_resource,
            'keyword_count': len(ag['keywords']),
        })

    return summary


@google_ads_writer_bp.route('/api/dashboard/google-ads/fill-missing-ads', methods=['POST', 'OPTIONS'])
def fill_missing_ads():
    """For each ad group in the M1 campaign, create the static M1 ad if there
    isn't one yet. Idempotent. Used to backfill ads that failed during launch."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    token = _get_access_token()
    if not token:
        return jsonify({'error': 'Google Ads credentials not configured'}), 500

    try:
        # Find ad groups in our campaign and which already have ads
        existing = _query(token, f"""
            SELECT
                ad_group.id,
                ad_group.name,
                ad_group.resource_name,
                ad_group_ad.ad.id
            FROM ad_group_ad
            WHERE campaign.name = '{M1_CAMPAIGN_NAME}'
        """)

        # Map ad group name -> resource name, and track which have ads
        ad_groups_with_ads = set()
        ad_group_resources = {}
        for row in existing:
            ag = row.get('adGroup', {})
            name = ag.get('name')
            resource = ag.get('resourceName')
            if name and resource:
                ad_group_resources[name] = resource
            if row.get('adGroupAd', {}).get('ad', {}).get('id'):
                ad_groups_with_ads.add(name)

        # If some ad groups have no ads at all, the previous query won't include them
        # because ad_group_ad is the FROM. Run a separate query for ad groups.
        all_ag_rows = _query(token, f"""
            SELECT ad_group.id, ad_group.name, ad_group.resource_name
            FROM ad_group
            WHERE campaign.name = '{M1_CAMPAIGN_NAME}'
        """)
        for row in all_ag_rows:
            ag = row.get('adGroup', {})
            name = ag.get('name')
            resource = ag.get('resourceName')
            if name and resource:
                ad_group_resources[name] = resource

        created = []
        errors = []
        for ag_def in M1_AD_GROUPS:
            name = ag_def['name']
            if name in ad_groups_with_ads:
                continue  # already has an ad

            ag_resource = ad_group_resources.get(name)
            if not ag_resource:
                errors.append(f'Ad group "{name}" not found in campaign')
                continue

            final_url = f"{SITE_URL}{ag_def['final_url_path']}"

            # Validate character limits
            bad_headlines = [h for h in ag_def['rsa']['headlines'] if len(h) > 30]
            bad_descriptions = [d for d in ag_def['rsa']['descriptions'] if len(d) > 90]
            if bad_headlines or bad_descriptions:
                errors.append(f'{name}: invalid lengths - headlines: {bad_headlines}, descriptions: {bad_descriptions}')
                continue

            rsa_op = {
                'create': {
                    'adGroup': ag_resource,
                    'status': 'ENABLED',
                    'ad': {
                        'finalUrls': [final_url],
                        'responsiveSearchAd': {
                            'headlines': [{'text': h[:30]} for h in ag_def['rsa']['headlines']],
                            'descriptions': [{'text': d[:90]} for d in ag_def['rsa']['descriptions']],
                        },
                    },
                }
            }
            try:
                _mutate(token, 'adGroupAds', [rsa_op])
                created.append(name)
            except Exception as e:
                errors.append(f'{name}: {str(e)[:300]}')

        return jsonify({
            'ok': len(errors) == 0,
            'created': created,
            'already_had_ads': sorted(ad_groups_with_ads),
            'errors': errors,
        })
    except Exception as e:
        logger.error(f'fill_missing_ads failed: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


@google_ads_writer_bp.route('/api/dashboard/google-ads/launch-m1', methods=['POST', 'OPTIONS'])
def launch_m1():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    token = _get_access_token()
    if not token:
        return jsonify({'error': 'Google Ads credentials not configured'}), 500

    # Check if a campaign with this name already exists
    try:
        existing = _query(token, f"""
            SELECT campaign.id, campaign.name, campaign.status
            FROM campaign
            WHERE campaign.name = '{M1_CAMPAIGN_NAME}'
        """)
        if existing:
            return jsonify({
                'ok': False,
                'error': f'Campaign "{M1_CAMPAIGN_NAME}" already exists. Delete or rename it first.',
                'existing': existing,
            }), 400
    except Exception as e:
        logger.error(f'Campaign existence check failed: {e}')
        return jsonify({'error': f'Failed to check existing campaigns: {e}'}), 500

    try:
        summary = _create_m1_campaign(token)
        return jsonify({
            'ok': True,
            'message': 'M1 campaign created in PAUSED state. Review in Google Ads UI then enable it.',
            'summary': summary,
        })
    except Exception as e:
        logger.error(f'M1 launch failed: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


# --- Analyzer ---

# Thresholds (tuned for early-stage account)
RULE_PAUSE_KW_NO_CONV_CLICKS = 50      # pause if > 50 clicks and 0 conversions
RULE_PAUSE_KW_LOW_CTR_IMPS = 100        # consider pause if > 100 imps
RULE_PAUSE_KW_LOW_CTR_PCT = 0.01        # CTR below 1%
RULE_NEGATIVE_BAD_TERM_CLICKS = 3       # add negative if a search term wastes 3+ clicks no conv
RULE_BUDGET_AT_RISK_PCT = 0.90          # alert if daily spend > 90% cap


def _create_proposal(prop_type: str, payload: Dict, reason: str) -> Optional[Dict]:
    """Insert a proposal into Supabase. Skip if an identical pending one already exists."""
    # De-dupe: same type + same key fingerprint
    fingerprint = payload.get('criterion_id') or payload.get('search_term') or payload.get('ad_id') or payload.get('keyword_text')
    if fingerprint:
        try:
            check = requests.get(
                f"{SUPABASE_URL}/rest/v1/ad_proposals",
                params={
                    'select': 'id',
                    'type': f'eq.{prop_type}',
                    'status': 'eq.pending',
                    'payload->>fingerprint': f'eq.{fingerprint}',
                },
                headers=supabase_headers(),
                timeout=10,
            )
            if check.ok and check.json():
                return None  # already pending
        except Exception:
            pass

    payload['fingerprint'] = fingerprint or ''

    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/ad_proposals",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json={
                'type': prop_type,
                'payload': payload,
                'reason': reason,
                'status': 'pending',
            },
            timeout=10,
        )
        if r.ok:
            data = r.json()
            return data[0] if isinstance(data, list) and data else None
    except Exception as e:
        logger.error(f'Failed to insert proposal: {e}')
    return None


def _run_analyzer(token: str) -> Dict:
    """Pull last 14 days of data and emit proposals to ad_proposals table."""
    proposals_created = 0
    errors = []

    # 1. Keyword performance: pause keywords with high clicks but no conversions
    try:
        rows = _query(token, """
            SELECT
                ad_group_criterion.criterion_id,
                ad_group_criterion.keyword.text,
                ad_group_criterion.status,
                ad_group.id,
                ad_group.name,
                campaign.id,
                campaign.name,
                metrics.clicks,
                metrics.impressions,
                metrics.conversions,
                metrics.cost_micros,
                metrics.ctr
            FROM keyword_view
            WHERE segments.date DURING LAST_14_DAYS
              AND ad_group_criterion.status = 'ENABLED'
              AND campaign.status = 'ENABLED'
        """)

        for row in rows:
            crit = row.get('adGroupCriterion', {})
            ag = row.get('adGroup', {})
            metrics = row.get('metrics', {})

            clicks = int(metrics.get('clicks', 0))
            impressions = int(metrics.get('impressions', 0))
            conversions = float(metrics.get('conversions', 0))
            cost_micros = int(metrics.get('costMicros', 0))
            ctr = float(metrics.get('ctr', 0))

            crit_id = crit.get('criterionId')
            kw_text = crit.get('keyword', {}).get('text', '')

            # Rule: high clicks, no conversions
            if clicks >= RULE_PAUSE_KW_NO_CONV_CLICKS and conversions == 0:
                p = _create_proposal(
                    'pause_keyword',
                    {
                        'criterion_id': crit_id,
                        'ad_group_id': ag.get('id'),
                        'ad_group_resource': f"customers/{GOOGLE_ADS_CUSTOMER_ID}/adGroupCriteria/{ag.get('id')}~{crit_id}",
                        'keyword_text': kw_text,
                        'clicks_14d': clicks,
                        'conversions_14d': conversions,
                        'cost_14d_usd': round(cost_micros / 1_000_000, 2),
                    },
                    f"{clicks} clicks, 0 conversions in 14 days, ${round(cost_micros/1_000_000,2)} wasted",
                )
                if p:
                    proposals_created += 1

            # Rule: low CTR with enough impressions
            elif impressions >= RULE_PAUSE_KW_LOW_CTR_IMPS and ctr < RULE_PAUSE_KW_LOW_CTR_PCT:
                p = _create_proposal(
                    'pause_keyword',
                    {
                        'criterion_id': crit_id,
                        'ad_group_id': ag.get('id'),
                        'ad_group_resource': f"customers/{GOOGLE_ADS_CUSTOMER_ID}/adGroupCriteria/{ag.get('id')}~{crit_id}",
                        'keyword_text': kw_text,
                        'impressions_14d': impressions,
                        'ctr_14d': round(ctr, 4),
                    },
                    f"{impressions} impressions, CTR {round(ctr*100,2)}% (below 1% threshold)",
                )
                if p:
                    proposals_created += 1
    except Exception as e:
        errors.append(f'Keyword analysis: {str(e)[:200]}')

    # 2. Search terms: add bad terms as negatives
    try:
        rows = _query(token, """
            SELECT
                search_term_view.search_term,
                campaign.id,
                campaign.name,
                metrics.clicks,
                metrics.impressions,
                metrics.conversions,
                metrics.cost_micros
            FROM search_term_view
            WHERE segments.date DURING LAST_14_DAYS
              AND campaign.status = 'ENABLED'
        """)

        for row in rows:
            stv = row.get('searchTermView', {})
            campaign = row.get('campaign', {})
            metrics = row.get('metrics', {})

            term = stv.get('searchTerm', '')
            clicks = int(metrics.get('clicks', 0))
            conversions = float(metrics.get('conversions', 0))
            cost_micros = int(metrics.get('costMicros', 0))

            if clicks >= RULE_NEGATIVE_BAD_TERM_CLICKS and conversions == 0 and term:
                p = _create_proposal(
                    'add_negative',
                    {
                        'search_term': term,
                        'campaign_id': campaign.get('id'),
                        'campaign_resource': f"customers/{GOOGLE_ADS_CUSTOMER_ID}/campaigns/{campaign.get('id')}",
                        'match_type': 'PHRASE',
                        'clicks_14d': clicks,
                        'cost_14d_usd': round(cost_micros / 1_000_000, 2),
                    },
                    f"Search term '{term}' wasted {clicks} clicks (${round(cost_micros/1_000_000,2)}), no conversions",
                )
                if p:
                    proposals_created += 1
    except Exception as e:
        errors.append(f'Search term analysis: {str(e)[:200]}')

    # 3. Budget at risk
    try:
        rows = _query(token, """
            SELECT
                campaign.id,
                campaign.name,
                campaign_budget.amount_micros,
                metrics.cost_micros
            FROM campaign
            WHERE segments.date DURING TODAY
              AND campaign.status = 'ENABLED'
        """)
        for row in rows:
            campaign = row.get('campaign', {})
            budget = row.get('campaignBudget', {})
            metrics = row.get('metrics', {})

            spend_today = int(metrics.get('costMicros', 0))
            cap = int(budget.get('amountMicros', 0))

            if cap > 0 and spend_today / cap >= RULE_BUDGET_AT_RISK_PCT:
                p = _create_proposal(
                    'budget_alert',
                    {
                        'campaign_id': campaign.get('id'),
                        'campaign_name': campaign.get('name'),
                        'spend_today_usd': round(spend_today / 1_000_000, 2),
                        'cap_usd': round(cap / 1_000_000, 2),
                    },
                    f"{campaign.get('name')} at {round(spend_today/cap*100)}% of daily budget",
                )
                if p:
                    proposals_created += 1
    except Exception as e:
        errors.append(f'Budget analysis: {str(e)[:200]}')

    return {
        'proposals_created': proposals_created,
        'errors': errors,
        'analyzed_at': datetime.now(timezone.utc).isoformat(),
    }


@google_ads_writer_bp.route('/api/dashboard/google-ads/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    token = _get_access_token()
    if not token:
        return jsonify({'error': 'Google Ads credentials not configured'}), 500

    try:
        result = _run_analyzer(token)
        return jsonify({'ok': True, **result})
    except Exception as e:
        logger.error(f'Analyzer failed: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


# --- Proposals CRUD + apply ---

@google_ads_writer_bp.route('/api/dashboard/google-ads/proposals', methods=['GET', 'OPTIONS'])
def list_proposals():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    status = request.args.get('status', 'pending')
    limit = request.args.get('limit', '50')

    try:
        params = {
            'select': '*',
            'order': 'created_at.desc',
            'limit': limit,
        }
        if status and status != 'all':
            params['status'] = f'eq.{status}'

        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/ad_proposals",
            headers=supabase_headers(),
            params=params,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        logger.error(f'List proposals failed: {e}')
        return jsonify({'error': str(e)}), 500


def _apply_proposal(token: str, proposal: Dict) -> Dict:
    """Execute the action described by a proposal. Returns a result dict."""
    p_type = proposal.get('type')
    payload = proposal.get('payload', {})

    if p_type == 'pause_keyword':
        resource = payload.get('ad_group_resource')
        if not resource:
            return {'ok': False, 'error': 'Missing ad_group_resource in payload'}
        op = {
            'update': {
                'resourceName': resource,
                'status': 'PAUSED',
            },
            'updateMask': 'status',
        }
        result = _mutate(token, 'adGroupCriteria', [op])
        return {'ok': True, 'mutation': result}

    if p_type == 'add_negative':
        campaign_resource = payload.get('campaign_resource')
        term = payload.get('search_term')
        match_type = payload.get('match_type', 'PHRASE')
        if not campaign_resource or not term:
            return {'ok': False, 'error': 'Missing campaign_resource or search_term'}
        op = {
            'create': {
                'campaign': campaign_resource,
                'negative': True,
                'keyword': {'text': term, 'matchType': match_type},
            }
        }
        result = _mutate(token, 'campaignCriteria', [op])
        return {'ok': True, 'mutation': result}

    if p_type == 'budget_alert':
        # Informational only - nothing to mutate
        return {'ok': True, 'message': 'Acknowledged budget alert (no mutation needed)'}

    if p_type == 'new_ad_copy':
        ag_resource = payload.get('ad_group_resource')
        headlines = payload.get('headlines') or []
        descriptions = payload.get('descriptions') or []
        final_url = payload.get('final_url')

        if not ag_resource or not headlines or not descriptions or not final_url:
            return {'ok': False, 'error': 'Missing required fields (ad_group_resource, headlines, descriptions, final_url)'}
        if len(headlines) < 3 or len(descriptions) < 2:
            return {'ok': False, 'error': f'Google Ads requires 3-15 headlines and 2-4 descriptions (got {len(headlines)} headlines, {len(descriptions)} descriptions)'}

        op = {
            'create': {
                'adGroup': ag_resource,
                'status': 'ENABLED',
                'ad': {
                    'finalUrls': [final_url],
                    'responsiveSearchAd': {
                        'headlines': [{'text': h[:30]} for h in headlines[:15]],
                        'descriptions': [{'text': d[:90]} for d in descriptions[:4]],
                    },
                },
            }
        }
        result = _mutate(token, 'adGroupAds', [op])
        return {'ok': True, 'mutation': result, 'message': 'New RSA created. Google Ads will rotate it against existing ads.'}

    return {'ok': False, 'error': f'Unknown proposal type: {p_type}'}


def _update_proposal_status(proposal_id: str, status: str, apply_result: Optional[Dict] = None):
    update = {
        'status': status,
        'resolved_at': datetime.now(timezone.utc).isoformat(),
    }
    if apply_result is not None:
        update['apply_result'] = apply_result

    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/ad_proposals",
            headers=supabase_headers(),
            params={'id': f'eq.{proposal_id}'},
            json=update,
            timeout=10,
        )
    except Exception as e:
        logger.error(f'Failed to update proposal {proposal_id} status: {e}')


@google_ads_writer_bp.route('/api/dashboard/google-ads/proposals/<proposal_id>/approve', methods=['POST', 'OPTIONS'])
def approve_proposal(proposal_id):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    # Fetch proposal
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/ad_proposals",
            headers=supabase_headers(),
            params={'id': f'eq.{proposal_id}', 'select': '*'},
            timeout=10,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return jsonify({'error': 'Proposal not found'}), 404
        proposal = rows[0]
    except Exception as e:
        return jsonify({'error': f'Failed to fetch proposal: {e}'}), 500

    if proposal['status'] != 'pending':
        return jsonify({'error': f"Proposal status is '{proposal['status']}', cannot approve"}), 400

    token = _get_access_token()
    if not token:
        return jsonify({'error': 'Google Ads credentials not configured'}), 500

    try:
        result = _apply_proposal(token, proposal)
        if result.get('ok'):
            _update_proposal_status(proposal_id, 'applied', result)
            return jsonify({'ok': True, 'result': result})
        else:
            _update_proposal_status(proposal_id, 'failed', result)
            return jsonify({'ok': False, 'result': result}), 500
    except Exception as e:
        err = {'ok': False, 'error': str(e)}
        _update_proposal_status(proposal_id, 'failed', err)
        return jsonify(err), 500


@google_ads_writer_bp.route('/api/dashboard/google-ads/proposals/<proposal_id>/reject', methods=['POST', 'OPTIONS'])
def reject_proposal(proposal_id):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    _update_proposal_status(proposal_id, 'rejected')
    return jsonify({'ok': True})


# ============================================================================
# AI AD COPY GENERATOR
# Acts as a Google Ads specialist for DMHOA. Pulls performance data, asks
# Claude for fresh creative tuned to brand voice and UPL rules, saves results
# as proposals. User approves/rejects through the existing queue.
# ============================================================================

DMHOA_BRAND_BRIEF = """About DisputeMyHOA:
- A self-help document preparation tool for homeowners fighting unfair HOA notices.
- Branded around "the Dispute Engine" — our proprietary product name.
- Customers pay $49 for a complete dispute letter package they review and send themselves.
- Free notice review available with no credit card required.
- Brand voice: practical, calm, "you've got this", slightly defiant. Not corporate.
- Audience: regular homeowners who got an HOA fine or violation letter and want help.

CRITICAL UPL CONSTRAINTS (Unauthorized Practice of Law):
- DisputeMyHOA is NOT a law firm and does not provide legal advice.
- NEVER use these words/phrases anywhere in the ad copy:
  lawyer, attorney, legal advice, legal counsel, legal representation,
  represent you, win your case, guaranteed, legal-grade, lawyer-grade,
  case won, sue your HOA, court, lawsuit.
- NEVER promise specific outcomes ("guaranteed reversal", "100% success", "we win every case").
- Frame everything as: self-help tool, document preparation, educational, "you draft and send yourself".
- It is fine to say "fight", "appeal", "respond", "draft", "dispute".

CRITICAL BRANDING RULES:
- Do NOT say "AI" anywhere in headlines or descriptions. Say "Smart", "Dispute Engine", or just describe what it does.
- Refer to the product as "the Dispute Engine" or "DisputeMyHOA".

GOOGLE ADS RSA FORMAT:
- Headlines: max 30 characters each (count carefully). Aim for 8-10 distinct headlines per ad group.
- Descriptions: max 90 characters each. Aim for 4 descriptions per ad group.
- Mix angles: practical benefit, educational, urgency, social proof, "free" hook, price anchor.
- Each headline should be readable on its own.
- Avoid overused openers like "Are you", "Tired of", "Looking for".
"""


def _call_claude_for_ad_copy(user_prompt: str) -> Dict:
    """Call Claude with the brand brief + performance context to generate ad copy."""
    if not ANTHROPIC_API_KEY:
        raise Exception('ANTHROPIC_API_KEY not configured')

    system_prompt = (
        "You are a senior Google Ads specialist and copywriter working in-house "
        "at DisputeMyHOA. You write Responsive Search Ad headlines and descriptions "
        "that are UPL-compliant, on brand, and tested for character limits.\n\n"
        + DMHOA_BRAND_BRIEF
        + "\n\nReturn JSON only, no markdown, no code fences. Use this exact shape:\n"
        + '{"ad_groups":[{"ad_group_id":"...","ad_group_name":"...","rationale":"1-2 sentence angle","headlines":["..."],"descriptions":["..."]}]}'
        + HUMAN_VOICE_RULES
    )

    response = requests.post(
        ANTHROPIC_API_URL,
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01',
        },
        json={
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4096,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_prompt}],
        },
        timeout=(10, 120),
    )

    if not response.ok:
        raise Exception(f'Claude API error: {response.status_code} - {response.text[:500]}')

    data = response.json()
    raw = data['content'][0]['text']
    # Strip code fences if present
    cleaned = raw.replace('```json', '').replace('```', '').strip()
    return json.loads(cleaned)


def _fetch_ad_group_performance(token: str, campaign_name: str = M1_CAMPAIGN_NAME) -> List[Dict]:
    """For each enabled ad group in our managed campaign, return current ads
    and 14-day performance metrics."""
    # Query 1: ad groups in our campaign with their final URLs and current ads
    ad_groups_query = f"""
        SELECT
            ad_group.id,
            ad_group.name,
            ad_group.resource_name,
            campaign.id,
            campaign.name
        FROM ad_group
        WHERE campaign.name = '{campaign_name}'
          AND ad_group.status = 'ENABLED'
    """
    ag_rows = _query(token, ad_groups_query)
    if not ag_rows:
        return []

    results = []
    for row in ag_rows:
        ag = row.get('adGroup', {})
        ag_id = ag.get('id')
        ag_name = ag.get('name')
        ag_resource = ag.get('resourceName')

        # Query 2: get current responsive search ads in this ad group
        ads_query = f"""
            SELECT
                ad_group_ad.ad.id,
                ad_group_ad.ad.responsive_search_ad.headlines,
                ad_group_ad.ad.responsive_search_ad.descriptions,
                ad_group_ad.ad.final_urls,
                ad_group_ad.status,
                metrics.clicks,
                metrics.impressions,
                metrics.conversions,
                metrics.ctr
            FROM ad_group_ad
            WHERE ad_group.id = {ag_id}
              AND ad_group_ad.status = 'ENABLED'
              AND segments.date DURING LAST_14_DAYS
        """
        try:
            ads_rows = _query(token, ads_query)
        except Exception as e:
            logger.warning(f'Could not fetch ads for ad group {ag_id}: {e}')
            ads_rows = []

        # Aggregate metrics across this ad group
        total_clicks = 0
        total_impressions = 0
        total_conversions = 0.0
        current_ads = []
        final_url = None

        for ar in ads_rows:
            metrics = ar.get('metrics', {})
            total_clicks += int(metrics.get('clicks', 0))
            total_impressions += int(metrics.get('impressions', 0))
            total_conversions += float(metrics.get('conversions', 0))

            ad = ar.get('adGroupAd', {}).get('ad', {})
            rsa = ad.get('responsiveSearchAd', {})
            headlines = [h.get('text', '') for h in rsa.get('headlines', [])]
            descriptions = [d.get('text', '') for d in rsa.get('descriptions', [])]
            urls = ad.get('finalUrls', [])
            if urls and not final_url:
                final_url = urls[0]
            current_ads.append({
                'ad_id': ad.get('id'),
                'headlines': headlines,
                'descriptions': descriptions,
            })

        ctr = (total_clicks / total_impressions) if total_impressions > 0 else 0

        results.append({
            'ad_group_id': ag_id,
            'ad_group_name': ag_name,
            'ad_group_resource': ag_resource,
            'final_url': final_url or f"{SITE_URL}/",
            'metrics_14d': {
                'clicks': total_clicks,
                'impressions': total_impressions,
                'conversions': total_conversions,
                'ctr': round(ctr, 4),
            },
            'current_ads': current_ads,
        })

    return results


def _build_copy_user_prompt(performance: List[Dict]) -> str:
    """Build the user prompt with current performance and existing ads."""
    lines = [
        "Generate fresh Google Ads RSA copy for the ad groups below. For each ad group, write 8-10 headlines (max 30 chars each) and 4 descriptions (max 90 chars each).",
        "",
        "Use a different angle than the existing copy. Mix benefits, education, urgency, social proof, and price hooks. Stay UPL-compliant and on brand.",
        "",
        "AD GROUPS:",
        "",
    ]

    for ag in performance:
        m = ag.get('metrics_14d', {})
        lines.append(f"--- Ad Group: {ag['ad_group_name']} (id: {ag['ad_group_id']}) ---")
        lines.append(f"Landing page: {ag['final_url']}")
        lines.append(f"Last 14 days: {m.get('clicks', 0)} clicks, {m.get('impressions', 0)} impressions, {m.get('conversions', 0)} conversions, CTR {m.get('ctr', 0)*100:.2f}%")
        lines.append("Current headlines:")
        for ad in ag.get('current_ads', []):
            for h in ad.get('headlines', [])[:8]:
                lines.append(f"  - {h}")
            break
        lines.append("Current descriptions:")
        for ad in ag.get('current_ads', []):
            for d in ad.get('descriptions', [])[:4]:
                lines.append(f"  - {d}")
            break
        lines.append("")

    lines.append("Return JSON only with the schema specified in the system prompt. The ad_group_id values must match exactly.")
    return "\n".join(lines)


def _create_ad_copy_proposals(token: str) -> Dict:
    """Run the full generator pipeline: fetch performance, ask Claude, save proposals."""
    performance = _fetch_ad_group_performance(token)
    if not performance:
        return {'proposals_created': 0, 'error': f'No enabled ad groups found in campaign "{M1_CAMPAIGN_NAME}". Launch the M1 campaign first.'}

    user_prompt = _build_copy_user_prompt(performance)
    claude_result = _call_claude_for_ad_copy(user_prompt)

    # Index performance by ad_group_id for quick lookup
    perf_by_id = {str(p['ad_group_id']): p for p in performance}

    proposals_created = 0
    for new_ad in claude_result.get('ad_groups', []):
        ag_id = str(new_ad.get('ad_group_id', ''))
        perf = perf_by_id.get(ag_id)
        if not perf:
            logger.warning(f'Claude returned ad group id {ag_id} that was not in our request')
            continue

        # Validate character limits and sanitize
        headlines = [h.strip()[:30] for h in (new_ad.get('headlines') or []) if h and h.strip()]
        descriptions = [d.strip()[:90] for d in (new_ad.get('descriptions') or []) if d and d.strip()]
        # Google Ads requires 3-15 headlines and 2-4 descriptions
        if len(headlines) < 3 or len(descriptions) < 2:
            logger.warning(f'Skipping ad group {ag_id}: not enough valid headlines/descriptions')
            continue
        headlines = headlines[:15]
        descriptions = descriptions[:4]

        payload = {
            'ad_group_id': ag_id,
            'ad_group_name': new_ad.get('ad_group_name') or perf['ad_group_name'],
            'ad_group_resource': perf['ad_group_resource'],
            'final_url': perf['final_url'],
            'rationale': new_ad.get('rationale', ''),
            'headlines': headlines,
            'descriptions': descriptions,
            'current_metrics_14d': perf['metrics_14d'],
        }
        reason = f"New RSA variant for {perf['ad_group_name']}: {new_ad.get('rationale', '')[:120]}"
        result = _create_proposal('new_ad_copy', payload, reason)
        if result:
            proposals_created += 1

    return {'proposals_created': proposals_created, 'ad_groups_analyzed': len(performance)}


_copy_gen_status = {'running': False, 'last_result': None}


def _run_copy_generation_background():
    """Background runner for ad copy generation. Avoids Heroku 30s timeout."""
    _copy_gen_status['running'] = True
    try:
        token = _get_access_token()
        if not token:
            _copy_gen_status['last_result'] = {'ok': False, 'error': 'Google Ads credentials not configured'}
            return
        result = _create_ad_copy_proposals(token)
        _copy_gen_status['last_result'] = {'ok': True, **result}
        logger.info(f'Ad copy generator: {result.get("proposals_created", 0)} proposals created for {result.get("ad_groups_analyzed", 0)} ad groups')
    except Exception as e:
        logger.error(f'Background ad copy generation failed: {e}')
        _copy_gen_status['last_result'] = {'ok': False, 'error': str(e)}
    finally:
        _copy_gen_status['running'] = False


@google_ads_writer_bp.route('/api/dashboard/google-ads/generate-copy', methods=['POST', 'OPTIONS'])
def generate_ad_copy():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if _copy_gen_status['running']:
        return jsonify({'ok': True, 'message': 'Ad copy generator is already running. Refresh in ~30 seconds.'})

    import threading
    thread = threading.Thread(target=_run_copy_generation_background, daemon=True)
    thread.start()

    return jsonify({
        'ok': True,
        'message': 'Ad copy generator started. New proposals will appear in ~30 seconds.',
    })


@google_ads_writer_bp.route('/api/dashboard/google-ads/generate-copy/status', methods=['GET', 'OPTIONS'])
def generate_ad_copy_status():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})
    return jsonify({
        'running': _copy_gen_status['running'],
        'last_result': _copy_gen_status['last_result'],
    })
