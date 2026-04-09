"""
Flask blueprint for Reddit lead management.
Routes: GET /api/dashboard/leads, PATCH /api/dashboard/leads/:id, POST /api/dashboard/leads/run-scraper
"""

import os
import threading
import logging
from datetime import datetime, timezone

import requests as http_requests
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

leads_bp = Blueprint('leads', __name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
TIMEOUT = (5, 60)

# --- Inline scraper (avoids subprocess timeout) ---

PULLPUSH_URL = 'https://api.pullpush.io/reddit/search/submission/'
HOA_SUBS = ['HOA', 'homeowners']
GENERAL_SUBS = ['FirstTimeHomeBuyer', 'personalfinance', 'legaladvice']

HOA_SUB_KEYWORDS = [
    'fine', 'fined', 'violation', 'dispute', 'letter', 'threatening',
    'appeal', 'fight', 'complaint', 'board', 'fee', 'assessment',
    'lien', 'rule', 'enforce', 'notice', 'penalty', 'cc&r', 'ccr',
    'covenant', 'bylaw', 'hearing', 'help', 'advice', 'what should',
    'need advice', 'frustrated', 'concern', 'issue', 'problem',
]

SCORE_WORDS_HIGH = ['fined', 'violation', 'threatened', 'lawsuit', 'lien', 'penalty', 'foreclos']
SCORE_WORDS_MED = ['help', 'advice', 'what do i do', 'what should i', 'need advice', 'frustrated']
SCORE_WORDS_LOW = ['lawyer', 'attorney', 'legal', 'court']

_scraper_status = {'running': False, 'last_result': None}


def _supabase_headers_upsert():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates',
    }


def _score_post(text):
    text_lower = text.lower()
    score = 0
    for w in SCORE_WORDS_HIGH:
        if w in text_lower:
            score += 3
    for w in SCORE_WORDS_MED:
        if w in text_lower:
            score += 2
    for w in SCORE_WORDS_LOW:
        if w in text_lower:
            score += 1
    return score


def _run_scrape():
    """Run the scraper in a background thread."""
    _scraper_status['running'] = True
    count = 0
    errors = []

    try:
        # Get existing skipped/replied IDs
        skip_ids = set()
        try:
            resp = http_requests.get(
                f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
                headers=supabase_headers(),
                params={'select': 'post_id', 'status': 'in.(replied,skipped)'},
                timeout=10
            )
            if resp.ok:
                skip_ids = {r['post_id'] for r in resp.json()}
        except Exception as e:
            logger.error(f"Failed to fetch existing IDs: {e}")

        headers = {'User-Agent': 'DMHOA-LeadScraper/1.0'}

        # HOA-focused subs: get all recent posts
        for sub in HOA_SUBS:
            try:
                resp = http_requests.get(PULLPUSH_URL, headers=headers,
                    params={'subreddit': sub, 'size': 100, 'sort': 'desc', 'sort_type': 'created_utc'},
                    timeout=20)
                if resp.status_code != 200:
                    errors.append(f"r/{sub}: HTTP {resp.status_code}")
                    continue

                posts = resp.json().get('data', [])
                for post in posts:
                    pid = post.get('id', '')
                    if pid in skip_ids:
                        continue
                    combined = f"{post.get('title', '')} {post.get('selftext', '')}"
                    if not any(kw in combined.lower() for kw in HOA_SUB_KEYWORDS):
                        continue

                    lead = {
                        'post_id': pid,
                        'subreddit': sub,
                        'title': post.get('title', '')[:500],
                        'url': f"https://reddit.com/r/{sub}/comments/{pid}",
                        'score': _score_post(combined),
                        'status': 'new',
                        'created_utc': datetime.fromtimestamp(
                            post.get('created_utc', 0), tz=timezone.utc
                        ).isoformat(),
                    }
                    r = http_requests.post(
                        f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
                        headers=_supabase_headers_upsert(), json=lead, timeout=10)
                    if r.status_code in (200, 201):
                        count += 1
            except Exception as e:
                errors.append(f"r/{sub}: {str(e)[:100]}")

        # General subs: search for "hoa"
        for sub in GENERAL_SUBS:
            try:
                resp = http_requests.get(PULLPUSH_URL, headers=headers,
                    params={'subreddit': sub, 'q': 'hoa', 'size': 50, 'sort': 'desc', 'sort_type': 'created_utc'},
                    timeout=20)
                if resp.status_code != 200:
                    errors.append(f"r/{sub}: HTTP {resp.status_code}")
                    continue

                posts = resp.json().get('data', [])
                for post in posts:
                    pid = post.get('id', '')
                    if pid in skip_ids:
                        continue
                    combined = f"{post.get('title', '')} {post.get('selftext', '')}"
                    lead = {
                        'post_id': pid,
                        'subreddit': sub,
                        'title': post.get('title', '')[:500],
                        'url': f"https://reddit.com/r/{sub}/comments/{pid}",
                        'score': _score_post(combined),
                        'status': 'new',
                        'created_utc': datetime.fromtimestamp(
                            post.get('created_utc', 0), tz=timezone.utc
                        ).isoformat(),
                    }
                    r = http_requests.post(
                        f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
                        headers=_supabase_headers_upsert(), json=lead, timeout=10)
                    if r.status_code in (200, 201):
                        count += 1
            except Exception as e:
                errors.append(f"r/{sub}: {str(e)[:100]}")

        _scraper_status['last_result'] = {
            'ok': True,
            'count': count,
            'errors': errors,
        }

    except Exception as e:
        _scraper_status['last_result'] = {'ok': False, 'error': str(e)}

    finally:
        _scraper_status['running'] = False
        logger.info(f"Scrape complete. {count} leads upserted. Errors: {errors}")


# --- Routes ---

def supabase_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    }


@leads_bp.route('/api/dashboard/leads', methods=['GET', 'OPTIONS'])
def get_leads():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    status = request.args.get('status', '')
    limit = request.args.get('limit', '20')

    url = f"{SUPABASE_URL}/rest/v1/dmhoa_leads"
    params = {
        'select': '*',
        'order': 'score.desc,created_utc.desc',
        'limit': limit,
    }
    if status and status != 'all':
        params['status'] = f'eq.{status}'

    try:
        resp = http_requests.get(url, headers=supabase_headers(), params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        logger.error(f"Error fetching leads: {e}")
        return jsonify({'error': str(e)}), 500


@leads_bp.route('/api/dashboard/leads/<lead_id>', methods=['PATCH', 'OPTIONS'])
def update_lead(lead_id):
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    body = request.get_json() or {}
    new_status = body.get('status')

    if new_status not in ('replied', 'skipped', 'new'):
        return jsonify({'error': 'Invalid status. Must be: new, replied, skipped'}), 400

    update = {'status': new_status}
    if new_status == 'replied':
        update['replied_at'] = datetime.now(timezone.utc).isoformat()
    elif new_status == 'new':
        update['replied_at'] = None

    url = f"{SUPABASE_URL}/rest/v1/dmhoa_leads"
    headers = {**supabase_headers(), 'Prefer': 'return=representation'}
    params = {'id': f'eq.{lead_id}'}

    try:
        resp = http_requests.patch(url, headers=headers, json=update, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return jsonify(data[0])
        return jsonify({'error': 'Lead not found'}), 404
    except Exception as e:
        logger.error(f"Error updating lead {lead_id}: {e}")
        return jsonify({'error': str(e)}), 500


@leads_bp.route('/api/dashboard/leads/run-scraper', methods=['POST', 'OPTIONS'])
def run_scraper():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if _scraper_status['running']:
        return jsonify({'ok': True, 'message': 'Scraper is already running'})

    # Run in background thread so we can respond immediately
    thread = threading.Thread(target=_run_scrape, daemon=True)
    thread.start()

    return jsonify({'ok': True, 'message': 'Scraper started. Refresh in ~30 seconds to see new leads.'})


@leads_bp.route('/api/dashboard/leads/scraper-status', methods=['GET', 'OPTIONS'])
def scraper_status():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    return jsonify({
        'running': _scraper_status['running'],
        'last_result': _scraper_status['last_result'],
    })
