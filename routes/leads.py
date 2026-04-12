"""
Flask blueprint for Reddit lead management.
Routes: GET /api/dashboard/leads, PATCH /api/dashboard/leads/:id, POST /api/dashboard/leads/run-scraper
"""

import os
import time
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
REDDIT_SEARCH_URL = 'https://www.reddit.com/r/{sub}/search.json'

HOA_SUBS = ['HOA', 'homeowners']
GENERAL_SUBS = ['FirstTimeHomeBuyer', 'personalfinance', 'legaladvice']

# Only keep posts newer than this many days
MAX_AGE_DAYS = 7

HOA_SUB_KEYWORDS = [
    'fine', 'fined', 'violation', 'dispute', 'letter', 'threatening',
    'appeal', 'fight', 'complaint', 'board', 'fee', 'assessment',
    'lien', 'rule', 'enforce', 'notice', 'penalty', 'cc&r', 'ccr',
    'covenant', 'bylaw', 'hearing', 'help', 'advice', 'what should',
    'need advice', 'frustrated', 'concern', 'issue', 'problem',
]

# Search queries used on general subs (tried on Reddit search first)
HOA_SEARCH_QUERY = 'hoa fine OR hoa violation OR hoa dispute OR hoa letter OR hoa appeal'
HOA_BROAD_QUERY = 'hoa'

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


def _is_recent(created_utc: float) -> bool:
    """True if the post is younger than MAX_AGE_DAYS."""
    if not created_utc:
        return False
    age_seconds = time.time() - created_utc
    return age_seconds < (MAX_AGE_DAYS * 86400)


def _fetch_reddit_search(sub: str, query: str, limit: int = 50) -> list:
    """Try Reddit's own search.json endpoint first (fresh data).
    Falls back to Pullpush if Reddit returns 403 from cloud IPs."""
    headers = {'User-Agent': 'DMHOA-LeadScraper/1.0'}
    posts = []

    # Attempt 1: Reddit search.json (returns real-time fresh posts)
    try:
        url = REDDIT_SEARCH_URL.format(sub=sub)
        resp = http_requests.get(url, headers=headers, params={
            'q': query,
            'restrict_sr': 'on',
            'sort': 'new',
            't': 'week',
            'limit': str(limit),
        }, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            children = data.get('data', {}).get('children', [])
            for child in children:
                p = child.get('data', {})
                posts.append({
                    'id': p.get('id', ''),
                    'title': p.get('title', ''),
                    'selftext': p.get('selftext', ''),
                    'permalink': p.get('permalink', ''),
                    'created_utc': p.get('created_utc', 0),
                    'subreddit': p.get('subreddit', sub),
                })
            if posts:
                logger.info(f'Reddit search: r/{sub} returned {len(posts)} posts')
                return posts
        elif resp.status_code == 403:
            logger.info(f'Reddit search: r/{sub} returned 403, falling back to Pullpush')
        else:
            logger.warning(f'Reddit search: r/{sub} returned {resp.status_code}')
    except Exception as e:
        logger.warning(f'Reddit search for r/{sub} failed: {e}')

    # Attempt 2: Pullpush (may be stale but accessible from any IP)
    try:
        cutoff = int(time.time() - (MAX_AGE_DAYS * 86400))
        resp = http_requests.get(PULLPUSH_URL, headers=headers, params={
            'subreddit': sub,
            'q': query,
            'size': limit,
            'sort': 'desc',
            'sort_type': 'created_utc',
            'after': cutoff,
        }, timeout=20)
        if resp.status_code == 200:
            for p in resp.json().get('data', []):
                posts.append({
                    'id': p.get('id', ''),
                    'title': p.get('title', ''),
                    'selftext': p.get('selftext', ''),
                    'permalink': f"/r/{sub}/comments/{p.get('id', '')}",
                    'created_utc': p.get('created_utc', 0),
                    'subreddit': sub,
                })
            if posts:
                logger.info(f'Pullpush: r/{sub} returned {len(posts)} posts')
    except Exception as e:
        logger.warning(f'Pullpush for r/{sub} failed: {e}')

    return posts


def _fetch_hoa_sub_posts(sub: str, limit: int = 100) -> list:
    """For HOA-focused subs, fetch all recent posts (they're all relevant).
    Tries Reddit new.json first, then search, then Pullpush."""
    headers = {'User-Agent': 'DMHOA-LeadScraper/1.0'}
    posts = []

    # Reddit new.json (most complete for HOA-focused subs)
    try:
        resp = http_requests.get(
            f'https://www.reddit.com/r/{sub}/new.json',
            headers=headers,
            params={'limit': min(limit, 100)},
            timeout=15,
        )
        if resp.status_code == 200:
            children = resp.json().get('data', {}).get('children', [])
            for child in children:
                p = child.get('data', {})
                posts.append({
                    'id': p.get('id', ''),
                    'title': p.get('title', ''),
                    'selftext': p.get('selftext', ''),
                    'permalink': p.get('permalink', ''),
                    'created_utc': p.get('created_utc', 0),
                    'subreddit': p.get('subreddit', sub),
                })
            if posts:
                logger.info(f'Reddit new.json: r/{sub} returned {len(posts)} posts')
                return posts
        elif resp.status_code == 403:
            logger.info(f'Reddit new.json: r/{sub} returned 403, trying search')
    except Exception as e:
        logger.warning(f'Reddit new.json for r/{sub} failed: {e}')

    # Fall back to search with a broad query
    return _fetch_reddit_search(sub, HOA_BROAD_QUERY, limit)


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

        # HOA-focused subs: get all recent posts, filter by keywords
        for sub in HOA_SUBS:
            try:
                posts = _fetch_hoa_sub_posts(sub)
                for post in posts:
                    pid = post.get('id', '')
                    if not pid or pid in skip_ids:
                        continue
                    if not _is_recent(post.get('created_utc', 0)):
                        continue
                    combined = f"{post.get('title', '')} {post.get('selftext', '')}"
                    if not any(kw in combined.lower() for kw in HOA_SUB_KEYWORDS):
                        continue

                    permalink = post.get('permalink', '')
                    url = f"https://reddit.com{permalink}" if permalink.startswith('/') else f"https://reddit.com/r/{sub}/comments/{pid}"
                    lead = {
                        'post_id': pid,
                        'subreddit': sub,
                        'title': post.get('title', '')[:500],
                        'url': url,
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

        # General subs: search for HOA-related terms
        for sub in GENERAL_SUBS:
            try:
                posts = _fetch_reddit_search(sub, HOA_SEARCH_QUERY, 50)
                for post in posts:
                    pid = post.get('id', '')
                    if not pid or pid in skip_ids:
                        continue
                    if not _is_recent(post.get('created_utc', 0)):
                        continue
                    combined = f"{post.get('title', '')} {post.get('selftext', '')}"

                    permalink = post.get('permalink', '')
                    url = f"https://reddit.com{permalink}" if permalink.startswith('/') else f"https://reddit.com/r/{sub}/comments/{pid}"
                    lead = {
                        'post_id': pid,
                        'subreddit': sub,
                        'title': post.get('title', '')[:500],
                        'url': url,
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


@leads_bp.route('/api/dashboard/leads/purge-stale', methods=['POST', 'OPTIONS'])
def purge_stale_leads():
    """Delete all leads older than MAX_AGE_DAYS that are still in 'new' status.
    Keeps replied/skipped leads for historical reference."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    cutoff = datetime.fromtimestamp(
        time.time() - (MAX_AGE_DAYS * 86400), tz=timezone.utc
    ).isoformat()

    try:
        resp = http_requests.delete(
            f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            params={
                'status': 'eq.new',
                'created_utc': f'lt.{cutoff}',
            },
            timeout=TIMEOUT,
        )
        if resp.ok:
            deleted = resp.json() or []
            return jsonify({'ok': True, 'purged': len(deleted)})
        return jsonify({'ok': False, 'error': resp.text[:300]}), 500
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
