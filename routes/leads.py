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

HOA_SUBS = ['HOA', 'homeowners', 'fuckhoa']
GENERAL_SUBS = ['FirstTimeHomeBuyer', 'personalfinance', 'legaladvice', 'RealEstate', 'neighborsfromhell', 'Landlord']

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
                    'ups': int(p.get('ups', 0) or 0),
                    'num_comments': int(p.get('num_comments', 0) or 0),
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

    # Attempt 2: RSS search feed
    try:
        rss_url = f'https://www.reddit.com/r/{sub}/search.rss'
        resp = http_requests.get(rss_url, headers=headers, params={
            'q': query,
            'restrict_sr': 'on',
            'sort': 'new',
            't': 'week',
            'limit': str(limit),
        }, timeout=15)
        if resp.status_code == 200 and '<entry>' in resp.text:
            posts = _parse_rss_posts(resp.text, sub)
            if posts:
                logger.info(f'Reddit RSS search: r/{sub} returned {len(posts)} posts')
                return posts
    except Exception as e:
        logger.warning(f'Reddit RSS search for r/{sub} failed: {e}')

    # Attempt 3: Pullpush (may be stale but accessible from any IP)
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


def _parse_rss_posts(xml_text: str, sub: str) -> list:
    """Parse Reddit Atom/RSS feed into our standard post format.
    Uses simple string parsing to avoid adding an XML dependency."""
    import re
    posts = []

    # Extract each <entry>...</entry> block
    entries = re.findall(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)
    for entry in entries:
        # Post ID: <id>t3_xxxxx</id> → extract just xxxxx
        id_match = re.search(r'<id>t3_([^<]+)</id>', entry)
        post_id = id_match.group(1) if id_match else ''

        # Title
        title_match = re.search(r'<title>([^<]*)</title>', entry)
        title = title_match.group(1) if title_match else ''
        # Decode HTML entities
        title = title.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', "'").replace('&quot;', '"')

        # Link (permalink)
        link_match = re.search(r'<link href="([^"]*)"', entry)
        permalink = link_match.group(1) if link_match else ''
        # Convert full URL to relative path
        if 'reddit.com' in permalink:
            permalink = '/' + permalink.split('reddit.com/', 1)[-1]

        # Published timestamp
        pub_match = re.search(r'<published>([^<]+)</published>', entry)
        created_utc = 0
        if pub_match:
            try:
                from datetime import datetime as dt
                pub_str = pub_match.group(1)
                # Parse ISO 8601
                pub_dt = dt.fromisoformat(pub_str.replace('Z', '+00:00'))
                created_utc = pub_dt.timestamp()
            except Exception:
                pass

        # Body text (strip HTML from content)
        content_match = re.search(r'<content[^>]*>(.*?)</content>', entry, re.DOTALL)
        selftext = ''
        if content_match:
            raw = content_match.group(1)
            # Strip HTML tags and decode entities
            selftext = re.sub(r'<[^>]+>', ' ', raw)
            selftext = selftext.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', "'").replace('&quot;', '"').replace('&nbsp;', ' ')
            selftext = ' '.join(selftext.split())[:2000]

        if post_id:
            posts.append({
                'id': post_id,
                'title': title,
                'selftext': selftext,
                'permalink': permalink,
                'created_utc': created_utc,
                'subreddit': sub,
                'ups': 0,  # RSS doesn't include vote/comment counts
                'num_comments': 0,
            })

    return posts


def _fetch_hoa_sub_posts(sub: str, limit: int = 100) -> list:
    """For HOA-focused subs, fetch all recent posts (they're all relevant).
    Strategy: RSS feed first (works from cloud IPs more often than JSON),
    then JSON, then Pullpush as last resort."""
    headers = {'User-Agent': 'DMHOA-LeadScraper/1.0'}
    posts = []

    # Attempt 1: RSS feed (Atom format, often less aggressively blocked)
    try:
        resp = http_requests.get(
            f'https://www.reddit.com/r/{sub}/new/.rss',
            headers=headers,
            params={'limit': min(limit, 100)},
            timeout=15,
        )
        if resp.status_code == 200 and '<entry>' in resp.text:
            posts = _parse_rss_posts(resp.text, sub)
            if posts:
                logger.info(f'Reddit RSS: r/{sub} returned {len(posts)} posts')
                return posts
        elif resp.status_code == 403:
            logger.info(f'Reddit RSS: r/{sub} returned 403')
    except Exception as e:
        logger.warning(f'Reddit RSS for r/{sub} failed: {e}')

    # Attempt 2: Reddit new.json
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
                    'ups': int(p.get('ups', 0) or 0),
                    'num_comments': int(p.get('num_comments', 0) or 0),
                })
            if posts:
                logger.info(f'Reddit new.json: r/{sub} returned {len(posts)} posts')
                return posts
    except Exception as e:
        logger.warning(f'Reddit new.json for r/{sub} failed: {e}')

    # Attempt 3: search with a broad query
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
                        'upvotes': post.get('ups', 0),
                        'num_comments': post.get('num_comments', 0),
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
                        'upvotes': post.get('ups', 0),
                        'num_comments': post.get('num_comments', 0),
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
        from zoneinfo import ZoneInfo
        update['replied_at'] = datetime.now(ZoneInfo('America/New_York')).isoformat()
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


@leads_bp.route('/api/dashboard/leads/daily-stats', methods=['GET', 'OPTIONS'])
def daily_lead_stats():
    """Return today's reply count for the daily activity counter."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    # Use local time (TZ=America/New_York on Heroku) so "today" matches EST
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo('America/New_York')
    today_start = datetime.now(eastern).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    try:
        resp = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
            headers=supabase_headers(),
            params={
                'select': 'id',
                'status': 'eq.replied',
                'replied_at': f'gte.{today_start}',
            },
            timeout=TIMEOUT,
        )
        replied_today = len(resp.json()) if resp.ok else 0

        return jsonify({
            'replied_today': replied_today,
            'goal': 10,
            'date': today_start[:10],
        })
    except Exception as e:
        return jsonify({'replied_today': 0, 'goal': 10, 'error': str(e)})


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


# ============================================================================
# REDDIT REPLY DRAFTER
# Scrapes the full Reddit post via Firecrawl, asks Claude to draft a
# personalized, helpful reply that soft-mentions DMHOA at the end.
# ============================================================================

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

REDDIT_USERNAME = 'disputemyhoa'

REPLY_SYSTEM_PROMPT = """You are Eric, founder of DisputeMyHOA.com, a self-help document preparation tool for homeowners dealing with HOA violations and fines.

You are writing a Reddit reply to someone who has a real HOA problem.

STEP 1: CATEGORIZE THE POST

Before drafting, decide which category this post falls into:

Category A "PROMO_OK": The person got an HOA violation letter, fine, or notice and doesn't know how to respond. They do NOT already have a lawyer. They are asking for help writing a response, appealing, or understanding their options. DisputeMyHOA is directly relevant.

Category B "HELPFUL_ONLY": Any of these are true:
- They already have a lawyer or legal representation
- They're in active litigation, arbitration, or mediation
- They're asking about HOA governance, elections, budgets, or board politics (not a violation/fine)
- They're asking about a non-dispute topic (maintenance, insurance, selling)
- The situation is too severe for a self-help tool (foreclosure, lien, lawsuit)

Output your category choice as the FIRST LINE of your reply in this exact format:
[CATEGORY: PROMO_OK] or [CATEGORY: HELPFUL_ONLY]

Then write the reply below that line.

STEP 2: WRITE THE REPLY

Rules for ALL replies:
1. Actually help them. Reference specifics from their post. Be concrete.
2. Keep it short. 3-5 paragraphs max.
3. Sound like a real person on Reddit. No corporate speak, no bullet lists.
4. Do NOT state legal conclusions about their specific case. Never say "they can't legally" or "this is illegal" or "you have a strong case." Instead frame as general education: "in [state], HOAs are generally expected to..." or "most CC&Rs require..." or "you might want to check whether..."
5. Do NOT say "I am not a lawyer" or "this is not legal advice." Screams bot.
6. Do NOT start with "Hey there!" or "Great question!" Just start.

If PROMO_OK:
- At the very end, one natural mention: "if you want help drafting an actual response letter, I built disputemyhoa.com for exactly this."
- One line. Not a paragraph. Not a sales pitch.

If HELPFUL_ONLY:
- Do NOT mention DisputeMyHOA, disputemyhoa.com, or any product at all.
- Just be genuinely helpful. The value is in building credibility as a knowledgeable Reddit user. Lurkers who see your helpful replies will check your profile on their own.

WRITING STYLE RULES (critical):
- Never use em-dashes or en-dashes. Use periods, commas, colons, or parentheses instead.
- Never use: delve, leverage, robust, seamlessly, comprehensive, holistic, empower, streamline, cutting-edge, state-of-the-art, embark, harness, tapestry, vibrant, transformative, paramount, pivotal, moreover, furthermore, in essence, it is worth noting, in conclusion, ultimately, navigate the complexities.
- Do not start sentences with "Indeed", "Notably", "Importantly", or "However,".
- Write plain, direct, conversational English. Short sentences. Sound like a real person on Reddit."""


def _find_replies_to_user(thread_json: list, username: str) -> list:
    """Walk the Reddit comment tree and find replies to comments by `username`.
    Returns a list of {parent_comment, reply_author, reply_body, reply_id, reply_created_utc}."""
    replies_found = []

    def _walk_comments(children: list, parent_author: str = ''):
        for child in children:
            if child.get('kind') != 't1':
                continue
            data = child.get('data', {})
            author = data.get('author', '')
            body = data.get('body', '')
            comment_id = data.get('id', '')
            created_utc = data.get('created_utc', 0)
            parent_body = data.get('parent_id', '')

            # If the parent of this comment was written by our user, this is a reply TO us
            if parent_author.lower() == username.lower() and author.lower() != username.lower():
                replies_found.append({
                    'reply_author': author,
                    'reply_body': body[:1000],
                    'reply_id': comment_id,
                    'reply_created_utc': created_utc,
                })

            # If THIS comment is by our user, check its direct children for replies
            nested = data.get('replies')
            if isinstance(nested, dict):
                nested_children = nested.get('data', {}).get('children', [])
                _walk_comments(nested_children, parent_author=author)

    # thread_json is [post_listing, comments_listing]
    if len(thread_json) > 1:
        top_level = thread_json[1].get('data', {}).get('children', [])
        # First pass: find our comments at top level
        for child in top_level:
            if child.get('kind') != 't1':
                continue
            data = child.get('data', {})
            author = data.get('author', '')
            # Walk nested replies under each top-level comment
            nested = data.get('replies')
            if isinstance(nested, dict):
                nested_children = nested.get('data', {}).get('children', [])
                _walk_comments(nested_children, parent_author=author)

    return replies_found


@leads_bp.route('/api/dashboard/leads/check-replies', methods=['POST', 'OPTIONS'])
def check_replies():
    """Check all 'replied' leads for new responses to u/disputemyhoa's comments.
    Fetches each replied thread from Reddit and walks the comment tree to find
    replies directed at our user."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    # Fetch all replied leads
    try:
        resp = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
            headers=supabase_headers(),
            params={
                'status': 'eq.replied',
                'select': 'id,post_id,title,url,subreddit,replied_at',
                'order': 'replied_at.desc',
                'limit': '50',
            },
            timeout=TIMEOUT,
        )
        if not resp.ok:
            return jsonify({'error': 'Failed to fetch replied leads'}), 500
        leads = resp.json() or []
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if not leads:
        return jsonify({'ok': True, 'threads_checked': 0, 'replies': []})

    all_replies = []
    threads_checked = 0
    errors = []

    for lead in leads:
        reddit_url = lead.get('url', '')
        if not reddit_url:
            continue

        # Build the .json URL for the thread
        json_url = reddit_url.replace('https://reddit.com', 'https://www.reddit.com')
        if not json_url.startswith('https://www.reddit.com'):
            json_url = f'https://www.reddit.com{reddit_url}' if reddit_url.startswith('/') else reddit_url
        json_url = json_url.rstrip('/') + '.json'

        try:
            r = http_requests.get(json_url, headers={'User-Agent': 'DMHOA-ReplyChecker/1.0'}, timeout=15)
            if r.status_code == 200:
                thread_data = r.json()
                replies = _find_replies_to_user(thread_data, REDDIT_USERNAME)
                threads_checked += 1

                for reply in replies:
                    all_replies.append({
                        'lead_id': lead.get('id'),
                        'lead_title': lead.get('title', ''),
                        'subreddit': lead.get('subreddit', ''),
                        'reddit_url': reddit_url,
                        **reply,
                    })
            elif r.status_code == 403:
                # Try RSS fallback — but RSS doesn't give us nested comments
                # so we can't find replies. Just skip.
                errors.append(f"r/{lead.get('subreddit')}: Reddit returned 403")
            else:
                errors.append(f"r/{lead.get('subreddit')}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{lead.get('post_id')}: {str(e)[:100]}")

        # Be polite between requests
        time.sleep(0.5)

    # Sort by most recent reply first
    all_replies.sort(key=lambda x: x.get('reply_created_utc', 0), reverse=True)

    return jsonify({
        'ok': True,
        'threads_checked': threads_checked,
        'replies': all_replies,
        'errors': errors if errors else None,
    })


@leads_bp.route('/api/dashboard/leads/<lead_id>/draft-follow-up', methods=['POST', 'OPTIONS'])
def draft_follow_up(lead_id):
    """Draft a follow-up reply to someone who responded to our Reddit comment.
    Body: { "their_reply": "...", "thread_context": "..." (optional) }"""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY not configured'}), 500

    body = request.get_json(silent=True) or {}
    their_reply = (body.get('their_reply') or '').strip()
    thread_context = (body.get('thread_context') or '').strip()

    if not their_reply:
        return jsonify({'error': 'their_reply is required'}), 400

    user_prompt = f"""Someone replied to my Reddit comment. I need to write a follow-up.

Their reply to me:
---
{their_reply[:2000]}
---

{f"Thread context: {thread_context[:2000]}" if thread_context else ""}

Write a short, natural follow-up response (1-3 paragraphs). Be helpful and conversational. Do NOT mention disputemyhoa.com again since I already mentioned it in my original comment. Just be genuinely helpful."""

    try:
        resp = http_requests.post(
            ANTHROPIC_API_URL,
            headers={
                'x-api-key': ANTHROPIC_API_KEY,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 512,
                'system': REPLY_SYSTEM_PROMPT,
                'messages': [{'role': 'user', 'content': user_prompt}],
            },
            timeout=(10, 60),
        )

        if not resp.ok:
            return jsonify({'error': f'Claude API error: {resp.status_code}'}), 500

        data = resp.json()
        raw_reply = data.get('content', [{}])[0].get('text', '')

        # Strip category tag if present
        import re
        reply_text = raw_reply
        cat_match = re.match(r'\[CATEGORY:\s*\w+\]\s*\n?', raw_reply)
        if cat_match:
            reply_text = raw_reply[cat_match.end():].strip()

        return jsonify({'ok': True, 'reply': reply_text})

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@leads_bp.route('/api/dashboard/leads/<lead_id>/draft-reply', methods=['POST', 'OPTIONS'])
def draft_reply(lead_id):
    """Scrape the Reddit post via Firecrawl, then ask Claude to draft a
    personalized reply. Returns the draft text for the user to review/edit.

    Body (optional): { "extra_context": "..." } — any notes the user wants
    Claude to incorporate into the reply.
    """
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY not configured'}), 500

    # Look up the lead to get the URL and title
    try:
        resp = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/dmhoa_leads",
            headers=supabase_headers(),
            params={'id': f'eq.{lead_id}', 'select': '*'},
            timeout=TIMEOUT,
        )
        if not resp.ok or not resp.json():
            return jsonify({'error': 'Lead not found'}), 404
        lead = resp.json()[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    reddit_url = lead.get('url', '')
    title = lead.get('title', '')
    subreddit = lead.get('subreddit', '')

    body = request.get_json(silent=True) or {}
    extra_context = (body.get('extra_context') or '').strip()

    # Step 1: Fetch the full Reddit post + comments
    # Reddit's .json endpoint returns the full post body + all comments.
    # We try this first (works from most IPs), fall back to RSS, then to
    # just the title if everything fails.
    scraped = ''

    # Extract the Reddit post path from the URL for the .json endpoint
    # URL format: https://reddit.com/r/HOA/comments/1si6nkl/...
    json_url = reddit_url.replace('https://reddit.com', 'https://www.reddit.com')
    if not json_url.startswith('https://www.reddit.com'):
        json_url = f'https://www.reddit.com{reddit_url}' if reddit_url.startswith('/') else reddit_url

    # Strip trailing slashes and add .json
    json_url = json_url.rstrip('/') + '.json'

    try:
        resp = http_requests.get(json_url, headers={'User-Agent': 'DMHOA-ReplyDrafter/1.0'}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Reddit returns [post_listing, comments_listing]
            post_data = data[0]['data']['children'][0]['data'] if data else {}
            post_title = post_data.get('title', title)
            post_body = post_data.get('selftext', '')
            post_author = post_data.get('author', '')
            post_sub = post_data.get('subreddit', subreddit)

            parts = [f"Subreddit: r/{post_sub}", f"Title: {post_title}", f"Author: u/{post_author}", "", post_body, "", "--- COMMENTS ---"]

            comments = data[1]['data']['children'] if len(data) > 1 else []
            for c in comments[:20]:
                cd = c.get('data', {})
                if cd.get('body'):
                    parts.append(f"\nu/{cd.get('author', 'unknown')}:")
                    parts.append(cd['body'][:500])

            scraped = '\n'.join(parts)
            logger.info(f'Reddit JSON: fetched {len(scraped)} chars for {reddit_url}')
        else:
            logger.warning(f'Reddit JSON returned {resp.status_code} for {json_url}')
    except Exception as e:
        logger.warning(f'Reddit JSON fetch failed: {e}')

    # Fallback: try RSS
    if not scraped:
        try:
            rss_url = json_url.replace('.json', '/.rss')
            resp = http_requests.get(rss_url, headers={'User-Agent': 'DMHOA-ReplyDrafter/1.0'}, timeout=15)
            if resp.status_code == 200 and '<entry>' in resp.text:
                # Quick parse: strip HTML tags from entries
                import re
                entries = re.findall(r'<content[^>]*>(.*?)</content>', resp.text, re.DOTALL)
                parts = [f"Subreddit: r/{subreddit}", f"Title: {title}", ""]
                for entry in entries[:15]:
                    text = re.sub(r'<[^>]+>', ' ', entry)
                    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', "'").replace('&quot;', '"')
                    parts.append(' '.join(text.split())[:600])
                    parts.append("")
                scraped = '\n'.join(parts)
                logger.info(f'Reddit RSS: fetched {len(scraped)} chars for {reddit_url}')
        except Exception as e:
            logger.warning(f'Reddit RSS fetch failed: {e}')

    if not scraped:
        scraped = f"Post title: {title}\nSubreddit: r/{subreddit}\n(Full post text could not be retrieved. Draft based on title only.)"

    # Truncate to ~8k chars to stay within token budget
    if len(scraped) > 8000:
        scraped = scraped[:8000] + '\n\n[... post truncated for length]'

    # Step 2: Ask Claude to draft the reply
    user_prompt = f"""Here is a Reddit post from r/{subreddit} that I want to reply to:

---
{scraped}
---

{f"Additional context from me: {extra_context}" if extra_context else ""}

Draft a Reddit reply from me (Eric, founder of disputemyhoa.com). Be genuinely helpful first, mention my tool once at the end. Keep it natural and short."""

    try:
        resp = http_requests.post(
            ANTHROPIC_API_URL,
            headers={
                'x-api-key': ANTHROPIC_API_KEY,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 1024,
                'system': REPLY_SYSTEM_PROMPT,
                'messages': [{'role': 'user', 'content': user_prompt}],
            },
            timeout=(10, 60),
        )

        if not resp.ok:
            return jsonify({'error': f'Claude API error: {resp.status_code} - {resp.text[:300]}'}), 500

        data = resp.json()
        raw_reply = data.get('content', [{}])[0].get('text', '')

        # Parse out the category tag from the first line
        import re
        category = 'unknown'
        reply_text = raw_reply
        cat_match = re.match(r'\[CATEGORY:\s*(PROMO_OK|HELPFUL_ONLY)\]\s*\n?', raw_reply)
        if cat_match:
            category = cat_match.group(1).lower()
            reply_text = raw_reply[cat_match.end():].strip()

        return jsonify({
            'ok': True,
            'reply': reply_text,
            'category': category,
            'lead_id': lead_id,
            'reddit_url': reddit_url,
            'subreddit': subreddit,
            'title': title,
            'scraped_length': len(scraped),
        })

    except Exception as e:
        logger.error(f'draft_reply failed for lead {lead_id}: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500
