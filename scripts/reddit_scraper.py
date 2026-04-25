"""
Reddit Lead Scraper for DMHOA
Monitors HOA-related subreddits for potential leads.
Uses Reddit's public JSON endpoints — no API key needed.

Usage:
    python scripts/reddit_scraper.py

Requires env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---

REDDIT_JSON_BASE = 'https://www.reddit.com'

MAX_POST_AGE_DAYS = 3

# HOA-focused subs: every post is HOA-related by definition
HOA_SUBS = ['HOA']

# Adjacent subs: HOA-relevant posts mixed with unrelated home-owner content
ADJACENT_SUBS = ['homeowners']

# General subs: search for "hoa" keyword
GENERAL_SUBS = ['FirstTimeHomeBuyer', 'personalfinance', 'legaladvice']

# Score thresholds — anything below skips save
MIN_SCORE_HOA_SUBS = 1
MIN_SCORE_ADJACENT_SUBS = 2
MIN_SCORE_GENERAL_SUBS = 2

# Terms used to confirm a post is actually about an HOA (used outside r/HOA)
HOA_PRESENCE_TERMS = [
    'hoa', 'h.o.a', 'homeowners association', 'homeowner association',
    'community association', 'condo association', 'property owners association',
    'cc&r', 'ccr', 'covenant',
]

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

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

HEADERS = {'User-Agent': 'DMHOA-LeadScraper/1.0'}


def supabase_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates',
    }


def matches_keywords(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in HOA_SUB_KEYWORDS)


def mentions_hoa(text):
    text_lower = text.lower()
    return any(term in text_lower for term in HOA_PRESENCE_TERMS)


def score_post(text):
    text_lower = text.lower()
    score = 0
    for word in SCORE_WORDS_HIGH:
        if word in text_lower:
            score += 3
    for phrase in SCORE_WORDS_MED:
        if phrase in text_lower:
            score += 2
    for word in SCORE_WORDS_LOW:
        if word in text_lower:
            score += 1
    return score


def get_existing_post_ids():
    """Fetch post_ids that are already replied or skipped."""
    url = f"{SUPABASE_URL}/rest/v1/dmhoa_leads"
    params = {
        'select': 'post_id',
        'status': 'in.(replied,skipped)',
    }
    resp = requests.get(url, headers=supabase_headers(), params=params, timeout=10)
    if resp.ok:
        return {row['post_id'] for row in resp.json()}
    return set()


def upsert_lead(lead):
    url = f"{SUPABASE_URL}/rest/v1/dmhoa_leads"
    resp = requests.post(url, headers=supabase_headers(), json=lead, timeout=10)
    if resp.status_code in (200, 201):
        logger.info(f"  Upserted: r/{lead['subreddit']} — {lead['title'][:60]}")
    else:
        logger.error(f"  Failed to upsert {lead['post_id']}: {resp.status_code} {resp.text}")


def fetch_reddit(subreddit, query=None, size=100):
    """Fetch newest posts from Reddit's public JSON. Caller filters by age."""
    if query:
        url = f"{REDDIT_JSON_BASE}/r/{subreddit}/search.json"
        params = {'q': query, 'restrict_sr': 'on', 'sort': 'new', 't': 'week', 'limit': size}
    else:
        url = f"{REDDIT_JSON_BASE}/r/{subreddit}/new.json"
        params = {'limit': size}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if resp.status_code != 200:
            logger.error(f"Reddit returned {resp.status_code} for r/{subreddit}")
            return []

        children = resp.json().get('data', {}).get('children', [])
        return [c.get('data', {}) for c in children]

    except Exception as e:
        logger.error(f"Error fetching r/{subreddit} from Reddit: {e}")
        return []


def scrape():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
        return 0

    skip_ids = get_existing_post_ids()
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=MAX_POST_AGE_DAYS)).timestamp()
    count = 0

    def _build_lead(post, sub_name, score):
        return {
            'post_id': post.get('id', ''),
            'subreddit': sub_name,
            'title': post.get('title', '')[:500],
            'url': f"https://reddit.com/r/{sub_name}/comments/{post.get('id', '')}",
            'score': score,
            'status': 'new',
            'created_utc': datetime.fromtimestamp(
                post.get('created_utc', 0), tz=timezone.utc
            ).isoformat(),
        }

    # HOA-focused subs (r/HOA): every post is HOA. Keyword match + min score.
    for sub_name in HOA_SUBS:
        logger.info(f"Scanning r/{sub_name} (HOA-focused)...")
        posts = fetch_reddit(sub_name, size=100)
        logger.info(f"  Fetched {len(posts)} posts")
        for post in posts:
            if post.get('id', '') in skip_ids:
                continue
            if post.get('created_utc', 0) < cutoff_ts:
                continue
            combined = f"{post.get('title', '')} {post.get('selftext', '')}"
            if not matches_keywords(combined):
                continue
            score = score_post(combined)
            if score < MIN_SCORE_HOA_SUBS:
                continue
            upsert_lead(_build_lead(post, sub_name, score))
            count += 1
        time.sleep(1)

    # Adjacent subs (r/homeowners): require explicit HOA mention + higher score.
    for sub_name in ADJACENT_SUBS:
        logger.info(f"Scanning r/{sub_name} (adjacent — HOA mention required)...")
        posts = fetch_reddit(sub_name, size=100)
        logger.info(f"  Fetched {len(posts)} posts")
        for post in posts:
            if post.get('id', '') in skip_ids:
                continue
            if post.get('created_utc', 0) < cutoff_ts:
                continue
            combined = f"{post.get('title', '')} {post.get('selftext', '')}"
            if not mentions_hoa(combined):
                continue
            if not matches_keywords(combined):
                continue
            score = score_post(combined)
            if score < MIN_SCORE_ADJACENT_SUBS:
                continue
            upsert_lead(_build_lead(post, sub_name, score))
            count += 1
        time.sleep(1)

    # General subs: search "hoa" then require min score.
    for sub_name in GENERAL_SUBS:
        logger.info(f"Scanning r/{sub_name} (searching 'hoa')...")
        posts = fetch_reddit(sub_name, query='hoa', size=50)
        logger.info(f"  Fetched {len(posts)} posts")
        for post in posts:
            if post.get('id', '') in skip_ids:
                continue
            if post.get('created_utc', 0) < cutoff_ts:
                continue
            combined = f"{post.get('title', '')} {post.get('selftext', '')}"
            score = score_post(combined)
            if score < MIN_SCORE_GENERAL_SUBS:
                continue
            upsert_lead(_build_lead(post, sub_name, score))
            count += 1

        time.sleep(1)

    logger.info(f"Scrape complete. {count} new leads found.")
    return count


if __name__ == '__main__':
    scrape()
