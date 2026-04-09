"""
Reddit Lead Scraper for DMHOA
Monitors HOA-related subreddits for potential leads.
Uses Pullpush API (public Reddit search) — no API key needed, works from cloud IPs.

Usage:
    python scripts/reddit_scraper.py

Requires env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---

PULLPUSH_URL = 'https://api.pullpush.io/reddit/search/submission/'

# HOA-focused subs: fetch all recent posts (they're all relevant)
HOA_SUBS = ['HOA', 'homeowners']

# General subs: search for HOA-related posts via keyword
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


def fetch_pullpush(subreddit, query=None, size=100):
    """Fetch posts from Pullpush API."""
    params = {
        'subreddit': subreddit,
        'size': size,
        'sort': 'desc',
        'sort_type': 'created_utc',
    }
    if query:
        params['q'] = query

    try:
        resp = requests.get(PULLPUSH_URL, headers=HEADERS, params=params, timeout=20)
        if resp.status_code != 200:
            logger.error(f"Pullpush returned {resp.status_code} for r/{subreddit}")
            return []

        data = resp.json()
        return data.get('data', [])

    except Exception as e:
        logger.error(f"Error fetching r/{subreddit} from Pullpush: {e}")
        return []


def scrape():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
        return 0

    skip_ids = get_existing_post_ids()
    count = 0

    # HOA-focused subs: get all recent posts, filter by keywords
    for sub_name in HOA_SUBS:
        logger.info(f"Scanning r/{sub_name} (all recent posts)...")
        posts = fetch_pullpush(sub_name, size=100)
        logger.info(f"  Fetched {len(posts)} posts")

        for post in posts:
            post_id = post.get('id', '')
            if post_id in skip_ids:
                continue

            combined = f"{post.get('title', '')} {post.get('selftext', '')}"
            if not matches_keywords(combined):
                continue

            lead = {
                'post_id': post_id,
                'subreddit': sub_name,
                'title': post.get('title', '')[:500],
                'url': f"https://reddit.com/r/{sub_name}/comments/{post_id}",
                'score': score_post(combined),
                'status': 'new',
                'created_utc': datetime.fromtimestamp(
                    post.get('created_utc', 0), tz=timezone.utc
                ).isoformat(),
            }
            upsert_lead(lead)
            count += 1

        time.sleep(1)

    # General subs: search for "hoa" keyword
    for sub_name in GENERAL_SUBS:
        logger.info(f"Scanning r/{sub_name} (searching 'hoa')...")
        posts = fetch_pullpush(sub_name, query='hoa', size=50)
        logger.info(f"  Fetched {len(posts)} posts")

        for post in posts:
            post_id = post.get('id', '')
            if post_id in skip_ids:
                continue

            combined = f"{post.get('title', '')} {post.get('selftext', '')}"
            lead = {
                'post_id': post_id,
                'subreddit': sub_name,
                'title': post.get('title', '')[:500],
                'url': f"https://reddit.com/r/{sub_name}/comments/{post_id}",
                'score': score_post(combined),
                'status': 'new',
                'created_utc': datetime.fromtimestamp(
                    post.get('created_utc', 0), tz=timezone.utc
                ).isoformat(),
            }
            upsert_lead(lead)
            count += 1

        time.sleep(1)

    logger.info(f"Scrape complete. {count} new leads found.")
    return count


if __name__ == '__main__':
    scrape()
