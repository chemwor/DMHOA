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
from datetime import datetime, timezone

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---

SUBREDDITS = ['HOA', 'homeowners', 'FirstTimeHomeBuyer', 'personalfinance', 'legaladvice']

KEYWORDS = [
    'hoa fine', 'hoa violation', 'hoa dispute', 'hoa letter', 'hoa fined me',
    "hoa won't", 'hoa ignored', 'hoa threatening', 'appeal hoa', 'fight hoa',
    'hoa not responding', 'hoa complaint', 'dispute hoa',
]

SCORE_WORDS_HIGH = ['fined', 'violation', 'threatened', 'lawsuit']
SCORE_WORDS_MED = ['help', 'advice', 'what do i do', 'what should i']
SCORE_WORDS_LOW = ['lawyer', 'attorney']

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

REDDIT_HEADERS = {
    'User-Agent': 'DMHOA-LeadScraper/1.0 (monitoring HOA subreddits)',
}


def supabase_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates',
    }


def matches_keywords(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORDS)


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
    """Fetch post_ids that are already replied or skipped so we can skip them."""
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
        logger.info(f"  Upserted: {lead['post_id']} (score={lead['score']})")
    else:
        logger.error(f"  Failed to upsert {lead['post_id']}: {resp.status_code} {resp.text}")


def fetch_subreddit_posts(subreddit, limit=100):
    """Fetch recent posts from a subreddit using Reddit's public JSON API."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    params = {'limit': min(limit, 100)}

    try:
        resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
        if resp.status_code == 429:
            logger.warning(f"Rate limited on r/{subreddit}, waiting 10s...")
            time.sleep(10)
            resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)

        if resp.status_code != 200:
            logger.error(f"Reddit returned {resp.status_code} for r/{subreddit}")
            return []

        data = resp.json()
        posts = []
        for child in data.get('data', {}).get('children', []):
            post = child.get('data', {})
            posts.append({
                'id': post.get('id', ''),
                'title': post.get('title', ''),
                'selftext': post.get('selftext', ''),
                'permalink': post.get('permalink', ''),
                'created_utc': post.get('created_utc', 0),
            })
        return posts

    except Exception as e:
        logger.error(f"Error fetching r/{subreddit}: {e}")
        return []


def scrape():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
        return 0

    skip_ids = get_existing_post_ids()
    count = 0

    for sub_name in SUBREDDITS:
        logger.info(f"Scanning r/{sub_name}...")

        posts = fetch_subreddit_posts(sub_name)
        logger.info(f"  Fetched {len(posts)} posts")

        for post in posts:
            if post['id'] in skip_ids:
                continue

            combined_text = f"{post['title']} {post['selftext']}"
            if not matches_keywords(combined_text):
                continue

            lead = {
                'post_id': post['id'],
                'subreddit': sub_name,
                'title': post['title'][:500],
                'url': f"https://reddit.com{post['permalink']}",
                'score': score_post(combined_text),
                'status': 'new',
                'created_utc': datetime.fromtimestamp(post['created_utc'], tz=timezone.utc).isoformat(),
            }
            upsert_lead(lead)
            count += 1

        # Be polite — wait between subreddits to avoid rate limits
        time.sleep(2)

    logger.info(f"Scrape complete. {count} new leads found.")
    return count


if __name__ == '__main__':
    scrape()
