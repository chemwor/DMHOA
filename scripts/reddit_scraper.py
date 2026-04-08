"""
Reddit Lead Scraper for DMHOA
Monitors HOA-related subreddits for potential leads.
Run manually or via Heroku Scheduler.

Usage:
    python scripts/reddit_scraper.py

Requires env vars: PRAW_CLIENT_ID, PRAW_CLIENT_SECRET, PRAW_USER_AGENT,
                   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""

import os
import re
import logging
from datetime import datetime, timezone

import praw
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---

SUBREDDITS = ['HOA', 'homeowners', 'FirstTimeHomeBuyer', 'personalfinance', 'legaladvice']

KEYWORDS = [
    'hoa fine', 'hoa violation', 'hoa dispute', 'hoa letter', 'hoa fined me',
    'hoa won\'t', 'hoa ignored', 'hoa threatening', 'appeal hoa', 'fight hoa',
    'hoa not responding', 'hoa complaint', 'dispute hoa',
]

SCORE_WORDS_HIGH = ['fined', 'violation', 'threatened', 'lawsuit']
SCORE_WORDS_MED = ['help', 'advice', 'what do i do', 'what should i']
SCORE_WORDS_LOW = ['lawyer', 'attorney']

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')


def supabase_headers():
    return {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates',
    }


def get_reddit():
    return praw.Reddit(
        client_id=os.environ.get('PRAW_CLIENT_ID'),
        client_secret=os.environ.get('PRAW_CLIENT_SECRET'),
        user_agent=os.environ.get('PRAW_USER_AGENT', 'dmhoa-lead-scraper/1.0'),
    )


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


def scrape():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
        return 0

    reddit = get_reddit()
    skip_ids = get_existing_post_ids()
    count = 0

    for sub_name in SUBREDDITS:
        logger.info(f"Scanning r/{sub_name}...")
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.new(limit=100):
                if post.id in skip_ids:
                    continue

                combined_text = f"{post.title} {post.selftext or ''}"
                if not matches_keywords(combined_text):
                    continue

                lead = {
                    'post_id': post.id,
                    'subreddit': sub_name,
                    'title': post.title[:500],
                    'url': f"https://reddit.com{post.permalink}",
                    'score': score_post(combined_text),
                    'status': 'new',
                    'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                }
                upsert_lead(lead)
                count += 1

        except Exception as e:
            logger.error(f"Error scanning r/{sub_name}: {e}")

    logger.info(f"Scrape complete. {count} new leads found.")
    return count


if __name__ == '__main__':
    scrape()
