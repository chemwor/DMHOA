"""
Firecrawl integration. Scrapes a URL and returns clean markdown content
that's ready to pass to an LLM.

Used by the Reddit Reply Drafter to fetch the full text of a Reddit post
(including all comments) before asking Claude to draft a reply.

Fire and forget on failure. Never blocks the caller. Returns empty string
on any error so the caller can still proceed without scraped context.
"""

import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.environ.get('FIRECRAWL_API_KEY')
FIRECRAWL_API_URL = 'https://api.firecrawl.dev/v1/scrape'

TIMEOUT = (5, 30)


def scrape_url(url: str, formats: Optional[list] = None) -> str:
    """Scrape a URL via Firecrawl and return the markdown content.

    Returns empty string on any failure (missing API key, timeout, bad URL,
    Firecrawl downtime, etc). Never raises.
    """
    if not FIRECRAWL_API_KEY:
        logger.warning('FIRECRAWL_API_KEY not configured, skipping scrape')
        return ''

    if not url:
        return ''

    payload = {
        'url': url,
        'formats': formats or ['markdown'],
    }

    try:
        resp = requests.post(
            FIRECRAWL_API_URL,
            headers={
                'Authorization': f'Bearer {FIRECRAWL_API_KEY}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=TIMEOUT,
        )

        if resp.status_code in (200, 201):
            data = resp.json()
            if data.get('success'):
                md = data.get('data', {}).get('markdown', '')
                if md:
                    logger.info(f'Firecrawl: scraped {url} ({len(md)} chars)')
                    return md
            logger.warning(f'Firecrawl: scrape returned success=false for {url}')
            return ''

        logger.warning(f'Firecrawl: HTTP {resp.status_code} for {url} - {resp.text[:200]}')
        return ''

    except requests.exceptions.Timeout:
        logger.warning(f'Firecrawl: timeout scraping {url}')
        return ''
    except Exception as e:
        logger.warning(f'Firecrawl: exception scraping {url}: {e}')
        return ''
