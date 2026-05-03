"""Fetch a relevant image for a blog post via the Unsplash API.

Returns (image_url, alt_text, credit_html). Falls back gracefully if
the API call fails or no key is configured — the caller should handle
None return values by leaving the post image-less.

Unsplash API docs: https://unsplash.com/documentation
Required attribution: "Photo by <name> on Unsplash" (handled here).
"""
import logging
import os
import random
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY')
UTM = '?utm_source=disputemyhoa&utm_medium=referral'


def fetch_blog_image(query: str) -> Optional[Tuple[str, str, str]]:
    """Search Unsplash for a photo matching `query`.

    Returns (image_url, alt, credit_html) tuple, or None on any failure.
    Caller stores image_url/image_alt on the post and image_credit_html
    can be rendered below the image.
    """
    if not UNSPLASH_ACCESS_KEY or not query:
        return None

    try:
        resp = requests.get(
            'https://api.unsplash.com/search/photos',
            params={
                'query': query,
                'per_page': 10,
                'orientation': 'landscape',
                'content_filter': 'high',
            },
            headers={
                'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}',
                'Accept-Version': 'v1',
            },
            timeout=15,
        )
        if not resp.ok:
            logger.warning(f'Unsplash search failed: {resp.status_code} {resp.text[:200]}')
            return None
        results = resp.json().get('results', []) or []
        if not results:
            logger.info(f'Unsplash: no results for query "{query}"')
            return None

        # Pick from the top half of results so we don't always grab the
        # same photo when the same topic recurs.
        top_n = max(1, min(5, len(results)))
        photo = random.choice(results[:top_n])

        # Use the "regular" size (1080w) for blog hero — fast and sharp
        image_url = photo.get('urls', {}).get('regular')
        if not image_url:
            return None

        # Alt text: prefer Unsplash's alt_description, fall back to query
        alt = (photo.get('alt_description') or photo.get('description') or query).strip()
        # Capitalize first letter
        alt = alt[:1].upper() + alt[1:] if alt else query

        # Required attribution
        photographer = photo.get('user', {}) or {}
        name = photographer.get('name') or 'Unsplash photographer'
        photographer_url = photographer.get('links', {}).get('html', '') or 'https://unsplash.com'
        credit_html = (
            f'Photo by <a href="{photographer_url}{UTM}" target="_blank" rel="noopener">{name}</a> '
            f'on <a href="https://unsplash.com{UTM}" target="_blank" rel="noopener">Unsplash</a>'
        )

        # Best practice: trigger a download tracking event per Unsplash terms
        try:
            dl_endpoint = (photo.get('links', {}) or {}).get('download_location')
            if dl_endpoint:
                requests.get(
                    dl_endpoint,
                    headers={'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'},
                    timeout=5,
                )
        except Exception:
            pass

        return image_url, alt[:200], credit_html

    except Exception as e:
        logger.warning(f'Unsplash fetch exception: {e}')
        return None
