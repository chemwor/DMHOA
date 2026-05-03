"""Generate SEO blog posts directly on the dyno (bypass 30s router limit)."""
import os, json, requests, time
from datetime import datetime
from utils.blog_images import fetch_blog_image

# Inline the generator logic to avoid HTTP roundtrip
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_ROLE_KEY']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']

def call_claude(prompt, system_prompt, max_tokens=4096):
    r = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01',
        },
        json={
            'model': 'claude-sonnet-4-5-20250929',
            'max_tokens': max_tokens,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()['content'][0]['text']

def generate_slug(title):
    import re
    s = re.sub(r'[^a-z0-9\s-]', '', title.lower())
    s = re.sub(r'\s+', '-', s).strip('-')
    return s[:80]

def estimate_read_time(content):
    words = len(content.split())
    return max(1, round(words / 200))

KEYWORDS = [
    'how to fight an hoa fine',
    'hoa appeal letter sample',
    'how to respond to hoa violation notice',
    'hoa fine appeal letter template',
    'how to dispute hoa fine',
]

system_prompt = (
    'You are an expert SEO content writer for DisputeMyHOA, a self-help platform that '
    'drafts professional response letters to HOA violation notices. Write helpful, '
    'authoritative, conversational content for homeowners. Talk like a person who '
    'knows HOAs. No em-dashes. No marketing fluff. Respond with valid JSON only.'
)

for kw in KEYWORDS:
    print(f'\n=== Generating: "{kw}" ===')
    prompt = (
        f'Write a comprehensive blog post optimized to rank for the search keyword: "{kw}"\n\n'
        f'Goals:\n'
        f'- Title naturally includes the keyword (or close variant) and reads as helpful, not salesy\n'
        f'- 1200-1800 words, markdown formatted with H2/H3 headings\n'
        f'- Practical, specific advice. Cover what the searcher actually wants to know.\n'
        f'- Include 2-3 internal links to /start-case (free preview) and /appeal-hoa-fine where contextually relevant. Format as markdown links.\n'
        f'- End with a brief soft CTA to start a free preview\n'
        f'- No legal advice claims; we are a self-help educational tool, not a law firm\n'
        f'- Voice: conversational, second-person, no marketing tropes, no em-dashes, no "100% free"\n\n'
        f'Respond with this JSON:\n'
        f'{{"title": "...", "content": "Full markdown post", "excerpt": "150-200 char summary", '
        f'"seo_title": "SEO title (max 60 chars)", "seo_description": "Meta description (150-160 chars)", '
        f'"seo_keywords": ["keyword1", "keyword2"], '
        f'"image_search_query": "2-4 word query for stock photo"}}'
    )

    try:
        text = call_claude(prompt, system_prompt)
        cleaned = text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned)
    except Exception as e:
        print(f'  Claude failed: {e}')
        continue

    slug = generate_slug(data['title'])
    insert = {
        'title': data['title'],
        'slug': slug,
        'excerpt': data.get('excerpt'),
        'content': data.get('content'),
        'category': 'seo',
        'tags': data.get('seo_keywords', []) + [kw],
        'status': 'published',
        'seo_title': data.get('seo_title'),
        'seo_description': data.get('seo_description'),
        'seo_keywords': data.get('seo_keywords', []),
        'source_article_ids': [],
        'read_time_minutes': estimate_read_time(data.get('content', '')),
        'published_at': datetime.utcnow().isoformat(),
    }

    # Image
    try:
        img = fetch_blog_image(data.get('image_search_query') or kw)
        if img:
            url, alt, credit = img
            insert['image_url'] = url
            insert['image_alt'] = alt
            insert['image_credit'] = credit
    except Exception as e:
        print(f'  image fetch failed: {e}')

    r = requests.post(
        f'{SUPABASE_URL}/rest/v1/blog_posts',
        headers={
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation',
        },
        json=insert,
        timeout=30,
    )
    if r.ok:
        b = r.json()[0]
        print(f'  ok: {b["title"]}')
        print(f'    slug: {b["slug"]}')
        print(f'    image: {"yes" if b.get("image_url") else "NO"}')
        print(f'    read_time: {b.get("read_time_minutes")}min')
        print(f'    https://disputemyhoa.com/blog/post.html?slug={b["slug"]}')
    else:
        print(f'  save failed: {r.status_code} {r.text[:200]}')
    time.sleep(1)
