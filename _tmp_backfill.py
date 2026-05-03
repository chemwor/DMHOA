"""Backfill Unsplash images on existing image-less blog posts."""
import os, requests
from utils.blog_images import fetch_blog_image

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_ROLE_KEY']
H_PATCH = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}',
    'Content-Type': 'application/json',
    'Prefer': 'return=minimal',
}

def query_for(post):
    title = (post.get('title') or '').lower()
    if 'architectural' in title or 'paint' in title:
        return 'suburban house front yard'
    if 'fence' in title or 'shed' in title:
        return 'backyard fence shed'
    if 'parking' in title:
        return 'residential street parked cars'
    if 'financial' in title or 'records' in title or 'assessment' in title:
        return 'home finance documents'
    if 'deadline' in title:
        return 'calendar reminder mail'
    if 'violation' in title or 'notice' in title or 'letter' in title:
        return 'mailbox letter envelope'
    if 'rights' in title or 'law' in title:
        return 'law book gavel'
    if 'red flag' in title or 'buying' in title:
        return 'suburban neighborhood houses'
    if 'solar' in title:
        return 'solar panels suburban roof'
    if 'amenity' in title or 'pool' in title or 'gym' in title:
        return 'community pool suburb'
    if 'technology' in title:
        return 'smart home technology'
    return 'suburban neighborhood homes'


resp = requests.get(
    f'{SUPABASE_URL}/rest/v1/blog_posts',
    params={
        'select': 'id,slug,title,category',
        'image_url': 'is.null',
        'order': 'published_at.desc',
    },
    headers={'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'},
    timeout=30,
)
posts = resp.json() if resp.ok else []
print(f'Posts to backfill: {len(posts)}')

ok = fail = 0
for post in posts:
    q = query_for(post)
    slug = post['slug']
    print(f'  {slug[:45]:<45}  query: {q}')
    img = fetch_blog_image(q)
    if not img:
        fail += 1
        print('    no image returned')
        continue
    url, alt, credit = img
    upd = requests.patch(
        f'{SUPABASE_URL}/rest/v1/blog_posts',
        params={'id': f'eq.{post["id"]}'},
        headers=H_PATCH,
        json={'image_url': url, 'image_alt': alt, 'image_credit': credit},
        timeout=15,
    )
    if upd.ok:
        ok += 1
        print(f'    ok: {url[:60]}')
    else:
        fail += 1
        print(f'    patch fail {upd.status_code}: {upd.text[:120]}')

print(f'\nDone. Updated {ok}, failed {fail}')
