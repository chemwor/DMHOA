"""
Flask blueprint for Reddit lead management.
Routes: GET /api/dashboard/leads, PATCH /api/dashboard/leads/:id, POST /api/dashboard/leads/run-scraper
"""

import os
import subprocess
import logging
from datetime import datetime, timezone

import requests as http_requests
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

leads_bp = Blueprint('leads', __name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
TIMEOUT = (5, 60)


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

    try:
        result = subprocess.run(
            ['python', 'scripts/reddit_scraper.py'],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return jsonify({
            'ok': result.returncode == 0,
            'stdout': result.stdout[-2000:] if result.stdout else '',
            'stderr': result.stderr[-500:] if result.stderr else '',
        })
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'error': 'Scraper timed out after 120s'}), 504
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
