"""
Test pipeline for the DMHOA email funnel + case analysis.

Lets the dashboard create fake test cases, advance them through the funnel
stages, manually trigger the nudge runner, and generate the full Claude
analysis ("plan") on demand. Every test row is tagged so cleanup is safe
and deterministic.

Tagging conventions (used by cleanup):
  - Test case tokens always start with "case_test_"
  - Test cases also have payload.test_run = true

All endpoints live under /api/dashboard/test-funnel.
"""

import os
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from flask import Blueprint, request, jsonify

from utils.funnel import log_funnel_stage
from utils import email_templates
from utils.email import send_email

logger = logging.getLogger(__name__)

test_funnel_bp = Blueprint('test_funnel', __name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
TIMEOUT = (5, 60)

TEST_TOKEN_PREFIX = 'case_test_'


def supabase_headers() -> Dict[str, str]:
    return {
        'apikey': SUPABASE_SERVICE_ROLE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE_KEY}',
        'Content-Type': 'application/json',
    }


def _is_test_token(token: str) -> bool:
    return bool(token) and token.startswith(TEST_TOKEN_PREFIX)


def _generate_test_token() -> str:
    """Build a unique tagged token for a test run."""
    ts = int(datetime.now(timezone.utc).timestamp())
    suffix = secrets.token_hex(4)
    return f'{TEST_TOKEN_PREFIX}{ts}_{suffix}'


def _delete_supabase_rows(table: str, params: dict) -> int:
    """Delete rows from a Supabase table by query params. Returns row count or -1 on error."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return -1
    try:
        headers = {**supabase_headers(), 'Prefer': 'return=representation'}
        resp = requests.delete(
            f'{SUPABASE_URL}/rest/v1/{table}',
            headers=headers,
            params=params,
            timeout=TIMEOUT,
        )
        if resp.ok:
            try:
                return len(resp.json() or [])
            except Exception:
                return 0
        logger.warning(f'Delete {table} failed: HTTP {resp.status_code} - {resp.text[:200]}')
        return -1
    except Exception as e:
        logger.error(f'Delete {table} exception: {e}')
        return -1


# ----------------------------------------------------------------------------
# CREATE A TEST CASE
# ----------------------------------------------------------------------------

SAMPLE_HOA_NOTICE = """NOTICE OF VIOLATION

Date: April 1, 2026

Dear Homeowner,

The Sunset Ridge Homeowners Association has received a complaint regarding your property at 1247 Maple Lane. Upon inspection on March 28, 2026, the following violations of our Community Guidelines were observed:

1. Trash receptacles left visible from the street outside of approved collection windows (Section 4.2.1 of the CC&Rs)
2. Garage door left open in excess of 30 minutes (Section 4.5.3)
3. Exterior light fixture not approved by the Architectural Review Committee (Section 6.1)

You are hereby notified that fines totaling $750 will be assessed if these violations are not corrected within 14 days of the date of this notice. Failure to comply may result in additional fines, suspension of community amenities, and possible legal action including a lien against your property.

You have the right to appeal this notice in writing within 14 days. Appeals must be addressed to the Board of Directors and submitted via certified mail.

Sincerely,
Sunset Ridge HOA
Board of Directors
"""


@test_funnel_bp.route('/api/dashboard/test-funnel/create', methods=['POST', 'OPTIONS'])
def create_test_case():
    """Create a tagged test case and immediately advance funnel to quick_preview_complete.

    Body: { "email": "test@example.com", "skip_preview": false (optional) }

    By default this also kicks off real OpenAI preview generation in a background
    thread so the case-preview.html page renders properly with realistic content.
    Pass skip_preview: true to skip the AI call (faster, no token cost).
    """
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500

    body = request.get_json(silent=True) or {}
    email = (body.get('email') or '').strip()
    skip_preview = bool(body.get('skip_preview'))
    if not email or '@' not in email:
        return jsonify({'error': 'Valid email required'}), 400

    token = _generate_test_token()

    # Build a realistic case payload that mirrors what /api/save-case stores
    # for a real customer. Includes pasted HOA notice text so the OpenAI
    # preview generator has something substantive to summarize.
    payload = {
        'email': email,
        'test_run': True,
        'token': token,
        'role': 'homeowner',
        'state': 'GA',
        'propertyType': 'single-family',
        'noticeType': 'violation',
        'outcome': 'reverse-fine',
        'issueText': 'Got a violation notice for trash cans visible from the street, an open garage door, and an unapproved exterior light. They are claiming $750 in fines if I do not respond in 14 days. The trash cans were only out for 20 minutes after pickup and the light was approved by the previous board.',
        'pastedText': SAMPLE_HOA_NOTICE,
        'amount': 750,
        'deadline': None,
        'createdAt': datetime.now(timezone.utc).isoformat(),
        'submittedAt': datetime.now(timezone.utc).isoformat(),
        'fullCaseFormLink': f'https://disputemyhoa.com/start-case-full.html?case={token}',
        'extract_status': 'pending',
        'additional_docs': [],
    }

    case_data = {
        'token': token,
        'email': email,
        'payload': payload,
        'status': 'quick_preview',
    }

    try:
        resp = requests.post(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers={**supabase_headers(), 'Prefer': 'return=representation'},
            json=case_data,
            timeout=TIMEOUT,
        )
        if not resp.ok:
            return jsonify({
                'error': 'Failed to insert test case',
                'detail': resp.text[:500],
            }), 500
        rows = resp.json() or []
        case_row = rows[0] if rows else case_data
        case_id = case_row.get('id')
    except Exception as e:
        logger.error(f'Test case create failed: {e}')
        return jsonify({'error': str(e)}), 500

    # Fire the funnel stage hook (this also sends the immediate quick_preview email)
    funnel_advanced = log_funnel_stage(email, 'quick_preview_complete', payload['fullCaseFormLink'])

    # Kick off real preview generation in the background so the case-preview.html
    # page on the public site renders with actual AI content. This is what makes
    # the test feel like a real customer flow.
    if not skip_preview and case_id:
        def _gen_preview():
            try:
                logger.info(f'Test pipeline: starting preview generation for {token}')
                # Lazy import to avoid circular dependency
                from app import generate_preview_for_pasted_text
                generate_preview_for_pasted_text(case_id, token, SAMPLE_HOA_NOTICE, payload)
                logger.info(f'Test pipeline: preview generation complete for {token}')
            except Exception as e:
                logger.error(f'Test pipeline: preview generation failed for {token}: {e}')

        try:
            t = threading.Thread(target=_gen_preview, daemon=True)
            t.start()
        except Exception as e:
            logger.warning(f'Could not start preview thread: {e}')

    return jsonify({
        'ok': True,
        'token': token,
        'email': email,
        'case': {
            'id': case_id,
            'token': token,
            'status': case_row.get('status', 'quick_preview'),
        },
        'funnel_advanced': funnel_advanced,
        'preview_generation_started': not skip_preview,
        'message': 'Test case created. Quick preview email sent.' + (
            ' AI preview rendering in background (~30s).' if not skip_preview else ''
        ),
    })


# ----------------------------------------------------------------------------
# ADVANCE A TEST CASE TO THE NEXT STAGE
# ----------------------------------------------------------------------------

@test_funnel_bp.route('/api/dashboard/test-funnel/<token>/advance', methods=['POST', 'OPTIONS'])
def advance_test_case(token):
    """Advance a test case to the named stage.

    Body: { "stage": "full_preview_viewed" | "purchased" }
    Only test-tagged tokens are accepted.
    """
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not _is_test_token(token):
        return jsonify({'error': 'Refusing to advance a non-test token'}), 400

    body = request.get_json(silent=True) or {}
    target_stage = (body.get('stage') or '').strip()
    if target_stage not in ('full_preview_viewed', 'purchased'):
        return jsonify({'error': 'stage must be full_preview_viewed or purchased'}), 400

    # Look up the case
    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers=supabase_headers(),
            params={'token': f'eq.{token}', 'select': 'id,token,email,payload,status'},
            timeout=TIMEOUT,
        )
        if not resp.ok:
            return jsonify({'error': 'Failed to fetch test case', 'detail': resp.text[:300]}), 500
        rows = resp.json() or []
        if not rows:
            return jsonify({'error': 'Test case not found'}), 404
        case = rows[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    case_payload = case.get('payload') or {}
    if isinstance(case_payload, str):
        try:
            case_payload = json.loads(case_payload)
        except Exception:
            case_payload = {}

    email = case.get('email') or case_payload.get('email')
    if not email:
        return jsonify({'error': 'Test case has no email'}), 400

    # Build the link passed into the email template
    if target_stage == 'full_preview_viewed':
        # nudge_2 (sent later) reminds them about the full preview they viewed
        link = f'https://disputemyhoa.com/case-preview.html?case={token}'
    else:  # purchased
        link = f'https://disputemyhoa.com/case.html?case={token}'
        # Mark the case as paid so the rest of the dashboard sees it correctly
        try:
            requests.patch(
                f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
                headers=supabase_headers(),
                params={'token': f'eq.{token}'},
                json={'status': 'paid'},
                timeout=TIMEOUT,
            )
        except Exception as e:
            logger.warning(f'Failed to mark test case paid: {e}')

    funnel_advanced = log_funnel_stage(email, target_stage, link)

    return jsonify({
        'ok': True,
        'token': token,
        'email': email,
        'stage': target_stage,
        'funnel_advanced': funnel_advanced,
        'message': f'Advanced to {target_stage}' if funnel_advanced else 'Already at or past this stage',
    })


# ----------------------------------------------------------------------------
# GENERATE FULL PLAN (CASE ANALYSIS) FOR A TEST CASE
# ----------------------------------------------------------------------------

@test_funnel_bp.route('/api/dashboard/test-funnel/<token>/generate-plan', methods=['POST', 'OPTIONS'])
def generate_test_plan(token):
    """Run the full Claude case analysis for a test case in a background thread.

    The dashboard polls /case-analysis-status to see when it completes.
    """
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not _is_test_token(token):
        return jsonify({'error': 'Refusing to generate plan for a non-test token'}), 400

    # Lazy import to avoid circular dependency at module load
    try:
        from app import run_case_analysis
    except Exception as e:
        return jsonify({'error': f'Could not import run_case_analysis: {e}'}), 500

    def _run():
        try:
            logger.info(f'Test pipeline: starting plan generation for {token}')
            result = run_case_analysis(token)
            logger.info(f'Test pipeline: plan generation complete for {token}: status={result.get("status") if isinstance(result, dict) else "unknown"}')
        except Exception as e:
            logger.error(f'Test pipeline: plan generation failed for {token}: {e}')

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

    return jsonify({
        'ok': True,
        'token': token,
        'message': 'Plan generation started in background. Check status in 30-60 seconds.',
    })


@test_funnel_bp.route('/api/dashboard/test-funnel/<token>/regenerate-preview', methods=['POST', 'OPTIONS'])
def regenerate_test_preview(token):
    """Regenerate the AI preview for an existing test case in a background thread."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not _is_test_token(token):
        return jsonify({'error': 'Refusing to regenerate preview for a non-test token'}), 400

    # Look up case to get id and pasted text
    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers=supabase_headers(),
            params={'token': f'eq.{token}', 'select': 'id,payload'},
            timeout=TIMEOUT,
        )
        if not resp.ok or not resp.json():
            return jsonify({'error': 'Test case not found'}), 404
        case = resp.json()[0]
        case_id = case.get('id')
        case_payload = case.get('payload') or {}
        if isinstance(case_payload, str):
            try:
                case_payload = json.loads(case_payload)
            except Exception:
                case_payload = {}
        pasted = case_payload.get('pastedText') or SAMPLE_HOA_NOTICE
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    def _gen():
        try:
            from app import generate_preview_for_pasted_text
            generate_preview_for_pasted_text(case_id, token, pasted, case_payload)
            logger.info(f'Test pipeline: preview regenerated for {token}')
        except Exception as e:
            logger.error(f'Test pipeline: regenerate preview failed for {token}: {e}')

    try:
        t = threading.Thread(target=_gen, daemon=True)
        t.start()
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

    return jsonify({
        'ok': True,
        'token': token,
        'message': 'Preview regeneration started in background. Refresh the case-preview page in ~30 seconds.',
    })


@test_funnel_bp.route('/api/dashboard/test-funnel/<token>/plan-status', methods=['GET', 'OPTIONS'])
def get_plan_status(token):
    """Return the current case analysis status for a test case."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not _is_test_token(token):
        return jsonify({'error': 'Refusing non-test token'}), 400

    try:
        # Look up the case to get id
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers=supabase_headers(),
            params={'token': f'eq.{token}', 'select': 'id'},
            timeout=TIMEOUT,
        )
        if not resp.ok or not resp.json():
            return jsonify({'error': 'Test case not found'}), 404
        case_id = resp.json()[0].get('id')

        # Check dmhoa_case_outputs for the analysis result
        out_resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_case_outputs',
            headers=supabase_headers(),
            params={'case_id': f'eq.{case_id}', 'select': 'id,status,model,created_at', 'order': 'created_at.desc', 'limit': '1'},
            timeout=TIMEOUT,
        )
        if out_resp.ok:
            outs = out_resp.json() or []
            if outs:
                return jsonify({'has_plan': True, 'output': outs[0]})

        return jsonify({'has_plan': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------------
# MANUAL NUDGE RUN
# ----------------------------------------------------------------------------

@test_funnel_bp.route('/api/dashboard/test-funnel/run-nudges', methods=['POST', 'OPTIONS'])
def trigger_nudges_now():
    """Run the email nudge job immediately. Same logic as the 30-min scheduler."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        from jobs.email_nudges import _run_nudge_1, _run_nudge_2, _run_nudge_3
        n1 = _run_nudge_1()
        n2 = _run_nudge_2()
        n3 = _run_nudge_3()
        return jsonify({
            'ok': True,
            'nudge_1_sent': n1,
            'nudge_2_sent': n2,
            'nudge_3_sent': n3,
        })
    except Exception as e:
        logger.error(f'trigger_nudges_now failed: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ----------------------------------------------------------------------------
# EMAIL TEMPLATE PREVIEW + STANDALONE TEST SEND
# Lets you inspect or send a single email of any template without touching
# the funnel state. Useful for tweaking copy and verifying deliverability.
# ----------------------------------------------------------------------------

EMAIL_TEMPLATES = {
    'quick_preview_confirmation': email_templates.quick_preview_confirmation,
    'nudge_1': email_templates.nudge_1,
    'nudge_2': email_templates.nudge_2,
    'nudge_3': email_templates.nudge_3,
    'purchase_confirmation': email_templates.purchase_confirmation,
}


@test_funnel_bp.route('/api/dashboard/test-funnel/email-preview/<template_name>', methods=['GET', 'OPTIONS'])
def preview_email_template(template_name):
    """Render any email template and return subject + body without sending.

    Optional query param: link=<url> to substitute into the template.
    """
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if template_name not in EMAIL_TEMPLATES:
        return jsonify({
            'error': f'Unknown template: {template_name}',
            'available': sorted(EMAIL_TEMPLATES.keys()),
        }), 400

    link = request.args.get('link', 'https://disputemyhoa.com/case-preview/example')
    try:
        subject, body = EMAIL_TEMPLATES[template_name](link)
        return jsonify({
            'template': template_name,
            'subject': subject,
            'body': body,
            'link_used': link,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@test_funnel_bp.route('/api/dashboard/test-funnel/send-test-email', methods=['POST', 'OPTIONS'])
def send_test_email():
    """Send a one-off email of any template to a specific address.

    Body: { "template": "nudge_1", "to": "you@example.com", "link": "..." (optional) }
    Does not touch the email_funnel table or any case data.
    """
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    body = request.get_json(silent=True) or {}
    template_name = (body.get('template') or '').strip()
    to_address = (body.get('to') or '').strip()
    link = (body.get('link') or 'https://disputemyhoa.com/case-preview/example').strip()

    if template_name not in EMAIL_TEMPLATES:
        return jsonify({
            'error': f'Unknown template: {template_name}',
            'available': sorted(EMAIL_TEMPLATES.keys()),
        }), 400

    if not to_address or '@' not in to_address:
        return jsonify({'error': 'Valid "to" email address required'}), 400

    try:
        subject, body_text = EMAIL_TEMPLATES[template_name](link)
        sent = send_email(to_address, subject, body_text)
        if sent:
            return jsonify({
                'ok': True,
                'message': f'Sent {template_name} to {to_address}',
                'subject': subject,
            })
        else:
            return jsonify({
                'ok': False,
                'error': 'Resend send returned non-success. Check Heroku logs and verify mail.disputemyhoa.com is configured in Resend.',
            }), 500
    except Exception as e:
        logger.error(f'send_test_email failed: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500


# ----------------------------------------------------------------------------
# LIST TEST CASES
# ----------------------------------------------------------------------------

@test_funnel_bp.route('/api/dashboard/test-funnel', methods=['GET', 'OPTIONS'])
def list_test_cases():
    """Return all tagged test cases with their funnel state."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        # Find all dmhoa_cases with test prefix
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers=supabase_headers(),
            params={
                'token': f'like.{TEST_TOKEN_PREFIX}*',
                'select': 'id,token,email,payload,status,created_at',
                'order': 'created_at.desc',
                'limit': '100',
            },
            timeout=TIMEOUT,
        )
        if not resp.ok:
            return jsonify({'error': 'Failed to list test cases', 'detail': resp.text[:300]}), 500
        cases = resp.json() or []

        # For each case, look up funnel state and analysis status
        results = []
        for case in cases:
            case_payload = case.get('payload') or {}
            if isinstance(case_payload, str):
                try:
                    case_payload = json.loads(case_payload)
                except Exception:
                    case_payload = {}
            email = case.get('email') or case_payload.get('email')

            funnel_row = None
            if email:
                try:
                    f_resp = requests.get(
                        f'{SUPABASE_URL}/rest/v1/email_funnel',
                        headers=supabase_headers(),
                        params={'email': f'eq.{email}', 'select': '*', 'limit': '1'},
                        timeout=TIMEOUT,
                    )
                    if f_resp.ok:
                        rows = f_resp.json() or []
                        if rows:
                            funnel_row = rows[0]
                except Exception:
                    pass

            has_plan = False
            try:
                out_resp = requests.get(
                    f'{SUPABASE_URL}/rest/v1/dmhoa_case_outputs',
                    headers=supabase_headers(),
                    params={'case_id': f'eq.{case["id"]}', 'select': 'id', 'limit': '1'},
                    timeout=TIMEOUT,
                )
                if out_resp.ok and out_resp.json():
                    has_plan = True
            except Exception:
                pass

            results.append({
                'id': case.get('id'),
                'token': case.get('token'),
                'email': email,
                'status': case.get('status'),
                'created_at': case.get('created_at'),
                'has_plan': has_plan,
                'funnel': funnel_row,
            })

        return jsonify({'cases': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------------
# DELETE A SINGLE TEST CASE (cascade across all related tables)
# ----------------------------------------------------------------------------

def _delete_test_case_artifacts(token: str, email: Optional[str], case_id: Optional[str]) -> Dict[str, int]:
    """Delete every Supabase row tied to a single test case."""
    deleted = {}
    if case_id:
        deleted['dmhoa_case_outputs'] = _delete_supabase_rows('dmhoa_case_outputs', {'case_id': f'eq.{case_id}'})
        deleted['dmhoa_case_previews'] = _delete_supabase_rows('dmhoa_case_previews', {'case_id': f'eq.{case_id}'})
        deleted['dmhoa_documents'] = _delete_supabase_rows('dmhoa_documents', {'case_id': f'eq.{case_id}'})
    deleted['dmhoa_messages'] = _delete_supabase_rows('dmhoa_messages', {'token': f'eq.{token}'})
    deleted['dmhoa_events'] = _delete_supabase_rows('dmhoa_events', {'token': f'eq.{token}'})
    deleted['dmhoa_cases'] = _delete_supabase_rows('dmhoa_cases', {'token': f'eq.{token}'})
    if email:
        deleted['email_funnel'] = _delete_supabase_rows('email_funnel', {'email': f'eq.{email}'})
    return deleted


@test_funnel_bp.route('/api/dashboard/test-funnel/<token>', methods=['DELETE', 'OPTIONS'])
def delete_test_case(token):
    """Delete a single test case and every Supabase row tied to it."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    if not _is_test_token(token):
        return jsonify({'error': 'Refusing to delete a non-test token'}), 400

    # Look up first to get id and email
    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers=supabase_headers(),
            params={'token': f'eq.{token}', 'select': 'id,email,payload'},
            timeout=TIMEOUT,
        )
        rows = resp.json() if resp.ok else []
        case = rows[0] if rows else None
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    case_id = None
    email = None
    if case:
        case_id = case.get('id')
        email = case.get('email')
        if not email:
            cp = case.get('payload') or {}
            if isinstance(cp, str):
                try:
                    cp = json.loads(cp)
                except Exception:
                    cp = {}
            email = cp.get('email')

    deleted = _delete_test_case_artifacts(token, email, case_id)
    return jsonify({'ok': True, 'token': token, 'deleted': deleted})


# ----------------------------------------------------------------------------
# DELETE ALL TEST CASES (bulk cleanup)
# ----------------------------------------------------------------------------

@test_funnel_bp.route('/api/dashboard/test-funnel', methods=['DELETE', 'OPTIONS'])
def delete_all_test_cases():
    """Delete every test-tagged case and every related Supabase row."""
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'})

    try:
        resp = requests.get(
            f'{SUPABASE_URL}/rest/v1/dmhoa_cases',
            headers=supabase_headers(),
            params={'token': f'like.{TEST_TOKEN_PREFIX}*', 'select': 'id,token,email,payload', 'limit': '500'},
            timeout=TIMEOUT,
        )
        cases = resp.json() if resp.ok else []
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    summary = {'cases_deleted': 0, 'tables': {}}
    for case in cases:
        token = case.get('token')
        case_id = case.get('id')
        email = case.get('email')
        if not email:
            cp = case.get('payload') or {}
            if isinstance(cp, str):
                try:
                    cp = json.loads(cp)
                except Exception:
                    cp = {}
            email = cp.get('email')

        deleted = _delete_test_case_artifacts(token, email, case_id)
        for table, count in deleted.items():
            summary['tables'][table] = summary['tables'].get(table, 0) + max(count, 0)
        summary['cases_deleted'] += 1

    return jsonify({'ok': True, **summary})
