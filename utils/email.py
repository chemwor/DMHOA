"""
Resend email sender. Plain text only. No HTML, no buttons, no decoration.

Used by the email funnel to send transactional and nudge emails. All emails
come from "Eric from Dispute My HOA <eric@mail.disputemyhoa.com>".

If RESEND_API_KEY is not configured, send_email logs a warning and returns
False without raising. Email failures never block the user flow.
"""

import os
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.environ.get('RESEND_API_KEY')
RESEND_API_URL = 'https://api.resend.com/emails'

FROM_ADDRESS = 'Eric from Dispute My HOA <eric@mail.disputemyhoa.com>'
REPLY_TO_ADDRESS = 'eric@disputemyhoa.com'

TIMEOUT = (5, 15)  # connect, read


def send_email(to: str, subject: str, body_text: str) -> bool:
    """Send a plain-text email via Resend.

    Returns True on success, False otherwise. Never raises.
    """
    if not RESEND_API_KEY:
        logger.warning(f'RESEND_API_KEY not configured, skipping email to {to}')
        return False

    if not to or '@' not in to:
        logger.warning(f'send_email called with invalid recipient: {to!r}')
        return False

    payload = {
        'from': FROM_ADDRESS,
        'to': [to],
        'reply_to': REPLY_TO_ADDRESS,
        'subject': subject,
        'text': body_text,
    }

    try:
        response = requests.post(
            RESEND_API_URL,
            headers={
                'Authorization': f'Bearer {RESEND_API_KEY}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=TIMEOUT,
        )

        if response.status_code in (200, 201, 202):
            logger.info(f'Resend: sent email to {to} (subject: {subject!r})')
            return True

        logger.error(
            f'Resend send failed for {to}: HTTP {response.status_code} - {response.text[:300]}'
        )
        return False

    except requests.exceptions.Timeout:
        logger.error(f'Resend timeout sending to {to}')
        return False
    except Exception as e:
        logger.error(f'Resend exception sending to {to}: {e}')
        return False
