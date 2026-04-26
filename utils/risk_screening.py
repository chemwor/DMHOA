"""Risk screening for HOA notices.

Scans pasted notice text for markers that indicate the dispute is beyond
the reasonable scope of a self-help letter template — e.g. opposing counsel
already involved, lien/foreclosure already in motion, animal-bite or
injury exposure, active litigation, criminal involvement.

When any HIGH-severity flag fires, the case-preview UI hides the unlock
CTA and routes the user to a legal-referral form instead.
"""
import re
from typing import Dict, List


# Each rule: a flag key, a user-facing label, and a list of regex patterns.
# Patterns are matched case-insensitively. Use word boundaries (\b) where
# false positives on substrings are likely (e.g. 'lien' would otherwise
# match 'client').
RISK_RULES = [
    {
        'flag': 'opposing_counsel',
        'label': 'The other side is already represented by an attorney',
        'patterns': [
            r'\bas counsel\b', r'\battorney for\b', r'\bcounsel for\b',
            r'\bour attorney\b', r'\btheir attorney\b', r'\blaw firm\b',
            r'\bp\.?l\.?l\.?c\b', r'\besq\.?', r'\bllp\b',
            r'\brepresented by counsel\b',
        ],
    },
    {
        'flag': 'lien_or_foreclosure',
        'label': 'Lien or foreclosure has been mentioned or threatened',
        'patterns': [
            r'\blien\b', r'\bforeclos', r'\bjudicial sale\b',
            r'\bnotice of default\b', r'\btrustee[\'’]?s sale\b',
        ],
    },
    {
        'flag': 'active_litigation',
        'label': 'Active or imminent litigation (lawsuit, summons, court date)',
        'patterns': [
            r'\blawsuit\b', r'\bcomplaint filed\b', r'\bsummons\b',
            r'\bsubpoena\b', r'\bcourt date\b', r'\bcivil action\b',
            r'\bserved with\b', r'\bsuperior court\b', r'\bdistrict court\b',
        ],
    },
    {
        'flag': 'animal_removal_or_bite',
        'label': 'Animal removal demand or bite/attack incident',
        'patterns': [
            r'\bdog bite\b', r'\bdog attack\b', r'\bbit my\b',
            r'\banimal attack\b', r'\bbite incident\b',
            r'\bremove the (dog|cat|pet|animal)\b',
            r'\b(permanent(ly)?|forever) (remove|removal)\b.{0,30}\b(dog|cat|pet|animal)\b',
            r'\b(dog|cat|pet|animal) .{0,30}\bremove(d|al)\b',
        ],
    },
    {
        'flag': 'injury_or_medical',
        'label': 'Injury, medical bills, or personal injury exposure',
        'patterns': [
            r'\binjur(y|ies|ed)\b', r'\bmedical bills?\b',
            r'\bhospital\b', r'\bambulance\b', r'\bemergency room\b',
            r'\bER visit\b',
        ],
    },
    {
        'flag': 'criminal_involvement',
        'label': 'Police, criminal, or assault language',
        'patterns': [
            r'\bpolice report\b', r'\barrest(ed)?\b',
            r'\bcriminal\b', r'\bassault\b', r'\bbattery\b',
            r'\bcharged with\b',
        ],
    },
    {
        'flag': 'large_money_at_stake',
        'label': 'A large dollar amount is in play (above $5,000)',
        # Matches dollar figures with thousands separators ($5,000+) or
        # bare numbers like $5000, $10,000.50, etc.
        'patterns': [
            r'\$\s?(?:[5-9]|[1-9]\d|\d{3,})(?:[,]\d{3})+(?:\.\d+)?',
            r'\$\s?(?:[5-9]\d{3}|\d{5,})(?:\.\d+)?',
        ],
    },
]


def detect_high_risk(text: str) -> List[Dict]:
    """Return a list of triggered risk flags. Empty list if low-risk.

    Each entry: {'flag': str, 'label': str, 'matches': List[str]} where
    matches contains up to 3 example snippets that triggered the rule.
    """
    if not text:
        return []

    triggered = []
    for rule in RISK_RULES:
        all_matches: List[str] = []
        for pat in rule['patterns']:
            try:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    snippet = text[max(0, m.start() - 20):m.end() + 20].strip()
                    snippet = re.sub(r'\s+', ' ', snippet)
                    all_matches.append(snippet[:120])
                    if len(all_matches) >= 3:
                        break
                if len(all_matches) >= 3:
                    break
            except re.error:
                continue
        if all_matches:
            triggered.append({
                'flag': rule['flag'],
                'label': rule['label'],
                'matches': all_matches[:3],
            })
    return triggered


def is_blocking(flags: List[Dict]) -> bool:
    """Should the unlock flow be replaced with a legal-referral gate?"""
    return len(flags) > 0
