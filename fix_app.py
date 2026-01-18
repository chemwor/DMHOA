#!/usr/bin/env python3
"""Fix syntax errors in app.py"""

# Read the file
with open('app.py', 'r') as f:
    content = f.read()

# Find and fix the corrupted schema in case_analysis
# The corrupted section starts at line ~3165
corrupted_section = '''                "risks_and_deadlines": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "deadlines": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "risks": {"type": "array", "items": {"type": "string"}, "minItems": 3}
                    },
                    "required": ["deadlines", "risks"]
                    f"- drafts.clarification MUST be: \\"{draft_titles['clarification']}\\"\\n"
                    f"- drafts.extension MUST be: \\"{draft_titles['extension']}\\"\\n"
                    f"- drafts.compliance MUST be: \\"{draft_titles['compliance']}\\"\\n\\n"
                    f"Also include draft_titles using these exact same strings.\\n\\n"
                    f"summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.\\n"
                    f"Drafts must be PLAIN TEXT ONLY with \\\\n, and must NOT include any HTML tags.\\n\\n"
                    f"Make this feel like a $30 deliverable: concrete, specific, complete.\\n"
                )
            }
        ]'''

fixed_section = '''                "risks_and_deadlines": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "deadlines": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "risks": {"type": "array", "items": {"type": "string"}, "minItems": 3}
                    },
                    "required": ["deadlines", "risks"]
                },
                "action_plan": {"type": "array", "items": {"type": "string"}, "minItems": 6},
                "drafts": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "clarification": {"type": "string"},
                        "extension": {"type": "string"},
                        "compliance": {"type": "string"}
                    },
                    "required": ["clarification", "extension", "compliance"]
                },
                "questions_to_ask": {"type": "array", "items": {"type": "string"}, "minItems": 6},
                "lowest_cost_path": {"type": "array", "items": {"type": "string"}, "minItems": 4}
            },
            "required": [
                "summary_html", "letter_summary", "draft_titles", "risks_and_deadlines",
                "action_plan", "drafts", "questions_to_ask", "lowest_cost_path"
            ]
        }

        doc_fingerprint = {
            'count': len(docs),
            'usableCount': len(usable_docs),
            'newestUpdatedAt': docs_newest,
            'ids': [d['id'] for d in docs],
            'statuses': [d.get('status') for d in docs],
            'charCounts': [d.get('char_count') for d in docs]
        }

        messages = [
            {
                "role": "system",
                "content": """
You generate HOA dispute assistance for a homeowner.
This is educational drafting help, not legal advice.

OUTPUT RULES (CRITICAL):
- ONLY "summary_html" may contain HTML.
- summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.
- ALL drafts (clarification/extension/compliance) MUST be PLAIN TEXT ONLY:
  - NO HTML tags
  - Use newlines with \\\\n
  - Bullets: use "- item" lines
- Return STRICT JSON that matches the schema exactly.

DRAFT QUALITY REQUIREMENTS:
- Each draft must be a complete, ready-to-send letter.
- MUST directly quote or reference concrete facts from the extracted documents when available
  (deadlines, email addresses, paragraph citations, dollar amounts, dates, etc.).
- Each must include:
  - Subject line
  - Short opening
  - 3–6 bullet-point requests (specific asks)
  - Proposed timeline (e.g., "Please respond within 10 business days" if no deadline is provided)
  - Request fines/penalties be paused/waived while pending (when relevant)
  - Closing requesting confirmation in writing

DEPTH REQUIREMENTS:
- action_plan >= 6 steps with timing hints (Today / 48 hours / Before deadline).
- risks >= 3 concrete risks tied to HOA enforcement.
- questions_to_ask >= 6 questions.
- lowest_cost_path >= 4 items.

STYLE:
- Calm, professional, firm, factual.
"""
            },
            {
                "role": "user",
                "content": (
                    f"Case payload JSON:\\n{json.dumps(payload)}\\n\\n"
                    f"Document fingerprint (debug):\\n{json.dumps(doc_fingerprint)}\\n\\n"
                    f"Extracted documents:\\n{docs_block}\\n\\n"
                    f"Draft types for this case (MUST follow exactly):\\n"
                    f"- drafts.clarification MUST be: \\"{draft_titles['clarification']}\\"\\n"
                    f"- drafts.extension MUST be: \\"{draft_titles['extension']}\\"\\n"
                    f"- drafts.compliance MUST be: \\"{draft_titles['compliance']}\\"\\n\\n"
                    f"Also include draft_titles using these exact same strings.\\n\\n"
                    f"summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.\\n"
                    f"Drafts must be PLAIN TEXT ONLY with \\\\n, and must NOT include any HTML tags.\\n\\n"
                    f"Make this feel like a $30 deliverable: concrete, specific, complete.\\n"
                )
            }
        ]'''

# Replace the corrupted section
content = content.replace(corrupted_section, fixed_section)

# Remove duplicate function definitions by finding the second occurrence
# Find the second trigger_document_extraction_async
lines = content.split('\n')
found_first_trigger = False
found_first_clamp = False
cleaned_lines = []
skip_until_next_def = False

for i, line in enumerate(lines):
    # Handle duplicate trigger_document_extraction_async
    if line.startswith('def trigger_document_extraction_async('):
        if not found_first_trigger:
            found_first_trigger = True
            cleaned_lines.append(line)
        else:
            # Skip the duplicate and everything until the next @app.route or def at the same indentation
            skip_until_next_def = True
            continue

    # Handle duplicate clamp_text
    elif line.startswith('def clamp_text('):
        if not found_first_clamp:
            found_first_clamp = True
            cleaned_lines.append(line)
        else:
            skip_until_next_def = True
            continue

    # Stop skipping when we hit the next function/route at base indentation
    elif skip_until_next_def and (line.startswith('@app.route(') or (line.startswith('def ') and not line.startswith('    '))):
        skip_until_next_def = False
        cleaned_lines.append(line)

    # Normal lines
    elif not skip_until_next_def:
        cleaned_lines.append(line)

content = '\n'.join(cleaned_lines)

# Write the fixed content
with open('app.py', 'w') as f:
    f.write(content)

print("Fixed app.py successfully!")

