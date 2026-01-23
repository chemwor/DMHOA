"""
Helper function to trigger case analysis after payment.
This will be integrated into app.py
"""

def trigger_case_analysis_after_payment(token: str):
    """
    Trigger case analysis generation in a background thread after payment.
    This creates a record in dmhoa_case_outputs with the full analysis.
    """
    def run_analysis():
        try:
            logger.info(f"Starting case analysis generation for token {token[:8]}... after payment")

            # 1) Ensure case exists and is unlocked/paid
            case_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
            case_params = {
                'token': f'eq.{token}',
                'select': 'token,unlocked,status,payload'
            }
            case_headers_local = supabase_headers()

            case_response = requests.get(case_url, params=case_params, headers=case_headers_local, timeout=TIMEOUT)
            case_response.raise_for_status()
            cases = case_response.json()
            case_row = cases[0] if cases else None

            if not case_row:
                logger.error(f"Case not found for token {token[:8]}... during post-payment analysis")
                return

            if not case_row.get('unlocked'):
                logger.warning(f"Case {token[:8]}... not unlocked yet during post-payment analysis, skipping")
                return

            # 2) Load extracted documents for this case
            docs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_documents"
            docs_params = {
                'token': f'eq.{token}',
                'select': 'id,filename,path,status,extracted_text,page_count,char_count,updated_at,error',
                'order': 'updated_at.desc',
                'limit': '10'
            }
            docs_headers_local = supabase_headers()

            docs_response = requests.get(docs_url, params=docs_params, headers=docs_headers_local, timeout=TIMEOUT)
            docs_response.raise_for_status()
            docs = docs_response.json()

            docs_newest = newest_updated_at(docs)

            # Consider usable if it has text
            usable_docs = [
                d for d in docs
                if isinstance(d.get('extracted_text'), str) and d['extracted_text'].strip()
            ]

            logger.info(f"Post-payment analysis: {len(docs)} docs found, {len(usable_docs)} usable for token {token[:8]}...")

            if usable_docs:
                docs_block = '\n'.join([
                    f"DOCUMENT {i + 1}: {d.get('filename') or d.get('path') or d['id']}\n"
                    f"---\n{(d.get('extracted_text', '') or '')[:12000]}\n---\n"
                    for i, d in enumerate(usable_docs[:5])
                ])
            else:
                statuses = ', '.join([d.get('status', 'unknown') for d in docs]) or 'none'
                errors = ' | '.join([d.get('error', '') for d in docs if d.get('error')])[:100] or 'none'
                docs_block = f"""No document text available yet.
Docs found: {len(docs)}
Statuses: {statuses}
Errors: {errors}"""

            # 3) Check if outputs already exist
            outputs_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
            outputs_params = {
                'case_token': f'eq.{token}',
                'select': 'case_token,status,outputs,error,updated_at'
            }
            outputs_headers_local = supabase_headers()

            outputs_response = requests.get(outputs_url, params=outputs_params, headers=outputs_headers_local, timeout=TIMEOUT)
            outputs_response.raise_for_status()
            existing_outputs = outputs_response.json()
            existing_out = existing_outputs[0] if existing_outputs else None

            out_updated = safe_iso(existing_out.get('updated_at') if existing_out else None)
            docs_are_newer = (
                docs_newest and out_updated and
                datetime.fromisoformat(docs_newest) > datetime.fromisoformat(out_updated)
            )

            # If we already have ready outputs and docs haven't changed, skip
            if (existing_out and existing_out.get('status') == 'ready' and
                existing_out.get('outputs') and not docs_are_newer):
                logger.info(f"Case outputs already exist and are current for token {token[:8]}..., skipping generation")
                return

            # 4) Mark outputs as pending (upsert)
            pending_data = {
                'case_token': token,
                'status': 'pending',
                'error': None,
                'model': 'gpt-4o-mini',
                'prompt_version': 'v3_docs_cache_invalidation',
                'updated_at': datetime.utcnow().isoformat()
            }

            upsert_url = f"{SUPABASE_URL}/rest/v1/dmhoa_case_outputs"
            upsert_headers_local = supabase_headers()
            upsert_headers_local['Prefer'] = 'resolution=merge-duplicates'

            upsert_response = requests.post(upsert_url, headers=upsert_headers_local, json=pending_data, timeout=TIMEOUT)
            upsert_response.raise_for_status()

            payload = case_row.get('payload') or {}
            draft_titles = get_draft_titles(payload)

            # 5) OpenAI Responses API call (strict JSON schema)
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary_html": {"type": "string"},
                    "letter_summary": {"type": "string"},
                    "draft_titles": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "clarification": {"type": "string"},
                            "extension": {"type": "string"},
                            "compliance": {"type": "string"}
                        },
                        "required": ["clarification", "extension", "compliance"]
                    },
                    "risks_and_deadlines": {
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
  - Use newlines with \\n
  - Bullets: use "- item" lines
- Return STRICT JSON that matches the schema exactly.

DRAFT QUALITY REQUIREMENTS:
- Each draft must be a complete, ready-to-send letter.
- MUST directly quote or reference concrete facts from the extracted documents when available
  (deadlines, email addresses, paragraph citations, dollar amounts, dates, etc.).
- Each must include:
  - Subject line
  - Short opening
  - 3-6 bullet-point requests (specific asks)
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
                        f"Case payload JSON:\n{json.dumps(payload)}\n\n"
                        f"Document fingerprint (debug):\n{json.dumps(doc_fingerprint)}\n\n"
                        f"Extracted documents:\n{docs_block}\n\n"
                        f"Draft types for this case (MUST follow exactly):\n"
                        f"- drafts.clarification MUST be: \"{draft_titles['clarification']}\"\n"
                        f"- drafts.extension MUST be: \"{draft_titles['extension']}\"\n"
                        f"- drafts.compliance MUST be: \"{draft_titles['compliance']}\"\n\n"
                        f"Also include draft_titles using these exact same strings.\n\n"
                        f"summary_html must be valid HTML using ONLY: <div>, <strong>, <ul>, <li>.\n"
                        f"Drafts must be PLAIN TEXT ONLY with \\n, and must NOT include any HTML tags.\n\n"
                        f"Make this feel like a $30 deliverable: concrete, specific, complete.\n"
                    )
                }
            ]

            openai_payload = {
                "model": "gpt-4o-mini",
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "dmhoa_case_outputs",
                        "strict": True,
                        "schema": schema
                    }
                }
            }

            # Make OpenAI API call
            openai_response = requests.post(
                'https://api.openai.com/v1/responses',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json=openai_payload,
                timeout=(10, 120)
            )

            if not openai_response.ok:
                error_text = openai_response.text
                logger.error(f'OpenAI call failed during post-payment analysis: {openai_response.status_code}, {error_text}')

                # Update outputs table with error
                error_data = {
                    'case_token': token,
                    'status': 'error',
                    'error': error_text or 'OpenAI call failed',
                    'updated_at': datetime.utcnow().isoformat()
                }
                requests.post(upsert_url, headers=upsert_headers_local, json=error_data, timeout=TIMEOUT)
                return

            openai_json = openai_response.json()
            structured = extract_structured_result(openai_json)

            if structured:
                outputs_to_store = {
                    **structured,
                    'draft_titles': structured.get('draft_titles', draft_titles),
                    'doc_fingerprint': doc_fingerprint
                }
            else:
                outputs_to_store = {
                    'raw': openai_json,
                    'draft_titles': draft_titles,
                    'doc_fingerprint': doc_fingerprint
                }

            # Save successful outputs
            success_data = {
                'case_token': token,
                'status': 'ready',
                'outputs': outputs_to_store,
                'error': None,
                'model': 'gpt-4o-mini',
                'prompt_version': 'v3_docs_cache_invalidation',
                'updated_at': datetime.utcnow().isoformat()
            }

            requests.post(upsert_url, headers=upsert_headers_local, json=success_data, timeout=TIMEOUT)

            # Update case updated_at timestamp
            try:
                case_update_url = f"{SUPABASE_URL}/rest/v1/dmhoa_cases"
                case_update_params = {'token': f'eq.{token}'}
                case_update_data = {'updated_at': datetime.utcnow().isoformat()}
                case_update_headers_local = supabase_headers()
                requests.patch(case_update_url, params=case_update_params,
                             headers=case_update_headers_local, json=case_update_data, timeout=TIMEOUT)
            except Exception:
                pass  # Best effort

            logger.info(f"Successfully generated case analysis for token {token[:8]}... after payment")

        except Exception as e:
            logger.error(f"Error in post-payment case analysis for token {token[:8]}...: {str(e)}")

    # Run in background thread so we don't block the webhook response
    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.daemon = True
    analysis_thread.start()
    logger.info(f"Started background case analysis thread for token {token[:8]}...")

