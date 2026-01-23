#!/usr/bin/env python3
"""Script to add trigger_case_analysis_after_payment function to app.py"""

# Read the current app.py
with open('app.py', 'r') as f:
    content = f.read()

# The function to add (stored in trigger_analysis.py)
with open('trigger_analysis.py', 'r') as f:
    trigger_func = f.read()

# Find the stripe webhook and insert the function before it
stripe_marker = "@app.route('/webhooks/stripe', methods=['POST'])"
pos = content.find(stripe_marker)

if pos == -1:
    print("ERROR: Could not find stripe webhook")
    exit(1)

# Insert the function before the stripe webhook
new_content = content[:pos] + "\n" + trigger_func + "\n\n" + content[pos:]

# Now add the call inside the stripe webhook after payment completion log
call_marker = 'logger.info(f"Payment completion processed successfully for token: {token}")'
call_pos = new_content.find(call_marker)

if call_pos == -1:
    print("ERROR: Could not find payment completion log line")
    exit(1)

# Find end of that line
end_of_line = new_content.find('\n', call_pos)

# The trigger call to add
trigger_call = '''

            # Trigger case analysis generation in background thread
            try:
                trigger_case_analysis_after_payment(token)
                logger.info(f"Triggered case analysis for token: {token}")
            except Exception as e:
                logger.warning(f"Failed to trigger case analysis (non-fatal): {str(e)}")'''

new_content = new_content[:end_of_line] + trigger_call + new_content[end_of_line:]

# Write the modified content
with open('app.py', 'w') as f:
    f.write(new_content)

print("SUCCESS: Added trigger_case_analysis_after_payment function and call to stripe webhook")

