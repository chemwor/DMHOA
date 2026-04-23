"""
Plain text email templates for the funnel. No HTML, no buttons, no checkmarks,
no arrows. Each template returns a (subject, body) tuple. Read like a person
typed it in Gmail.
"""


def quick_preview_confirmation(preview_link: str = '') -> tuple:
    subject = 'Your HOA Preview Is Ready'
    body = (
        "Hey,\n"
        "\n"
        "Saw your HOA notice come through. I pulled up a quick preview based on what you submitted.\n"
        "\n"
        "If you want the full breakdown (the actual response letter, the statutes, the deadline, all of it), the next step is the full preview.\n"
        + (f"\n{preview_link}\n" if preview_link else "")
        + "\n"
        "Reply to this email if you have any questions about your case. I read these myself.\n"
        "\n"
        "Eric\n"
        "Dispute My HOA"
    )
    return subject, body


def nudge_1(preview_link: str = '') -> tuple:
    subject = 'Did Something Go Wrong?'
    body = (
        "Hey,\n"
        "\n"
        "Noticed you started a preview a few hours ago but didn't finish.\n"
        "\n"
        "Sometimes the form glitches or people get pulled away. Either way, your case is still saved on my end. You can pick up where you left off.\n"
        + (f"\n{preview_link}\n" if preview_link else "")
        + "\n"
        "If something didn't work, just reply and let me know. I'll fix it.\n"
        "\n"
        "Eric"
    )
    return subject, body


def nudge_2(preview_link: str = '') -> tuple:
    subject = 'Your HOA Response Letter Is Ready'
    body = (
        "Hey,\n"
        "\n"
        "You saw the preview of your case earlier. The full response letter is drafted and waiting for you.\n"
        "\n"
        "Quick reminder of what you get for \$29:\n"
        "\n"
        "  Your actual response letter, ready to edit and send\n"
        "  The state statutes that apply to your situation\n"
        "  A step by step checklist so nothing gets missed\n"
        "  Deadline tracking so you don't run out of time\n"
        "\n"
        "Most HOA notices have a 14 to 30 day response window. Once that passes, your options shrink.\n"
        + (f"\n{preview_link}\n" if preview_link else "")
        + "\n"
        "If \$29 doesn't feel worth it after you see the full letter, there's a 14 day money back guarantee. No questions asked.\n"
        "\n"
        "Eric\n"
        "Dispute My HOA"
    )
    return subject, body


def nudge_3(preview_link: str = '') -> tuple:
    subject = 'Your HOA Deadline Is Getting Closer'
    body = (
        "Hey,\n"
        "\n"
        "Last note from me.\n"
        "\n"
        "Your drafted response letter is still saved. If you want it, it's \$29 and ready to go. If not, no hard feelings.\n"
        + (f"\n{preview_link}\n" if preview_link else "")
        + "\n"
        "The only thing I'd say is don't let the deadline pass without responding at all. Even a short written response is better than silence. HOAs count on people ignoring notices.\n"
        "\n"
        "Good luck with it.\n"
        "\n"
        "Eric\n"
        "Dispute My HOA"
    )
    return subject, body


def purchase_confirmation(case_link: str = '') -> tuple:
    subject = "You're Ready To Respond To Your HOA"
    body = (
        "Hey,\n"
        "\n"
        "Got your purchase. Thanks for trusting me with this.\n"
        "\n"
        "Here's what you have access to now:\n"
        "\n"
        "  Your full response letter, drafted and ready to send\n"
        "  The state statutes that back you up\n"
        "  A checklist of what to do and when\n"
        "  Deadline reminders so nothing slips\n"
        + (f"\nYou can pull it all up here:\n{case_link}\n" if case_link else "")
        + "\n"
        "Read through it once before you send anything. Make any tweaks you want, the wording is yours to edit. Then send it certified mail so you have proof of delivery.\n"
        "\n"
        "If you hit a wall or need to talk through anything, reply to this email. I read every one.\n"
        "\n"
        "Eric\n"
        "Dispute My HOA\n"
        "\n"
        "p.s. just so we're clear, this is a self help tool, not legal advice. If your HOA is threatening a lien or court, talk to a licensed attorney in your state."
    )
    return subject, body
