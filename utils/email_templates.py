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
    subject = 'Your HOA Case Details Are Still Here'
    body = (
        "Hey,\n"
        "\n"
        "You looked at the preview earlier but didn't grab the full response. Just a heads up, your case is still saved and ready when you are.\n"
        "\n"
        "The full version is \$29 and includes the actual response letter you can send back, the state statutes that apply, and a step by step on what to do next.\n"
        + (f"\n{preview_link}\n" if preview_link else "")
        + "\n"
        "No pressure. If you decide to handle it on your own that's totally fine. Just wanted to make sure you knew it was still here.\n"
        "\n"
        "Eric"
    )
    return subject, body


def nudge_3(preview_link: str = '') -> tuple:
    subject = 'Last Reminder: Your HOA Response'
    body = (
        "Hey,\n"
        "\n"
        "Last note from me on this one.\n"
        "\n"
        "Your HOA case is still saved if you want to come back to it. After this I'll stop emailing you about it.\n"
        + (f"\n{preview_link}\n" if preview_link else "")
        + "\n"
        "Good luck either way.\n"
        "\n"
        "Eric"
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
