"""A generated support-ticket task with known labels, three prompt versions and the response checks.

The task: read a short ticket and return JSON with ``category``, ``priority`` and ``refund_requested``. Every
ticket is built from snippets with known attributes, so the true label is a rule over what the ticket contains.
The labels are used only to measure how well a monitor tracks real quality; a monitor never sees them.
"""

from __future__ import annotations

import random

CATEGORIES = ["billing", "technical", "account", "shipping", "product_question"]
PRECEDENCE = ["billing", "account", "shipping", "technical", "product_question"]  # first one present wins
PRIORITIES = ["low", "medium", "high"]

# (text, blocked, refund_requested)
SNIPPETS = {
    "billing": [
        ("I was charged twice for my subscription this month.", True, False),
        ("There is a charge on my card from you that I do not recognize.", False, False),
        ("Please send me an invoice for last quarter.", False, False),
        ("I cancelled last week but was billed again and I want my money back.", False, True),
        ("The price on my invoice does not match the plan I signed up for.", False, False),
    ],
    "account": [
        ("I cannot log in because the password reset email never arrives.", True, False),
        ("Please change the email address on my profile.", False, False),
        ("A teammate needs admin permissions added to our workspace.", False, False),
        ("I am locked out of my account after too many attempts.", True, False),
        ("I want to merge two accounts that I created by mistake.", False, False),
    ],
    "shipping": [
        ("My order was due on Monday and it is still not here.", False, False),
        ("The tracking page says delivered but I never received the package.", True, False),
        ("I need to change the delivery address for my order.", False, False),
        ("The box arrived damaged and I would like a refund.", False, True),
        ("Part of my order is missing from the parcel.", False, False),
    ],
    "technical": [
        ("The app crashes every time I open the settings page.", False, False),
        ("I get error 502 when I try to export my data.", False, False),
        ("The dashboard has not loaded for two hours and my whole team is blocked.", True, False),
        ("Search returns nothing even for items I know exist.", False, False),
        ("Notifications arrive twice on my phone.", False, False),
    ],
    "product_question": [
        ("Does the pro plan include API access?", False, False),
        ("How do I connect the tool to my calendar?", False, False),
        ("Is there a way to export reports as PDF?", False, False),
        ("Which browsers do you support?", False, False),
        ("Can I use one licence on two devices?", False, False),
    ],
}
OPENERS = ["Hi,", "Hello team,", "Hey,", "Good morning,", ""]
CLOSERS = ["Thanks.", "Please help.", "Regards, Sam", "Any update soon would be great.", ""]

DEFINITIONS = """Categories:
- billing: charges, invoices, prices, payment methods, subscription fees.
- account: logging in, passwords, profile, permissions, team members.
- shipping: delivery, tracking, addresses, damaged or missing parcels.
- technical: errors, crashes, slow or broken features of the product.
- product_question: how to use the product, plans, features, compatibility.
If a ticket has more than one issue, pick the first category in this order that applies:
billing, account, shipping, technical, product_question.
Priority: high if the customer is blocked (cannot log in, cannot use the product) or has lost money or goods;
low if the ticket only asks a question; otherwise medium.
refund_requested: true only if the customer explicitly asks for money back."""

OUTPUT_RULE = "Return only a JSON object with the keys category, priority and refund_requested. No other text."

PROMPTS = {
    # the stable prompt
    "v1": "Classify the support ticket.\n" + DEFINITIONS + "\n" + OUTPUT_RULE,
    # format regression: ask for a sentence before the JSON
    "v2": "Classify the support ticket.\n"
    + DEFINITIONS
    + "\nAnswer in one friendly sentence first, then give the JSON object "
    + "with the keys category, priority and refund_requested.",
    # silent quality regression: the definitions and rules are gone, the output format is unchanged
    "v3": "Classify the support ticket. category is one of billing, technical, account, shipping, product_question. "
    "priority is one of low, medium, high. refund_requested is true or false.\n" + OUTPUT_RULE,
}

SPEC = {
    "json": True,
    "required_keys": ["category", "priority", "refund_requested"],
    "allowed": {"category": CATEGORIES, "priority": PRIORITIES, "refund_requested": [True, False]},
    "max_words": 40,
}

JUDGE_PROMPT = (
    "You grade a support-ticket classifier. Use these rules.\n" + DEFINITIONS + "\n\n"
    "Ticket:\n{ticket}\n\nClassifier answer:\n{answer}\n\n"
    "Is the category correct under the rules, and is the priority consistent with them? Reply with exactly YES or NO."
)


def make_ticket(index: int, seed: int = 20261010) -> dict:
    """Ticket ``index`` of the stream, with its true label. The same index always gives the same ticket."""
    rng = random.Random(f"{seed}-{index}")
    primary = rng.choice(CATEGORIES)
    issues = [(primary, *rng.choice(SNIPPETS[primary]))]
    if rng.random() < 0.4:  # a second issue from another category makes the precedence rule matter
        other = rng.choice([c for c in CATEGORIES if c != primary])
        issues.append((other, *rng.choice(SNIPPETS[other])))
        rng.shuffle(issues)
    present = {c for c, *_ in issues}
    category = next(c for c in PRECEDENCE if c in present)
    if any(blocked for _, _, blocked, _ in issues):
        priority = "high"
    elif present == {"product_question"}:
        priority = "low"
    else:
        priority = "medium"
    text = " ".join(t for t in [rng.choice(OPENERS), *[t for _, t, _, _ in issues], rng.choice(CLOSERS)] if t)
    return {
        "id": f"t{index}",
        "index": index,
        "text": text,
        "issues": len(issues),
        "label": {"category": category, "priority": priority, "refund_requested": any(r for _, _, _, r in issues)},
    }
