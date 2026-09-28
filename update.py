"""Add newly exported WhatsApp messages to the existing data/qa.json, without
rebuilding or rewording anything that's already there.

Usage: GEMINI_API_KEY=... python update.py data/export
       GEMINI_API_KEY=... python update.py data/export --apply

Without --apply, this only prints a preview and writes data/qa.new.json for you
to look over. Run again with --apply once you're happy, to replace data/qa.json
and move the "already processed" marker forward.

Existing questions are never reworded or removed. A new answer either raises an
existing answer's count (same phone or same name) or is added underneath it;
genuinely new questions are appended at the end.
"""
import json
import os
import re
import sys
from pathlib import Path

from google import genai
from pydantic import BaseModel

from build import (
    LOG_DIR, MODEL, QA, Answer, categorize, extract,
    generate_with_retry, logger, parse, setup_logging,
)
from query_terms import TERMS

QA_PATH = Path("data/qa.json")
QA_NEW_PATH = Path("data/qa.new.json")
LAST_PROCESSED_PATH = LOG_DIR / "last_processed.txt"
NEW_TERMS_PATH = LOG_DIR / "new_terms.json"

# How many messages right before the cutoff to hand the model as context, so a
# reply to an older question still makes sense. They're never turned into QAs.
CONTEXT_MESSAGES = 50


def timestamp(m: dict) -> str:
    return f"{m['date']} {m['time']}"


# --- Attach new Q&As to existing ones (Gemini pass) -----------------------

class AttachResult(BaseModel):
    new_number: int
    existing_number: int | None = None


ATTACH_PROMPT = """Below are existing questions (numbered) already saved for the category
"{category}", and new questions (also numbered) just extracted from more recent chat
messages in the same category.

For each new question, decide whether it asks EXACTLY the same thing as one of the
existing questions. Questions with any meaningful difference are NOT the same: a
requirement ("English-speaking", "kosher", "delivers", "open on Saturday"), a different
place ("Tel Aviv" vs "Jerusalem"), or a different service ("cleaner" vs "dry cleaner").
Soft wishes ("good", "cheap", "preferably ...") are not a meaningful difference.

Return, for every new question, its number and the matching existing question's number
(existing_number), or leave existing_number out if it's genuinely a new question.

Existing questions:
{existing}

New questions:
{new}
"""


def attach(client, qas: list[QA], categories: list[str], existing: list[dict]) -> list[dict]:
    result = [dict(qa) for qa in existing]  # shallow copy of top-level dicts is enough
    for qa in result:
        qa["answers"] = [dict(a) for a in qa["answers"]]

    by_category: dict[str, list[int]] = {}
    for idx, qa in enumerate(existing):
        by_category.setdefault(qa["category"], []).append(idx)

    new_questions_added = []
    new_answers_added = 0
    counts_increased = 0

    new_by_category: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        new_by_category.setdefault(cat, []).append(i)

    for category, new_idxs in new_by_category.items():
        existing_idxs = by_category.get(category, [])
        mapping: dict[int, int | None] = {}
        if existing_idxs:
            resp = generate_with_retry(
                client,
                model=MODEL,
                contents=ATTACH_PROMPT.format(
                    category=category,
                    existing="\n".join(
                        f"{n}. {result[gi]['question']}" for n, gi in enumerate(existing_idxs)
                    ),
                    new="\n".join(
                        f"{n}. {qas[gi].question}" for n, gi in enumerate(new_idxs)
                    ),
                ),
                config={"response_mime_type": "application/json", "response_schema": list[AttachResult]},
            )
            mapping = {r.new_number: r.existing_number for r in resp.parsed or []}

        for n, gi in enumerate(new_idxs):
            qa = qas[gi]
            existing_number = mapping.get(n)
            if existing_number is not None and 0 <= existing_number < len(existing_idxs):
                target = result[existing_idxs[existing_number]]
                added, increased = merge_answers(target["answers"], qa.answers)
                new_answers_added += added
                counts_increased += increased
            else:
                new_qa = {
                    "question": qa.question, "category": category,
                    "answers": [
                        {"text": a.text, "name": a.name, "phone": a.phone, "count": 1, "date": a.date}
                        for a in qa.answers
                    ],
                }
                result.append(new_qa)
                new_questions_added.append(new_qa["question"])
                new_answers_added += len(qa.answers)

    logger.info(
        "Attach: %d new question(s), %d new answer(s), %d count(s) increased",
        len(new_questions_added), new_answers_added, counts_increased,
    )
    for q in new_questions_added:
        logger.info("  new question: %s", q)
    return result


def phone_digits(phone: str | None) -> str | None:
    return re.sub(r"\D", "", phone) if phone else None


def merge_answers(existing_answers: list[dict], new_answers: list[Answer]) -> tuple[int, int]:
    added = increased = 0
    for a in new_answers:
        a_phone = phone_digits(a.phone)
        a_name = a.name.strip().lower() if a.name else None
        match = next(
            (e for e in existing_answers
             if (a_phone and phone_digits(e.get("phone")) == a_phone)
             or (a_name and e.get("name") and e["name"].strip().lower() == a_name)),
            None,
        )
        if match:
            match["count"] = match.get("count", 1) + 1
            match["date"] = max(match["date"], a.date)
            increased += 1
        else:
            existing_answers.append(
                {"text": a.text, "name": a.name, "phone": a.phone, "count": 1, "date": a.date}
            )
            added += 1
    existing_answers.sort(key=lambda a: -a.get("count", 1))
    return added, increased


# --- Suggest new search terms (Gemini pass) --------------------------------

class TermSuggestion(BaseModel):
    term: str
    translation: str


NEW_TERMS_PROMPT = """Here are recent messages from a Tel Aviv newcomers WhatsApp group.

Find transliterated Hebrew words or English shorthand that a search tool would need
translated to plain English to understand (like "mazgan" -> "air conditioner AC", or
"gp" -> "family doctor"). Skip anything already in this list:
{known}

For each new one found, give the word and a short plain translation, not a guess at
what the person wants.

Messages:
{chunk}
"""


def suggest_terms(client, messages: list[dict]) -> list[TermSuggestion]:
    lines = [f"{m['date']} | {m['text']}" for m in messages]
    resp = generate_with_retry(
        client,
        model=MODEL,
        contents=NEW_TERMS_PROMPT.format(known=", ".join(sorted(TERMS)), chunk="\n".join(lines)),
        config={"response_mime_type": "application/json", "response_schema": list[TermSuggestion]},
    )
    return resp.parsed or []


# --- main -------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    args = [a for a in args if a != "--apply"]
    if len(args) != 1:
        raise SystemExit("Usage: python update.py <export_folder> [--apply]")
    setup_logging()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    messages = parse(Path(args[0]))
    if not messages:
        raise SystemExit("No messages found in export.")

    if not LAST_PROCESSED_PATH.exists():
        seed = max(timestamp(m) for m in messages)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        LAST_PROCESSED_PATH.write_text(seed)
        logger.info(
            "First run: treating everything up to %s as already processed. "
            "Nothing to do yet -- run update.py again after new messages come in.", seed,
        )
        return

    last_processed = LAST_PROCESSED_PATH.read_text().strip()
    cutoff = next(
        (i for i, m in enumerate(messages) if timestamp(m) > last_processed), len(messages)
    )
    new_messages = messages[cutoff:]
    if not new_messages:
        logger.info("No new messages since %s.", last_processed)
        return
    context_lines = [
        f"{m['date']} | {m['text']}" for m in messages[max(0, cutoff - CONTEXT_MESSAGES):cutoff]
    ]

    logger.info("Extracting Q&A pairs from %d new message(s)", len(new_messages))
    qas = extract(client, new_messages, context_lines=context_lines)
    if not qas:
        logger.info("No recommendation Q&As found in the new messages.")
        if apply:
            LAST_PROCESSED_PATH.write_text(max(timestamp(m) for m in new_messages))
        return

    logger.info("Assigning categories")
    categories = categorize(client, qas)

    existing = json.loads(QA_PATH.read_text())
    logger.info("Attaching to existing questions")
    merged = attach(client, qas, categories, existing)
    QA_NEW_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    logger.info("Wrote preview to %s (%d questions total, was %d)",
                QA_NEW_PATH, len(merged), len(existing))

    terms = suggest_terms(client, new_messages)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    NEW_TERMS_PATH.write_text(json.dumps([t.model_dump() for t in terms], ensure_ascii=False, indent=2))
    if terms:
        logger.info("Possible new search terms (not added automatically) -- see %s:", NEW_TERMS_PATH)
        for t in terms:
            logger.info("  %s -> %s", t.term, t.translation)

    if apply:
        QA_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        LAST_PROCESSED_PATH.write_text(max(timestamp(m) for m in new_messages))
        logger.info("Applied: data/qa.json updated, last-processed marker moved forward.")
    else:
        logger.info("Preview only. Re-run with --apply to update data/qa.json for real.")


if __name__ == "__main__":
    main()
