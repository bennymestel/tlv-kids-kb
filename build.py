"""One-time build: WhatsApp export -> data/qa.json

Usage: GEMINI_API_KEY=... python build.py data/export
       GEMINI_API_KEY=... python build.py --merge-only   (redo merge + categories only)

Writes intermediate outputs to data/log/ so each step can be inspected:
  data/log/parsed.json     - messages after parsing (incl. resolved contacts)
  data/log/extracted.json  - raw Q&A pairs from pass 1, before merging
  data/log/build.log       - a line per chunk/category with counts
data/qa.json is the final merged output the app reads.
"""
import json
import logging
import os
import re
import sys
import time
from enum import Enum
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel

CATEGORIES = {
    "Home & Repairs": "fixing, cleaning or moving a home: plumbers, electricians, AC, handymen, "
                      "locksmiths, cleaners, movers, pest control, appliance repair, renovation",
    "Shopping": "where to buy things: stores, online shops, furniture, clothes, electronics, Judaica",
    "Kids & Education": "children, schools, gan, babysitters, tutors, classes, ulpan and Hebrew lessons",
    "Health": "doctors, dentists, clinics, therapists, physio, pharmacies, gyms and fitness",
    "Bureaucracy & Services": "lawyers, accountants, notaries, insurance, phone and internet "
                              "providers, printing, photographers, tailors, dry cleaners, "
                              "hair, nails and beauty",
    "Food & Restaurants": "restaurants, cafes, bars, takeout, bakeries, butchers, groceries, catering",
    "Transport": "cars, taxis, drivers, shuttles, car rental and repair, driving lessons, bikes, "
                 "scooters",
    "Other": "anything else: religious services, events, DJs, pets, tours, travel, donations",
}

CHUNK_SIZE = 300
MODEL = "gemini-3.8-flash"
LOG_DIR = Path("data/log")

logger = logging.getLogger("build")

RETRYABLE_CODES = {429, 500, 503, 504}
MAX_RETRIES = 6


def generate_with_retry(client, **kwargs):
    """Gemini's own retry gives up on transient 503s within seconds; a ~40-chunk
    build easily outlasts that, so retry ourselves with longer backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                raise
            delay = min(10 * 2 ** (attempt - 1), 120)
            logger.warning(
                "Gemini call failed (%s), retrying in %ds (attempt %d/%d)",
                e.code, delay, attempt, MAX_RETRIES,
            )
            time.sleep(delay)

# --- 1. Parse the export -----------------------------------------------

# iOS:     [23/09/2025, 14:05:12] Name: message
# Android: 23/09/2025, 14:05 - Name: message
LINE_RE = re.compile(
    r"^(?:\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})(?::\d{2})?\] |"
    r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - )(.+?): (.*)$"
)

SKIP_RE = re.compile(
    r"<Media omitted>|image omitted|video omitted|audio omitted|"
    r"This message was deleted|You deleted this message|"
    r"Missed voice call|Missed video call"
)

# WhatsApp contact-share message text, e.g.
#   "Yossi Plumber.vcf (file attached)"
#   "<attached: 0000-Yossi Plumber.vcf>"
# Anchored on the surrounding literal text rather than an allowed-character class:
# real contact/business names contain '&', parens, commas, emoji, etc. that a
# character class would need to enumerate and inevitably miss.
CONTACT_RE = re.compile(r"<attached: (.+)\.vcf>|(.+)\.vcf \(file attached\)")


INVISIBLE_MARKS_RE = re.compile("[\u200E\u200F﻿]")


def strip_invisible(s: str) -> str:
    """Remove LTR/RTL marks and BOM that WhatsApp sprinkles into export text
    (e.g. before timestamps and inside attachment names); left in place they
    break the line regex and vcf-name lookups."""
    return INVISIBLE_MARKS_RE.sub("", s)


def normalize_date(d: str) -> str:
    day, month, year = d.split("/")
    if len(year) == 2:
        year = "20" + year
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def normalize_time(t: str) -> str:
    hour, minute = t.split(":")
    return f"{hour.zfill(2)}:{minute}"


def load_vcards(folder: Path) -> dict[str, str]:
    """Map .vcf filename -> 'Name, phone' (or just 'Name' if no phone)."""
    cards = {}
    for vcf in folder.glob("*.vcf"):
        text = vcf.read_text(errors="ignore")
        name_m = re.search(r"^FN:(.+)$", text, re.MULTILINE)
        tel_m = re.search(r"^(?:item\d+\.)?TEL[^:]*:(.+)$", text, re.MULTILINE)
        name = name_m.group(1).strip() if name_m else vcf.stem
        key = strip_invisible(vcf.name)
        cards[key] = f"{name}, {tel_m.group(1).strip()}" if tel_m else name
    logger.info("Found %d .vcf contact files: %s", len(cards), sorted(cards))
    return cards


def parse(folder: Path) -> list[dict]:
    chat_files = list(folder.glob("*.txt"))
    if not chat_files:
        raise SystemExit(f"No .txt chat file found in {folder}")
    text = chat_files[0].read_text(errors="ignore")
    vcards = load_vcards(folder)

    messages = []
    skipped = 0
    unmatched = 0
    resolved_contacts = 0
    unresolved_contacts = 0
    for line in text.splitlines():
        line = strip_invisible(line)
        m = LINE_RE.match(line)
        if m:
            date = m.group(1) or m.group(3)
            clock = m.group(2) or m.group(4)
            body = m.group(6)
            if SKIP_RE.search(body):
                skipped += 1
                continue
            contact_m = CONTACT_RE.search(body)
            if contact_m:
                # The attachment name always carries the same numeric prefix (and any
                # trailing space) as the actual .vcf filename, so look it up as-is;
                # only the leading space the regex picks up before "-NNNN-" is ours to drop.
                raw_name = (contact_m.group(1) or contact_m.group(2)).lstrip()
                info = vcards.get(f"{raw_name}.vcf")
                if info:
                    resolved_contacts += 1
                else:
                    info = raw_name
                    unresolved_contacts += 1
                body = f"[CONTACT: {info}]"
            messages.append({
                "date": normalize_date(date), "time": normalize_time(clock), "text": body,
            })
        elif line.strip():
            if messages:
                messages[-1]["text"] += "\n" + line
            else:
                unmatched += 1

    logger.info(
        "Parsed %d messages (skipped %d media/system lines, %d unmatched lines, "
        "%d contacts resolved from .vcf, %d contacts without a matching .vcf)",
        len(messages), skipped, unmatched, resolved_contacts, unresolved_contacts,
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "parsed.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2)
    )
    return messages


# --- 2. Extract Q&A pairs (Gemini pass 1) -------------------------------

class Answer(BaseModel):
    text: str
    name: str | None = None
    phone: str | None = None
    date: str


class QA(BaseModel):
    question: str
    answers: list[Answer]


EXTRACT_PROMPT = """You are reading a slice of a WhatsApp group chat for newcomers to Tel Aviv.
Each line is "date | text".

Find messages that ask for a local recommendation (plumbers, shops, services, etc.)
and were answered usefully. For each one, output:
- question: rewritten as a clear, standalone English question
- answers: the useful replies, each with:
  - text: the recommendation, keeping names/phone numbers/places
  - name / phone: fill in if the answer recommends a specific person or business
    (including from a "[CONTACT: Name, phone]" message)
  - date: the date of that answer

Gloss any transliterated Hebrew words with a short English explanation in parentheses,
e.g. "shiputznik (renovation contractor)".

Skip small talk and questions with no useful answer. Do not include author names.

Lines marked [CONTEXT] are earlier messages included only so you can understand what a
reply refers to; never create a question or answer from a [CONTEXT] line itself, only
from the other lines.

Messages:
{chunk}
"""


def extract(client, messages: list[dict], context_lines: list[str] | None = None) -> list[QA]:
    lines = [f"{m['date']} | {m['text']}" for m in messages]
    context_block = "\n".join(f"[CONTEXT] {line}" for line in context_lines or [])
    all_qas = []
    n_chunks = (len(lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk_num = i // CHUNK_SIZE + 1
        chunk = "\n".join(lines[i:i + CHUNK_SIZE])
        if context_block:
            chunk = context_block + "\n" + chunk
        resp = generate_with_retry(
            client,
            model=MODEL,
            contents=EXTRACT_PROMPT.format(chunk=chunk),
            config={
                "response_mime_type": "application/json",
                "response_schema": list[QA],
            },
        )
        chunk_qas = resp.parsed or []
        all_qas.extend(chunk_qas)
        logger.info(
            "Extract chunk %d/%d (%d messages): %d QAs found",
            chunk_num, n_chunks, len(lines[i:i + CHUNK_SIZE]), len(chunk_qas),
        )
        for qa in chunk_qas:
            logger.debug("  %s (%d answers)", qa.question, len(qa.answers))

        # Written after every chunk (not just at the end) so a failure deep into a
        # long run still leaves the completed chunks inspectable/recoverable.
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "extracted.json").write_text(
            json.dumps([q.model_dump() for q in all_qas], ensure_ascii=False, indent=2)
        )
    return all_qas


# --- 3. Assign categories (Gemini pass 2) --------------------------------
# Done once over all questions, so similar questions land in the same category.

Category = Enum("Category", {name: name for name in CATEGORIES})


class Categorized(BaseModel):
    number: int
    category: Category


CATEGORIZE_PROMPT = """Assign each numbered question from a Tel Aviv newcomers group chat
to exactly one category. Similar questions must get the same category.

Categories:
{categories}

Questions:
{questions}
"""


def categorize(client, qas: list[QA]) -> list[str]:
    resp = generate_with_retry(
        client,
        model=MODEL,
        contents=CATEGORIZE_PROMPT.format(
            categories="\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items()),
            questions="\n".join(f"{i}. {qa.question}" for i, qa in enumerate(qas)),
        ),
        config={"response_mime_type": "application/json", "response_schema": list[Categorized]},
    )
    category = {c.number: c.category.value for c in resp.parsed or []}
    missing = [qa.question for i, qa in enumerate(qas) if i not in category]
    if missing:
        logger.warning("No category returned for %d questions, using Other: %s",
                       len(missing), missing)
    return [category.get(i, "Other") for i in range(len(qas))]


# --- 4. Merge duplicates per category (Gemini pass 3) --------------------

class MergedAnswer(BaseModel):
    text: str
    name: str | None = None
    phone: str | None = None
    count: int
    date: str


class MergedQA(BaseModel):
    question: str
    answers: list[MergedAnswer]


MERGE_PROMPT = """Below are recommendation Q&A pairs from a Tel Aviv newcomers group chat,
all in the category "{category}". The same question is often asked multiple times in
different words, and the same person/business is often recommended multiple times.

Merge questions that ask for exactly the same thing into one canonical question (pick the
clearest phrasing). Questions with any meaningful difference are NOT the same: a requirement
("English-speaking", "kosher", "delivers", "open on Saturday"), a different place
("Tel Aviv" vs "Jerusalem", "North Tel Aviv") or a different service ("cleaner" vs
"dry cleaner"). Soft wishes ("good", "cheap", "affordable", "preferably ...") are not a
meaningful difference.

Within each merged question, combine answers that recommend the same person/business/place
(same phone number, or same name) into a single answer: set "count" to how many times it
was recommended, and "date" to the most recent date. Sort answers by count, descending.

Q&A pairs (JSON):
{qas}
"""


def merge(client, qas: list[QA], categories: list[str]) -> list[dict]:
    by_category: dict[str, list[QA]] = {}
    for qa, category in zip(qas, categories):
        by_category.setdefault(category, []).append(qa)

    merged = []
    for category, items in by_category.items():
        resp = generate_with_retry(
            client,
            model=MODEL,
            contents=MERGE_PROMPT.format(
                category=category,
                qas=json.dumps([q.model_dump() for q in items], ensure_ascii=False),
            ),
            config={
                "response_mime_type": "application/json",
                "response_schema": list[MergedQA],
            },
        )
        result = resp.parsed or []
        logger.info("Merge [%s]: %d raw QAs -> %d merged QAs", category, len(items), len(result))
        merged.extend(
            {"question": qa.question, "category": category,
             "answers": [a.model_dump() for a in qa.answers]}
            for qa in result
        )
    return merged


# --- main -----------------------------------------------------------------

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_DIR / "build.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # keep console less noisy than the file; INFO+ on console, DEBUG in file
    logging.getLogger().handlers[1].setLevel(logging.INFO)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python build.py <export_folder>\n"
                         "       python build.py --merge-only   (reuses data/log/extracted.json)")
    setup_logging()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    if sys.argv[1] == "--merge-only":
        qas = [QA(**qa) for qa in json.loads((LOG_DIR / "extracted.json").read_text())]
    else:
        folder = Path(sys.argv[1])
        logger.info("Parsing chat export from %s", folder)
        messages = parse(folder)

        logger.info("Extracting Q&A pairs (pass 1)")
        qas = extract(client, messages)

    logger.info("Assigning categories (pass 2)")
    categories = categorize(client, qas)

    logger.info("Merging duplicates per category (pass 3)")
    merged = merge(client, qas, categories)

    out_path = Path("data/qa.json")
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    logger.info("Wrote %d final QAs to %s", len(merged), out_path)
    logger.info("Inspect data/log/parsed.json, data/log/extracted.json and data/log/build.log "
                "to see how each step behaved.")


if __name__ == "__main__":
    main()
