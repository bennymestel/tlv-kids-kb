# TLV Newcomers KB — minimal plan

## Context
A WhatsApp group of Tel Aviv newcomers gets the same questions over and over (plumbers, where to buy things, services). The goal is to make past answers searchable. Requirements:
- Each question is stored once.
- Repeated recommendations are merged and ranked by how often they were given.
- Shared WhatsApp contacts show up as name + phone.

The chat is a one-time static export. The stack is plain Python + Gemini (`gemini-3.8-flash`, extraction) + sentence-transformers + in-memory Chroma + Streamlit. The project dir `/Users/a/Documents/tlv-kids-kb` is empty.

## File structure
```
tlv-kids-kb/
  requirements.txt   # streamlit, chromadb, sentence-transformers, google-genai, pydantic
  build.py           # one-time: parse export → Gemini extract → Gemini merge → qa.json
  search.py          # load qa.json → in-memory Chroma → search()
  app.py             # Streamlit UI
  data/export/       # unzipped WhatsApp export: _chat.txt + any .vcf files (keep out of git)
  data/qa.json       # generated; this is the only data the app needs
```

## build.py (~120 lines)
**1. `parse(folder)` → `list[dict(date, text)]`**
- One regex handles both export formats:
  - iOS `[23/09/2025, 14:05:12] Name: msg`
  - Android `23/09/2025, 14:05 - Name: msg`
- Lines that don't match continue the previous message.
- Skip media/deleted/system lines. Normalize dates to `YYYY-MM-DD`.
- **Contacts:** WhatsApp shows a shared contact as `Name.vcf (file attached)` or `<attached: 0000-Name.vcf>`.
  - If that `.vcf` is in the export folder, read its `FN:` and `TEL` lines with a regex. The file is there when the chat was exported "with media".
  - Replace the message with `[CONTACT: Name, +972...]`.
  - If the file is missing, use `[CONTACT: Name]`. The phone number simply isn't in the .txt, so the name is the best available.

**2. `extract(messages)` (Gemini pass 1, per chunk of ~300 messages)**
- Uses `google-genai` with JSON output and a pydantic `response_schema`.
- Output per chunk: `list[{question, category, answers: [{text, name, phone, date}]}]`
- The prompt tells Gemini to:
  - Keep only recommendation questions that have at least one useful answer.
  - Rewrite each question as a clear, standalone English question.
  - Gloss transliterated Hebrew, e.g. "shiputznik (renovation contractor)".
  - Fill in `name`/`phone` when an answer recommends a specific person or business, including from `[CONTACT: …]`.
  - Pick one category from a fixed list: Home & Repairs, Shopping, Kids & Education, Health, Bureaucracy & Services, Food & Restaurants, Transport, Other.

**3. `merge(qas)` (Gemini pass 2, one call per category)**
- Send all pass-1 QAs for that category.
- Gemini merges questions that ask the same thing into one canonical question.
- Within each question, answers recommending the same person/business/place (same phone or name) collapse into one answer with `count` and the latest `date`.
- Answers are sorted by `count` descending.
- Output: `list[{question, category, answers: [{text, name, phone, count, date}]}]`, written to `data/qa.json`.
- Why an LLM rather than embeddings: matching "Yossi the plumber" to "[CONTACT: Yossi Plumber, 054…]" needs judgment. A single call per category also sees duplicates across chunk boundaries.

Run it with `GEMINI_API_KEY=... python build.py data/export`.

## search.py (~30 lines)
- `load()`: read `qa.json` and create an in-memory `chromadb.Client()`.
  - Use `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")`.
  - Each document is the question plus its answer texts. The metadata is `{category}`, and the id is the list index.
  - Rebuild on every startup, which takes seconds at this size.
- `search(col, qas, query, category=None, k=10)`: run `col.query(..., where={"category": category} if category else None)` and return the QA dicts.

## app.py (~35 lines)
- `@st.cache_resource` wraps `search.load()`.
- Controls: a text input and a category selectbox ("All" + the categories).
- Each result shows `### question` with a category caption, then answers sorted by count. Each answer line is:
  `**Name** · 📞 phone · recommended ×N · last: date`, followed by the text.
  Name and phone appear only if present; `×N` appears only if N > 1.
- When the query is empty, list all QAs in the selected category.

## Verification
1. `pip install -r requirements.txt`
2. I'll build a fake export folder in the scratchpad containing:
   - lines in both formats, including a multi-line message
   - a `.vcf` contact that exists and one that doesn't
   - the same shiputznik question asked twice in different words, where the same contact is recommended twice
3. Run `build.py` on it. Expected:
   - one merged question
   - that contact with `count: 2`, name + phone
   - a gloss on "shiputznik"
4. Run `streamlit run app.py`. Search "renovation contractor", confirm the merged result appears with ×2, and check that the category filter works.
