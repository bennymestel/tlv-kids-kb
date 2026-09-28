import re

import streamlit as st

from search import load, search
from sheet_log import log_feedback, log_rating, log_search

st.set_page_config(page_title="TLV Kids KB", page_icon="assets/icon.jpg")
st.title("TLV Kids Knowledge Base")


@st.cache_resource
def get_data():
    return load()


def note(a):
    """The answer text minus the name, phone and contact card already shown in bold.

    'Manny (+972 52-702-9498) - he's great!' -> "he's great!"; '' if nothing is left.
    """
    text = re.sub(r"\[CONTACT:[^\]]*\]", "", a["text"]).strip()
    # Only strip them from the start, and only when followed by a separator or a number:
    # in "Noa Burg is in Yafo" or "pizza at Brooklyn on dizengoff" the name is part of the sentence.
    if a.get("name"):
        # "קלין בקליק (Clean BeClick)": the text may have either half on its own.
        names = [a["name"], *re.fullmatch(r"(.*?)\s*(?:\((.*)\))?", a["name"]).groups()]
        for name in filter(None, names):
            text = re.sub(
                r"^" + re.escape(name) + r"(?=\s*(?:$|[,(:\-–+\d]))[\s,:\-–]*",
                "", text, flags=re.IGNORECASE,
            )
    if a.get("phone"):
        text = re.sub(r"^\(?" + re.escape(a["phone"]) + r"\)?[\s,.:\-–]*", "", text)
    text = text.strip(" ,:;-–\n")
    # Leftovers like ")" from "Lima ;)" aren't worth a bullet.
    return text if re.search(r"\w", text) else ""


CATEGORY_STYLE = {
    "Food & Restaurants": ("🍽️", "orange"),
    "Bureaucracy & Services": ("📋", "blue"),
    "Shopping": ("🛍️", "violet"),
    "Home & Repairs": ("🔧", "red"),
    "Health": ("🩺", "green"),
    "Transport": ("🚗", "yellow"),
    "Kids & Education": ("🧸", "primary"),
    "Other": ("✨", "gray"),
}


def category_label(category):
    return f"{CATEGORY_STYLE.get(category, ('✨', 'gray'))[0]} {category}"


def phone_links(phone):
    """Tap-to-call link, plus a WhatsApp link for mobile numbers."""
    digits = re.sub(r"\D", "", phone)
    if phone.startswith("+"):
        intl = digits
    elif digits.startswith("0"):
        intl = "972" + digits[1:]
    elif len(digits) == 9 and digits.startswith("5"):
        intl = "972" + digits
    else:
        return f"📞 [{phone}](tel:{digits})"
    links = f"📞 [{phone}](tel:+{intl})"
    if not intl.startswith("972") or intl[3] == "5":
        links += f" · 💬 [WhatsApp](https://wa.me/{intl})"
    return links


def show_qa(qa):
    st.markdown(f"### {qa['question']}")
    color = CATEGORY_STYLE.get(qa["category"], ("", "gray"))[1]
    st.markdown(f":{color}-badge[{category_label(qa['category'])}]")
    for a in sorted(qa["answers"], key=lambda a: -a.get("count", 1)):
        parts = []
        if a.get("name"):
            # Isolate the name so a Hebrew name doesn't flip the whole line right-to-left.
            parts.append(f"\u2066**{a['name']}**\u2069")
        if a.get("phone"):
            parts.append(phone_links(a["phone"]))
        if parts:
            st.markdown(" · ".join(parts))
        meta = []
        if a.get("count", 1) > 1:
            meta.append(f"👍 recommended {a['count']}×")
        if a.get("date"):
            meta.append(f"last mentioned {a['date']}")
        if meta:
            st.caption(" · ".join(meta))
        if note(a):
            st.markdown(f"- {note(a)}")
    st.divider()


def recommendations(qa):
    return sum(a.get("count", 1) for a in qa["answers"])


index, qas = get_data()
categories = sorted({qa["category"] for qa in qas})


# Search and category browsing replace each other: whichever was used last wins.
def on_search():
    st.session_state.category = None


def on_category():
    st.session_state.query = ""


query = st.text_input(
    "Search", placeholder="e.g. renovation contractor", key="query", on_change=on_search
)
category = st.pills(
    "Or browse a category", categories, key="category", on_change=on_category,
    format_func=category_label,
)


def count(n):
    st.caption(f"{n} result{'s' if n != 1 else ''}")


if query.strip():
    # Search always covers every category.
    results = search(index, qas, query, None)
    stripped = query.strip()
    if st.session_state.get("last_logged") != stripped:
        log_search(stripped, results)
        st.session_state.last_logged = stripped
    if not results:
        st.info(
            "No results. This may not have come up in the group yet — "
            "try different words, or ask the group directly."
        )
    else:
        count(len(results))
        st.caption("Are these results helpful?")
        st.feedback(
            "thumbs", key=f"rating_{stripped}",
            on_change=lambda q=stripped, r=results: log_rating(
                q, "up" if st.session_state[f"rating_{q}"] == 1 else "down", r
            ),
        )
    for qa in results:
        show_qa(qa)
elif category:
    in_category = [qa for qa in qas if qa["category"] == category]
    count(len(in_category))
    for qa in sorted(in_category, key=recommendations, reverse=True):
        show_qa(qa)
else:
    st.markdown(
        "Recommendations shared in the TLV kids group. "
        "Search above or pick a category."
    )
    for qa in sorted(qas, key=recommendations, reverse=True)[:5]:
        show_qa(qa)

with st.form("feedback", clear_on_submit=True):
    text = st.text_area("Suggestion, bug, or a recommendation we're missing?")
    if st.form_submit_button("Send") and text.strip():
        log_feedback(text.strip())
        st.success("Thanks!")

st.caption(
    "Want to be removed from a listing? "
    "Email [bennymestel@gmail.com](mailto:bennymestel@gmail.com)"
)
