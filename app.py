import re

import streamlit as st

from search import load, search

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


def show_qa(qa):
    st.markdown(f"### {qa['question']}")
    st.caption(qa["category"])
    for a in sorted(qa["answers"], key=lambda a: -a.get("count", 1)):
        parts = []
        if a.get("name"):
            parts.append(f"**{a['name']}**")
        if a.get("phone"):
            parts.append(f"📞 {a['phone']}")
        if a.get("count", 1) > 1:
            parts.append(f"recommended ×{a['count']}")
        if a.get("date"):
            parts.append(f"last: {a['date']}")
        if parts:
            st.markdown(" · ".join(parts))
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
category = st.pills("Or browse a category", categories, key="category", on_change=on_category)

if query.strip():
    # Search always covers every category.
    results = search(index, qas, query, None)
    if not results:
        st.info("No results.")
    for qa in results:
        show_qa(qa)
elif category:
    in_category = [qa for qa in qas if qa["category"] == category]
    for qa in sorted(in_category, key=recommendations, reverse=True):
        show_qa(qa)

st.caption(
    "Found a bug, have a suggestion, or listed here and want to be removed? "
    "Email [bennymestel@gmail.com](mailto:bennymestel@gmail.com)"
)
