import streamlit as st

from search import load, search

st.set_page_config(page_title="TLV Kids KB", page_icon="assets/icon.jpg")
st.title("TLV Kids Knowledge Base")


@st.cache_resource
def get_data():
    return load()


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
        st.markdown(f"- {a['text']}")
    st.divider()


def recommendations(qa):
    return sum(a.get("count", 1) for a in qa["answers"])


index, qas = get_data()
categories = sorted({qa["category"] for qa in qas})

query = st.text_input("Search", placeholder="e.g. renovation contractor")

if query.strip():
    # Search always covers every category; categories are only for browsing.
    results = search(index, qas, query, None)
    if not results:
        st.info("No results.")
    for qa in results:
        show_qa(qa)
else:
    st.subheader("Browse by category")
    for category in categories:
        in_category = [qa for qa in qas if qa["category"] == category]
        with st.expander(f"{category} ({len(in_category)})"):
            for qa in sorted(in_category, key=recommendations, reverse=True):
                show_qa(qa)

st.caption(
    "Found a bug, have a suggestion, or listed here and want to be removed? "
    "Email [bennymestel@gmail.com](mailto:bennymestel@gmail.com)"
)
