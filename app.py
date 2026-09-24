import streamlit as st

from search import load, search

st.set_page_config(page_title="TLV Kids KB", page_icon="assets/icon.jpg")
st.title("TLV Kids Knowledge Base")


@st.cache_resource
def get_data():
    return load()


index, qas = get_data()
categories = sorted({qa["category"] for qa in qas})

query = st.text_input("Search", placeholder="e.g. renovation contractor")
category = st.selectbox("Category", ["All"] + categories)
category = None if category == "All" else category

results = search(index, qas, query, category)

if not results:
    st.info("No results.")

for qa in results:
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

st.caption(
    "Found a bug, have a suggestion, or listed here and want to be removed? "
    "Email [bennymestel@gmail.com](mailto:bennymestel@gmail.com)"
)
