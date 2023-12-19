import streamlit as st

st.set_page_config(
    page_title="XAI - a quick demo",
    page_icon=":christmas_tree:",
    layout='wide'
)

st.sidebar.success("Select from the menu above.")

st.subheader("""
            [Explainable AI: ](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
            """)
st.markdown("A collection of frameworks to:")
st.markdown(" 👉 Help understand black-box model predictions;")
st.markdown(" 👉 Identify data impact on model outputs;")
st.markdown(" 👉 Facilitate model improvement;")
st.markdown(" 👉 Build stakeholder trust / buy-in.")

st.subheader(
    """
    """
    )