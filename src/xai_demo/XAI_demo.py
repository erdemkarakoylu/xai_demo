import streamlit as st

st.set_page_config(
    page_title="XAI, briefly",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

st.markdown("""
            [Explainable AI](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence): 
            A collection of frameworks to:
                * help understand black-box model predictions;
                * identify data impact on model outputs;
                * facilitate model improvement;
                * build stakeholder trust / buy-in. 
             
            """)