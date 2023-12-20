from pathlib import Path

import loguru
import matplotlib.pyplot as pp
import shap
import streamlit as st
import streamlit.components.v1 as components
import transformers
from transformers_interpret import SequenceClassificationExplainer

PROJECT_PATH = Path.cwd().parent.parent
MODEL_PATH = PROJECT_PATH / 'model_assets'
ASSETS_PATH = PROJECT_PATH / 'assets'
model_path = MODEL_PATH / 'transformer_model'


def run_prediction(input):
    prediction = classifier(input)[0]
    if prediction['label'] == 'NEGATIVE':
        st.error(
            f"NEGATIVE Sentiment {prediction['score']*100:.2f}%",
            )
    else:
        st.success(
            f"POSITIVE Sentiment {prediction['score']*100:.2f}%"
            )
    return prediction

def run_explainer(input, pred_label):
    explainer = shap.Explainer(classifier)
    shap_values = explainer([input])
    shap_vals_4_lbl = shap_values[:, :, pred_label]
    return shap.plots.text(shap_vals_4_lbl, display=False)


def st_shap(plot, height=800):
    #shap_html = f"<head>{shap.getjs()}</head><body{plot.html()}</body>"
    shap_html = f"<head>{shap.getjs()}</head><body{plot}</body>"
    components.html(shap_html, height=height)

@st.cache_resource
def load_classifier(model_path):
    classifier = transformers.pipeline(
        model=model_path, task='sentiment-analysis')
    return classifier

shap.initjs()



classifier = load_classifier(model_path)
text = st.text_area(
        "Enter text for evaluation",
        "I like you, I love you",
        height=40,
        max_chars=200,
    )
predict = st.button("Predict & Interpret...")

if predict:  
    prediction = run_prediction(text)
    st.markdown(" ")
    with st.spinner("Computing interpretation..."):

        st_shap(run_explainer(text, prediction['label']))

cls_explainer = SequenceClassificationExplainer(
    model=classifier.model, tokenizer=classifier.tokenizer)

