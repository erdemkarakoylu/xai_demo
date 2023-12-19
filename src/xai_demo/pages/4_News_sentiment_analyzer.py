from pathlib import Path

import loguru
import shap
import streamlit as st
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
    shap.plot.text(shap_vals_4_lbl)


classifier = transformers.pipeline(
    model=model_path, task='sentiment-analysis')


text = st.text_area(
        "Enter text to be interpreted",
        "I like you, I love you",
        height=200,
        max_chars=500,
    )
columns = st.columns([1, 3])
with columns[0]:
    if st.button("Predict Sentiment"):  
        prediction = run_prediction(text)
with columns[1]:
    if st.button("Interpret Result"):
        run_explainer(text, prediction['label'])


