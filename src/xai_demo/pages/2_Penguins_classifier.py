from pathlib import Path
from pandas import DataFrame
import streamlit as st
import streamlit.components.v1 as components
import pickle
import shap
import matplotlib.pyplot as pp


st.title("What's this Penguin?")
st.subheader("Real Time Prediction API & Explainer")
#st.sidebar.header("User Features Input")

PROJECT_PATH = Path.cwd()
MODEL_PATH = PROJECT_PATH / 'model_assets'
ASSETS_PATH = PROJECT_PATH / 'assets'

def user_feature_input():
    input_features = dict.fromkeys([
        'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g'
        ])
    input_features["bill_length_mm"] = st.slider(
        "Bill Length (mm)", min_value=25, max_value=65)
    input_features["bill_depth_mm"] = st.slider(
        "Bill Depth (mm)", min_value=5, max_value=25)
    input_features["flipper_length_mm"] = st.slider(
        "Flipper Length (mm)", min_value=100, max_value=240)
    input_features["body_mass_g"] = st.slider(
        "Body Mass (g)", min_value=2500, max_value=7000)
    #return input_features
    return DataFrame.from_dict([input_features])

def load_model_assets():
    with open(MODEL_PATH / 'random_forest_penguins.pkl', 'rb') as rfc_pkl:
        rfc = pickle.load(rfc_pkl)
    with open(MODEL_PATH / 'out_uniques_penguins.pkl', 'rb') as out_pkl:
        label_decoder = pickle.load(out_pkl)
    with open(MODEL_PATH / 'explainer_penguins.pkl', 'rb') as epl:
        explainer = pickle.load(epl)
    return rfc, label_decoder, explainer 

def st_shap(plot, height=800):
    shap_html = f"<head>{shap.getjs()}</head><body{plot.html()}</body>"
    components.html(shap_html, height=height)

def explain_model_prediction():
    #shap_values = explainer.shap_values(data)
    return shap.plots.force(
        explainer(input_df)[:, :, prediction], matplotlib=False)

model, label_decoder, explainer = load_model_assets()

shap.initjs()
poster_path = (ASSETS_PATH / 'penguins.png').as_posix()
st.image(poster_path)
col1, col2 = st.columns(2)
with col1:
    st.text(" ")
    st.text(" ")
    input_df = user_feature_input()
pred_probas = model.predict_proba(input_df)
max_proba = pred_probas.max()
prediction = pred_probas.argmax()
prediction_species = label_decoder[prediction]
predicted_image_path = (ASSETS_PATH / f'{prediction_species}.jpg').as_posix()
with col2:
    st.subheader(
        'Predicted Species:')
    st.image(predicted_image_path, width=300)
    st.subheader(f'{prediction_species} ({100*max_proba:.1f}%)')

st_shap(explain_model_prediction())
#p = shap.plots.force(
#    explainer(input_df)[:, :, prediction], 
#    matplotlib=True, figsize=(20, 6), show=True)
#st.write(p)