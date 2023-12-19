from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_PATH = Path.cwd().parent.parent
ASSETS_PATH = PROJECT_PATH / 'assets'

st.subheader("Know your penguins:")
galleries = st.columns(3)


def load_and_set_image(image_path, width=400, height=600):
    image = Image.open(image_path)
    return image.resize((width, height))

with galleries[0]:
    pic_path = ASSETS_PATH / 'Adelie.jpg'
    resized_pic = load_and_set_image(pic_path)
    st.image(resized_pic, caption='Adelie')
with galleries[1]:
    pic_path = ASSETS_PATH / 'Chinstrap.jpg'
    resized_pic = load_and_set_image(pic_path)
    st.image(resized_pic, caption='Chinstrap')
with galleries[2]:
    pic_path = ASSETS_PATH / 'Gentoo.jpg'
    resized_pic = load_and_set_image(pic_path)
    st.image(resized_pic, caption='Gentoo')

st.subheader("Species-specific morpological trait distributions")
columns = st.columns(3)
with columns[0]:
    pic_path = ASSETS_PATH / 'hist_bill_depth_mm.png'
    st.image(pic_path.as_posix(), caption='')

with columns[1]:
    pic_path = ASSETS_PATH / 'hist_flipper_length_mm.png'
    st.image(pic_path.as_posix(), caption='')

with columns[2]:
    pic_path = ASSETS_PATH / 'hist_bill_length_mm.png'
    st.image(pic_path.as_posix(), caption='')

st.image((ASSETS_PATH / 'penguins_pairplot.png').as_posix())

