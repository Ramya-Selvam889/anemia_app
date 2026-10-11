import streamlit as st
import numpy as np
from PIL import Image

st.title("Anemia Screening - Test")
img = st.camera_input("Conjunctiva photo edunga")
if img:
    im = Image.open(img)
    st.image(im, caption="Captured")
    st.write("Image size:", im.size, "| Mean brightness:", round(np.array(im).mean(), 1))