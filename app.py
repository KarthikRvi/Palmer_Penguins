import streamlit as st
import os
from PIL import Image
from pickle import load

import pandas as pd
import numpy as np
import pickle


image_path=os.path.abspath(os.path.join(os.getcwd(), "images", "lter_penguins.png"))

st.title("Palmer Penguins Prediction App")

#Reading the Image File
img= Image.open(image_path)
st.image(img, use_column_width= True, width=700)


if st.button("Meet the Palmer Penguins"):
    img=Image.open('images/lter_1.png')
    st.image(img,width=700, caption="We are the Palmer Penguins 🐧")

scaler = load(open('models/standard_scaler.pkl', 'rb'))
lr_model = load(open('models/lr_model.pkl', 'rb'))

print("Enter Penguins Species Details")
a = st.text_input("Enter the Bill Length", placeholder="Enter value in mm")
b = st.text_input("Enter the Bill Depth", placeholder="Enter value in mm")
c = st.text_input("Enter the Flipper Length", placeholder="Enter value in mm")
d = st.text_input("Enter the Body Mass", placeholder="Enter value in g")
e = st.text_input("Enter the island_Biscoe", placeholder="Enter 0 or 1")
f = st.text_input("Enter the island_Dream", placeholder="Enter 0 or 1")
g = st.text_input("Enter the island_Torgersen", placeholder="Enter 0 or 1")
h = st.text_input("Enter the gender_Female", placeholder="Enter 0 or 1")
i = st.text_input("Enter the gender_Male", placeholder="Enter 0 or 1")

btn_click = st.button("Predict")

if btn_click == True:
    if a and b and c and d and e and f and g and h and i:
        query_point = np.array([float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h), float(i)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = lr_model.predict(query_point_transformed)
        st.success(pred)
        st.text ("Predictor: [0] is Adelie , [1] is Chinstrap and [2] is Gentoo")
    else:
        st.error("Enter the values properly.")