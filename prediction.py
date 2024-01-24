# -*- coding: utf-8 -*-
"""predicition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tbNxT28UmgnvD-kzHUA78q7MeQykfnMD
"""

import joblib
def predict(data):
    clf = joblib.load('rf_model.sav')
    return clf.predict(data)

!pip install streamlit

import streamlit as st
import numpy as np
import joblib

# Assume the predict function is defined as mentioned in the previous response

st.header('Plant Features')

col1, col2 = st.columns(2)

with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text('Petal characteristics')
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

if st.button('Predict type of Iris'):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])