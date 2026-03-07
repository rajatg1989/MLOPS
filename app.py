import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Iris Flower Predictor")

sepal_length = st.slider("Sepal Length", 4.0, 8.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0)
petal_length = st.slider("Petal Length", 1.0, 7.0)
petal_width = st.slider("Petal Width", 0.1, 3.0)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    
    prediction = model.predict(features)

    flower = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Prediction: {flower[prediction[0]]}")