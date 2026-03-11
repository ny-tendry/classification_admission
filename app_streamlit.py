import joblib 
import numpy as np
import streamlit as st
import time

def predict(X, w, b):
    z = np.dot(X, w) + b
    p = 1/(1+np.exp(-z))
    return p

params = joblib.load("./model.pkl")

w = params['weight']
b = params['bias']

st.title("Classication d' admission")

examen1 = st.number_input("Note à l' examen 1 :")
examen2 = st.number_input("Note à l' examen 2 :")

X = np.array((float(examen1), float(examen2)))
X = X.reshape(1, 2)

if st.button("Predire"):
    resultat = predict(X, w, b)
    time.sleep(2)
    if resultat > 0.5:
        st.success("Féliciations! vous êtes admis")
    else:
        st.error("Desole! vous n' êtes pas admis")
