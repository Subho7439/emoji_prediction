import streamlit as st
import pandas as pd

import pickle
ss = pickle.load(open("scaler.pkl", "rb"))
rf = pickle.load(open("model.pkl", "rb"))

st.title("HOUSE PRICE PREDICTION SYSTEM")
st.markdown("### Dataset that we have used to build the system : ")
data = pd.read_csv("california_housing_test.csv")

st.dataframe(data.head(10))
st.bar_chart(data.head(50))
long = st.sidebar.number_input("longitude : ")
lang = st.sidebar.number_input("langitude : ")
age = st.sidebar.number_input("House_age : ")
room = st.sidebar.number_input("total_rooms : ")
bed = st.sidebar.number_input("total_bedrooms : ")
ppl = st.sidebar.number_input("popolation : ")
hold = st.sidebar.number_input("house_hold: ")
sal = st.sidebar.number_input("salary : ")
v = st.sidebar.button("SUBMIT")
if v:
    st.sidebar.success("Input accepted")
    test = [[long,lang,age, room, bed, ppl,hold,sal]]
    test = ss.transform(test)
    predict = rf.predict(test)
    st.sidebar.write("House price prediction : ",predict)