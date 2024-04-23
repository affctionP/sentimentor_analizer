import streamlit as st
import os
from classifiers import ModelClassifier

MODEL_NAME=['logistic_regression','random_forest','xgboost']
def main():


    
    select = st.selectbox("Choose an clssifier", options=MODEL_NAME, index=0)


    comment = st.text_input("enter your text")
    if st.button("شروع"):
        cls=ModelClassifier("xgboost","paraphrase-multilingual-mpnet-base-v2")
        sentense=cls.predict([comment])

        if sentense[0] == 0:
            st.write("😡")
        elif sentense[0] == 1:
            st.write("😐")
        else :
            st.write("😊")


if __name__ == '__main__':
    main()