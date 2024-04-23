

import warnings
import xgboost
from sentence_transformers import SentenceTransformer
from joblib import load
import streamlit as st
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
class  ModelClassifier:
    def __init__(self,name,version) -> None:
        self.name=name
        self.version=version
        self.model = self._get_classifier(name)


    def _get_classifier(self, name):
        classifiers = {


            'random_forest': self.load_model(),
            'svm':  self.load_model(),
            'xgboost': self.load_model(),
        }
        
        return classifiers.get(name.lower(), None)
    
    def load_model(self):
        # Load model from file
        
        with open(f'models/{self.name}/{self.version}.joblib', 'rb') as file:
            print(file.name)
            loaded_model = load(file)

        return loaded_model
    st.cache_resource(show_spinner="wait embedding is loading ")
    def load_embedding(self):
        try:
            return SentenceTransformer(f'sentence-transformers/{self.version}')
        except requests.exceptions.ConnectTimeout as e:
            print(f"Connection timed out: {e}")
            return None

        
        
    def predict(self, X_test):
        if self.model is None:
            raise ValueError(f"Invalid classifier name: {self.name}")
        sentence_model = self.load_embedding()
        embedded = sentence_model.encode(X_test)
        return self.model.predict(embedded)
    

    


