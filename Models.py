from abc import ABC, abstractmethod
from joblib import load
import requests
from sentence_transformers import SentenceTransformer
from preprocess_texts import clean_unicode
class PreprocessingDecorator(ABC):
    def __init__(self, sentiment_model: 'SentimentAnalysisModel'):
        self.sentiment_model = sentiment_model

    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass

    def common_preprocess(self, text: str) -> str:
        # Implement common preprocessing steps
        #return text.lower().strip()
        unicode_removed=list(map(clean_unicode, text))
        return  list(map(clean_unicode, unicode_removed))
            
    

    def analyze(self, text: str) -> float:
        preprocessed_text = self.common_preprocess(text)
        preprocessed_text = self.preprocess(preprocessed_text)
        return self.sentiment_model.analyze(preprocessed_text)

class XgboostPreprocessor(PreprocessingDecorator):

    def preprocess(self, text: str) -> str:
        # Implement preprocessing steps specific to Model A
        return text





class BertPreprocessor(PreprocessingDecorator):

    def preprocess(self, text: str) -> str:
        # Implement preprocessing steps specific to Model B
        return text.upper().strip()

class SentimentAnalysisModel(ABC):
    def __init__(self, version: str):
        self.version = version

    @abstractmethod
    def analyze(self, text: str) -> float:
        pass

class XGBOOST(SentimentAnalysisModel):
    def load_model(self):
        with open(f'models/{self.__class__.__name__.lower()}/{self.version}.joblib', 'rb') as file:
            print(file.name)
            loaded_model = load(file)

        return loaded_model
    
    def load_embedding(self):
        #return( self.sentiment_model.version)
        try:
            return SentenceTransformer(f'sentence-transformers/{self.version}')
        except requests.exceptions.ConnectTimeout as e:
            print(f"Connection timed out: {e}")
    

    def analyze(self, text: str) -> float:
        # Implement sentiment analysis logic for Model A
        self.model= self.load_model()
        sentence_model=self.load_embedding()
        print(text)
        embedded = sentence_model.encode([text])
        return self.model.predict(embedded)

class BERT(SentimentAnalysisModel):


    def analyze(self, text: str) -> float:
        # Implement sentiment analysis logic for Model B
        return 0.6

class SentimentAnalysisService:
    def __init__(self):
        self.models = {
            'model_a_v1': XgboostPreprocessor(XGBOOST(version='paraphrase-multilingual-mpnet-base-v2')),
            'model_a_v2': XgboostPreprocessor(XGBOOST(version='paraphrase-multilingual-mpnet-base-v2')),
        }

    def analyze_sentiment(self, model_name: str, text: str) -> float:
        if model_name not in self.models:
            raise ValueError(f"Invalid model name: {model_name}")
        return self.models[model_name].analyze(text=text)

# Example usage
service = SentimentAnalysisService()


print(service.analyze_sentiment(model_name='model_a_v1', text="it is fun\u2020"))  # Output: 0.8
#print(service.analyze_sentiment('model_a_v2', 'This is a great movie!'))  # Output: 0.8
