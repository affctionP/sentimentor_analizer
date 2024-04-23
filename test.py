from abc import ABC, abstractmethod

class PreprocessingDecorator(ABC):
    def __init__(self, sentiment_model: 'SentimentAnalysisModel'):
        self.sentiment_model = sentiment_model

    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass

    def common_preprocess(self, text: str) -> str:
        # Implement common preprocessing steps
        return text.lower().strip()
    

    def analyze(self, text: str) -> float:
        preprocessed_text = self.common_preprocess(text)
        preprocessed_text = self.preprocess(preprocessed_text)
        return self.sentiment_model.analyze(preprocessed_text)

class XgboostPreprocessor(PreprocessingDecorator):

    def preprocess(self, text: str) -> str:
        # Implement preprocessing steps specific to Model A
        return text.lower().strip()

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
        pass
    def analyze(self, text: str) -> float:
        # Implement sentiment analysis logic for Model A
        return 0.8

class BERT(SentimentAnalysisModel):


    def analyze(self, text: str) -> float:
        # Implement sentiment analysis logic for Model B
        return 0.6

class SentimentAnalysisService:
    def __init__(self):
        self.models = {
            'model_a_v1': XgboostPreprocessor(XGBOOST(version='v1')),
            'model_a_v2': XgboostPreprocessor(XGBOOST(version='v2')),
        }

    def analyze_sentiment(self, model_name: str, text: str) -> float:
        if model_name not in self.models:
            raise ValueError(f"Invalid model name: {model_name}")
        return self.models[model_name].analyze(text)

# Example usage
service = SentimentAnalysisService()
print(service.analyze_sentiment('model_a_v1', 'This is a great movie!'))  # Output: 0.8
print(service.analyze_sentiment('model_a_v2', 'This is a great movie!'))  # Output: 0.8
