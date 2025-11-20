from joblib import load
from src.preprocess import Preprocess
from scipy.sparse import csr_matrix
import os
import numpy as np

class InferenceModel(object):
    def __init__(self,
                model_path: str = os.getenv("MODEL_PATH", "assets/Logistic Regression_v1.0.0.joblib"), # Kalo misalkan ModelPathnya emptry, 
                vectorizer_path: str = os.getenv("VECTORIZER_PATH", 'assets/vectorizer.pickle'),
                classes: list = ['negative', 'positive']
                ) -> None:
        self.model = load(model_path)
        self.classes = classes
        self.preprocess_class: Preprocess = Preprocess(vectorizer_path=vectorizer_path)
        
        self.uncertainty_threshold = float(os.getenv("UNCERTAINTY_THRESHOLD", 0.6))
    
    def preprocess(self, text: str) -> csr_matrix:
        return self.preprocess_class.run_preprocess(text=text)
    
    def postprocess(self, probs: np.array):
         # Label, Confidence, Uncertainty Score, and is_uncertainty flag
        pred_index = probs.argmax() # index value terbesar 0
        prediction = self.classes[pred_index]
        conf = probs[pred_index]
        # Implement Monitoring Performance untuk Setiap inputan yang ada
        ## Uncertainty Sampling -> Uncertainty Score
        is_uncertain, uncertainty_score = self.uncertainty_detection(probs)
        # is_uncertain, uncertainty_score = 0, 0
        return prediction, conf, is_uncertain, uncertainty_score
    
    def uncertainty_detection(self, probs: np.array):
        probs = probs.tolist()
        probs = sorted(probs, reverse=True)
        most_conf = probs[0]
        next_most_conf = probs[1]
        uncertainty_score = float(1 - (most_conf - next_most_conf))
        is_uncertain = 1 if uncertainty_score > self.uncertainty_threshold else 0
        return is_uncertain, uncertainty_score
    
    def predict(self, text: str) -> dict:
        preprocessed_text = self.preprocess(text=text) # [[input_preprocess]]
        probs = self.model.predict_proba(preprocessed_text)[0] # [[0.95, 0.05]] -> [0.95, 0.05]
        prediction, conf, is_uncertain, uncertainty_score = self.postprocess(probs=probs)

        return {
            "text": text,
            "prediction": prediction,
            "confidence": float(conf),
            "is_uncertain": is_uncertain,
            "uncertainty_score": uncertainty_score
        }