from sentence_transformers import SentenceTransformer, util
from config.settings import DEVICE, CONFERENCES

class ConferenceClassifier:
    def _init_(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.model = self.model.to(DEVICE)
    
    def find_conference_scores(self, pdf_text: str) -> dict:
        """Calculate similarity scores between the paper and each conference's description"""
        paper_embedding = self.model.encode(pdf_text, convert_to_tensor=True).to(DEVICE)
        
        similarities = {}
        for name, desc in CONFERENCES.items():
            conf_embedding = self.model.encode(desc, convert_to_tensor=True).to(DEVICE)
            sim = float(util.cos_sim(paper_embedding.unsqueeze(0), conf_embedding.unsqueeze(0))[0][0])
            similarities[name] = sim
        
        return similarities
