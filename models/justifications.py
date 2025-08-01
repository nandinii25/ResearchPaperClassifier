import torch
from transformers import pipeline
from config.settings import DEVICE, CONFERENCES

class JustificationGenerator:
    def _init_(self):
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=DEVICE
        )
    
    def generate_justification(self, pdf_text: str, conference_name: str) -> str:
        """Generate a justification for why a paper fits a particular conference"""
        abstract = pdf_text[:2000]
        
        prompt = f"""Based on the following paper abstract, provide a specific justification (4-5 points) 
        for why this paper would be suitable for the {conference_name} conference. Focus on matching the paper's 
        topics and methodologies with the conference's themes. Avoid summarizing the paper.
        
        Conference description: {CONFERENCES[conference_name]}
        
        Paper abstract: {abstract}
        
        Provide justification starting with 'This paper aligns with {conference_name} because'"""
        
        response = self.pipe(prompt, max_new_tokens=150, 
                           do_sample=True, 
                           temperature=0.7,
                           top_p=0.9)[0]['generated_text']
        
        start_phrase = f"This paper aligns with {conference_name} because"
        if start_phrase in response:
            justification = response[response.index(start_phrase):]
            justification = justification.strip()
            if '.' in justification:
                justification = '.'.join(justification.split('.')[:-1]) + '.'
            return justification
        else:
            return f"This paper aligns with {conference_name} because it addresses key themes in {CONFERENCES[conference_name].split('.')[0]}"
