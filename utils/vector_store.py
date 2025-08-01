import pdfplumber
import fitz
from typing import List, Dict, Tuple
from langchain_community.vectorstores import PathwayVectorClient

class PDFProcessor:
    def _init_(self):
        self.vector_store = PathwayVectorClient(
            url="https://demo-document-indexing.pathway.stream"
        )
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF and extract text and metadata"""
        text = self.extract_text_from_pdf(pdf_path)
        bold_text = self.extract_bold_text(pdf_path)
        
        metadata = {
            'id': pdf_path.split('/')[-1],
            'text': text,
            'bold_text': bold_text,
            'path': pdf_path
        }
        
        return metadata
    
    def extract_bold_text(self, pdf_path: str) -> List[str]:
        """Extract bold text from PDF"""
        bold_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(
                    keep_blank_chars=True,
                    use_text_flow=True,
                    extra_attrs=['fontname']
                )
                bold_words = [word['text'] for word in words if 'NimbusRomNo9L-Medi' in word.get('fontname', '')]
                bold_text.extend(bold_words)
        return bold_text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF"""
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    
    def find_similar_papers(self, query_text: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for similar papers in the vector store"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query_text,
                k=k
            )
            return [(doc.metadata, score) for doc, score in results]
        except Exception as e:
            print(f"Error finding similar papers: {str(e)}")
            return []
