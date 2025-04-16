import fitz  # PyMuPDF

class PDFProcessor:
    def is_text_based(self, pdf_path):
    
        doc = fitz.open(pdf_path)
        text_content = ""
        
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text_content += page.get_text()
        
        doc.close()
        
    
        return len(text_content.strip()) > 100
    
    def extract_text(self, pdf_path):

        text_content = ""
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_content += page.get_text()
        
        doc.close()
        return text_content