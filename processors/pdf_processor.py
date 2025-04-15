import fitz  # PyMuPDF

class PDFProcessor:
    def is_text_based(self, pdf_path):
        """
        Determine if a PDF is text-based or scanned/image-based.
        Returns True if text-based, False if likely scanned.
        """
        doc = fitz.open(pdf_path)
        text_content = ""
        
        # Check first few pages
        pages_to_check = min(3, len(doc))
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text_content += page.get_text()
        
        doc.close()
        
        # If we found substantial text, assume it's a text-based PDF
        # This is a simple heuristic - could be improved
        return len(text_content.strip()) > 100
    
    def extract_text(self, pdf_path):
        """
        Extract text from a text-based PDF.
        """
        text_content = ""
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_content += page.get_text()
        
        doc.close()
        return text_content