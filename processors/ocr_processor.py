import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
import os

class OCRProcessor:
    def __init__(self):
        # Configure pytesseract path if needed (point directly to the executable)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def process_scanned_pdf(self, pdf_path):

        text_content = ""

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get image list from the page
                image_list = page.get_images(full=True)

                if not image_list:
                    # If no images, render the page as an image
                    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    text_content += text + "\n"
                else:
                    # OCR for each embedded image
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            print(f"Processing image from page {page_num + 1}, image {img_idx + 1}")
                            text = pytesseract.image_to_string(img)
                            text_content += text + "\n"

            doc.close()
        except Exception as e:
            print("Error processing file:", e)

        return text_content

    
    #def process_with_visual_llm(self, pdf_path):
     
        