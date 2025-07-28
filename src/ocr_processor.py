


import logging
from typing import List, Dict, Any
import fitz
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

class OCRProcessor:
  
    def __init__(self, dpi: int = 300):
        self.logger = logging.getLogger(__name__)
        self.dpi = dpi
        
        self.tesseract_config = '--oem 3 --psm 3'

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
       
        
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        
        binary_img = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary_img

    def process_scanned_pages(self, doc: fitz.Document, 
                            classifications: List[str], lang: str = 'eng') -> Dict[int, List[Dict[str, Any]]]:
        
        ocr_results = {}
        for page_num, classification in enumerate(classifications):
            if classification == 'scanned':
                try:
                    page = doc[page_num]
                    mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    
                    processed_img = self._preprocess_image(img_array)
                    pil_image = Image.fromarray(processed_img)

                    
                    ocr_data = pytesseract.image_to_data(
                        pil_image, lang=lang, config=self.tesseract_config, output_type=pytesseract.Output.DICT
                    )

                    page_spans = []
                    scale_factor = 72 / self.dpi
                    for i in range(len(ocr_data['text'])):
                        text = ocr_data['text'][i].strip()
                        conf = int(ocr_data['conf'][i])
                        if conf > 50 and text: # Use a confidence threshold
                            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                                          ocr_data['width'][i], ocr_data['height'][i])
                            
                            bbox = (x * scale_factor, y * scale_factor, 
                                    (x + w) * scale_factor, (y + h) * scale_factor)

                            page_spans.append({
                                'text': text, 'bbox': bbox,
                                'font_name': "OCR-Font", 'font_size': h * scale_factor,
                                'flags': 0, 'page_num': page_num, 'source': 'ocr'
                            })
                    
                    ocr_results[page_num] = page_spans
                    self.logger.info(f"OCR processed page {page_num + 1} using language '{lang}'.")
                    
                except Exception as e:
                    self.logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
                    ocr_results[page_num] = []
        
        return ocr_results
