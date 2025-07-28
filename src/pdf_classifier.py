

import logging
import fitz
from pathlib import Path
from typing import List, Tuple

class PDFClassifier:
    

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def classify_pages(self, pdf_path: Path) -> Tuple[fitz.Document, List[str]]:
       
        doc = fitz.open(pdf_path)
        classifications = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            
            text = page.get_text().strip()
            
           
            images = page.get_images()
            
            
            if len(text) > 100:  
                classification = 'digital'
            elif len(images) > 0 and len(text) < 50:  
                classification = 'scanned'
            else:
                
                classification = 'digital'
            
            classifications.append(classification)
            
        self.logger.info(f"Classified {len(doc)} pages: "
                        f"{classifications.count('digital')} digital, "
                        f"{classifications.count('scanned')} scanned")
        
        return doc, classifications