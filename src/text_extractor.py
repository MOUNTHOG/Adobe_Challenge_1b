


import fitz
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.utils.text_processing import clean_text
from langdetect import detect, LangDetectException

class TextExtractor:
   

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_spans_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
       
        spans = []
        # (Internal logic for span extraction remains the same)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = clean_text(span["text"])
                        if text:
                            spans.append({
                                'text': text, 'bbox': span["bbox"], 'font_name': span["font"],
                                'font_size': span["size"], 'flags': span["flags"],
                                'page_num': page_num, 'source': 'digital'
                            })
        return spans
    
    def detect_language(self, text: str) -> str:
        if not text or len(text.strip()) < 50: 
            return 'eng'
        try:
           
            lang_code = detect(text)
            
            lang_map = {'en': 'eng', 'fr': 'fra', 'es': 'spa', 'de': 'deu', 'zh-cn': 'chi_sim', 'ja': 'jpn'}
            return lang_map.get(lang_code, 'eng') # Default to English
        except LangDetectException:
            self.logger.warning("Language detection failed. Defaulting to English.")
            return 'eng'

    def extract_text_with_layout(self, doc: fitz.Document,
                               classifications: List[str],
                               ocr_results: Dict[int, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], str, str]:
        
        all_spans = []
        for page_num, classification in enumerate(classifications):
            if classification == 'digital':
                all_spans.extend(self.extract_spans_from_page(doc[page_num], page_num))
            else:
                all_spans.extend(ocr_results.get(page_num, []))

        
        all_spans.sort(key=lambda x: (x['page_num'], x['bbox'][1], x['bbox'][0]))
        full_text = ' '.join(span['text'] for span in all_spans)
        full_text = clean_text(full_text)

        detected_lang = self.detect_language(full_text)
        self.logger.info(f"Detected document language: {detected_lang}")

        return all_spans, full_text, detected_lang
