

import logging
import re
from typing import List, Dict, Any

class HeadingFilter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.table_headers = {
            'remarks', 'syllabus', 'days', 'identifier', 'reference', 'version', 'date',
            'name', 'age', 's.no', 'relationship', 'designation', 'description'
        }
        self.toc_pattern = re.compile(r'\s[\._\s]{4,}\s*\d+\s*$')
        self.noise_pattern = re.compile(r'^[•\-\–\*\d\s\.]*$')

    def _is_invalid(self, text: str, page_num: int) -> bool:
        if not text or len(text.strip()) < 3 or self.noise_pattern.fullmatch(text.strip()):
            return True

        text_lower = text.lower().strip()
        words = text_lower.split()

        if text.strip().endswith('.') and len(words) > 8 and not self.toc_pattern.search(text):
            return True

        if self.toc_pattern.search(text):
            return True
            
        if len(words) <= 2 and text_lower in self.table_headers:
            return True

        if 'page' in text_lower and any(char.isdigit() for char in text_lower):
            if re.search(r'page\s*\d+\s*(of\s*\d+)?', text_lower):
                return True

        if text.isupper() and len(words) < 3:
            return True

        return False

    def filter_headings(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if not candidates:
            return []
        
        filtered_candidates = []
        for candidate in candidates:
            text = candidate.get('text', '').strip()
            page_num = candidate.get('page_num', 0)
            
            if not self._is_invalid(text, page_num):
                filtered_candidates.append(candidate)
            else:
                self.logger.info(f"FILTERED (intelligent rules): '{text}'")

        return filtered_candidates