



# src/hierarchy_builder.py
import logging
import re
from typing import List, Dict, Any
from collections import Counter

class HierarchyBuilder:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_hierarchy(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if not headings:
            return []
        
    
        self._assign_relative_levels(headings)
        
        self.logger.info(f"Successfully built final relative hierarchy for {len(headings)} headings.")
        return headings

    def _assign_relative_levels(self, headings: List[Dict[str, Any]]):
    
        if not headings:
            return

        last_assigned_level = 0
        for heading in headings:
            numbering_level = heading.get('numbering_level', 0)
            if numbering_level > 0:
                current_level = numbering_level
            else:
                font_cluster = heading.get('font_cluster', 'H3')
                font_level = int(re.sub(r'\D', '', font_cluster) or 3)
                
                if last_assigned_level > 0:
                    current_level = min(font_level, last_assigned_level + 1)
                else:
                    current_level = font_level

            if last_assigned_level > 0 and current_level > last_assigned_level + 1:
                current_level = last_assigned_level + 1
            
            final_level = max(1, min(current_level, 4))
            heading['assigned_level'] = f"H{final_level}"
            last_assigned_level = final_level

    def cleanup_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       
        if not headings:
            return []

        
        headers_to_remove = set()
        if len(headings) > 3:
            num_pages = headings[-1]['page_num'] - headings[0]['page_num'] + 1
            if num_pages > 2:
                text_counter = Counter(h['text'].strip().lower() for h in headings)
                for text, count in text_counter.items():
                   
                    if count > num_pages * 0.4:
                        is_header_or_footer = False
                        for h in headings:
                            if h['text'].strip().lower() == text:
                                
                                if h['bbox'][1] < 120 or h['bbox'][3] > 680:
                                    is_header_or_footer = True
                                    break
                        if is_header_or_footer:
                            self.logger.info(f"CLEANUP: Identifying running header/footer: '{text}'")
                            headers_to_remove.add(text)

      
        cleaned, seen_on_page = [], {}
        for heading in headings:
            text = re.sub(r'\s+', ' ', heading['text']).strip()

            if text.lower() in headers_to_remove:
                continue

           
            page_key = heading['page_num']
            if page_key not in seen_on_page:
                seen_on_page[page_key] = set()
            
            if text.lower() in seen_on_page[page_key]:
                continue
            
            seen_on_page[page_key].add(text.lower())
            heading['text'] = text
            cleaned.append(heading)

        self.logger.info(f"Cleanup: {len(headings)} -> {len(cleaned)} headings.")
        return cleaned