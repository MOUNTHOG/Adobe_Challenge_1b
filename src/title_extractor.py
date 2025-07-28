
import logging
import re 
from typing import List, Dict, Any, Tuple

class TitleExtractor:
   

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _group_spans_into_lines(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       
        if not spans:
            return []
        
        lines = []
        spans.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
        current_line_spans = [spans[0]]
        for span in spans[1:]:
            
            if abs(span['bbox'][1] - current_line_spans[0]['bbox'][1]) < 5:
                current_line_spans.append(span)
            else:
                lines.append(current_line_spans)
                current_line_spans = [span]
        lines.append(current_line_spans)
        return lines

    def extract_title_block(self, first_page_spans: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
       
        if not first_page_spans:
            return "Untitled Document", []

        lines_of_spans = self._group_spans_into_lines(first_page_spans)

        
        page_height = max(s['bbox'][3] for s in first_page_spans)
        top_third_spans = [s for s in first_page_spans if s['bbox'][1] < page_height * 0.3]
        if not top_third_spans:
            top_third_spans = first_page_spans

        max_font_size = max(s['font_size'] for s in top_third_spans)

       
        title_block_lines_of_spans = []
        for line_spans in lines_of_spans:
            line_font_size = max(s['font_size'] for s in line_spans)
            line_y_pos = line_spans[0]['bbox'][1]

           
            if line_y_pos > page_height * 0.4:
                break
            
           
            if line_font_size >= max_font_size * 0.85:
                title_block_lines_of_spans.append(line_spans)
            elif title_block_lines_of_spans:
                
                break
        
        if not title_block_lines_of_spans:
            
            title_block_lines_of_spans.append(lines_of_spans[0])

       
        title_spans = [span for line in title_block_lines_of_spans for span in line]
        
        title_text = ' '.join(span['text'] for span in sorted(title_spans, key=lambda s: (s['bbox'][1], s['bbox'][0])))
        title_text = re.sub(r'\s+', ' ', title_text).strip()

        self.logger.info(f"Extracted title block: '{title_text}'")
        return title_text, title_spans