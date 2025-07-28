
import logging
import re
from typing import List, Dict, Any

class HeadingDetector:
    """Detects heading candidates using a balanced scoring model and multi-column awareness."""
    def __init__(self, line_tolerance: float = 2.0, score_threshold: float = 4.5):
        self.line_tolerance = line_tolerance
        self.score_threshold = score_threshold
        self.logger = logging.getLogger(__name__)
        self.number_prefix_regex = re.compile(r'^\s*(\d+(\.\d+)*|[A-Z]|[一二三四五六七八九十])\.?\s+')

    def _group_spans_into_lines(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not spans:
            return []
        
        spans.sort(key=lambda s: (s['page_num'], s['bbox'][1], s['bbox'][0]))
        
        lines = []
        current_line_spans = []
        
        for span in spans:
            if not current_line_spans or \
               span['page_num'] != current_line_spans[0]['page_num'] or \
               abs(span['bbox'][1] - current_line_spans[0]['bbox'][1]) > self.line_tolerance:
                
                if current_line_spans:
                    lines.append(self._merge_line_spans(current_line_spans))
                current_line_spans = [span]
            else:
                current_line_spans.append(span)
        
        if current_line_spans:
            lines.append(self._merge_line_spans(current_line_spans))
            
        return [line for line in lines if line and line['text'].strip()]

    def _merge_line_spans(self, line_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merges a list of spans on the same line into a single line dictionary."""
        if not line_spans:
            return None

        line_spans.sort(key=lambda s: s['bbox'][0])
        text = ' '.join(s['text'].strip() for s in line_spans if s['text'].strip())
        
        if not text:
            return None

        x0 = min(s['bbox'][0] for s in line_spans)
        y0 = min(s['bbox'][1] for s in line_spans)
        x1 = max(s['bbox'][2] for s in line_spans)
        y1 = max(s['bbox'][3] for s in line_spans)
        
        main_span = max(line_spans, key=lambda s: (s['font_size'], (s['bbox'][2] - s['bbox'][0])))
        is_bold = "bold" in main_span.get('font_name', '').lower() or (main_span.get('flags', 0) & 2**4)

        return {
            'text': text,
            'bbox': (x0, y0, x1, y1),
            'font_name': main_span['font_name'],
            'font_size': round(main_span['font_size'], 1),
            'page_num': main_span['page_num'],
            'is_bold': is_bold
        }

    def _merge_multiline_headings(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges headings that span multiple lines."""
        if not lines:
            return []

        merged_lines = []
        i = 0
        while i < len(lines):
            current_line = lines[i]
            
            while i + 1 < len(lines):
                next_line = lines[i+1]
                
                is_same_page = current_line['page_num'] == next_line['page_num']
                is_similar_font = abs(current_line['font_size'] - next_line['font_size']) < 1.0 and \
                                  current_line['is_bold'] == next_line['is_bold']
                is_not_new_heading = not self.number_prefix_regex.match(next_line['text'])
                line_height = current_line['bbox'][3] - current_line['bbox'][1]
                is_close_vertically = (next_line['bbox'][1] - current_line['bbox'][3]) < (line_height * 0.8)

                if is_same_page and is_similar_font and is_not_new_heading and is_close_vertically:
                    current_line['text'] += ' ' + next_line['text']
                    current_line['bbox'] = (
                        min(current_line['bbox'][0], next_line['bbox'][0]),
                        current_line['bbox'][1],
                        max(current_line['bbox'][2], next_line['bbox'][2]),
                        next_line['bbox'][3]
                    )
                    i += 1 
                else:
                    break
            
            merged_lines.append(current_line)
            i += 1
        
        return merged_lines

    def detect_candidates(self, spans: List[Dict[str, Any]], font_clusters: Dict[tuple, str]) -> List[Dict[str, Any]]:
        """Detects heading candidates using a semantic and visual scoring model."""
        lines = self._group_spans_into_lines(spans)
        processed_lines = self._merge_multiline_headings(lines)
        
        candidates = []
        for i, line in enumerate(processed_lines):
            text = line['text']
            score = 0.0

            # Visual cues
            font_key = (line['font_name'], line['font_size'])
            cluster = font_clusters.get(font_key, 'Body')
            if cluster == 'H1': score += 5.0
            elif cluster == 'H2': score += 4.0
            elif cluster == 'H3': score += 3.0

            if line.get('is_bold', False): score += 2.5

            if i > 0 and line['page_num'] == processed_lines[i-1]['page_num']:
                vertical_gap = line['bbox'][1] - processed_lines[i-1]['bbox'][3]
                line_height = line['bbox'][3] - line['bbox'][1]
                if vertical_gap > line_height * 0.5: score += 1.5

            numbering_match = self.number_prefix_regex.match(text)
            if numbering_match: score += 4.0

            if text.strip().endswith('?'):
                score += 2.5

            if score >= self.score_threshold:
                numbering, numbering_level = None, 0
                if numbering_match:
                    numbering = numbering_match.group(0).strip()
                    clean_numbering = numbering.strip().rstrip('.')
                    numbering_level = len(clean_numbering.split('.'))

                candidates.append({
                    'text': text, 'bbox': line['bbox'], 'page_num': line['page_num'],
                    'font_size': line['font_size'], 'font_cluster': cluster,
                    'numbering': numbering, 'numbering_level': numbering_level, 'score': score
                })

        candidates.sort(key=lambda x: (-x['score'], x['page_num'], x['bbox'][1]))
        self.logger.info(f"Detected {len(candidates)} candidates with final scoring model.")
        return candidates