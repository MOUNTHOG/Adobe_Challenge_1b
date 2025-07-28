




import logging
from typing import List, Dict, Any

class OutputGenerator:
    

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_output(self, title: str, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
       
        try:
            outline = self._format_outline(headings)

            output = {
                'title': title,
                'outline': outline
            }
            self.logger.info(f"Assembled final output for title: '{title}'")
            return output
        except Exception as e:
            self.logger.error(f"Error assembling final output: {e}")
            return {'title': 'Error Assembling Document', 'outline': []}

    def _format_outline(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       
        outline = []
        for heading in headings:
            text = heading['text'].strip()
            if text:
                outline.append({
                    'level': heading.get('assigned_level', 'H3'),
                    'text': text,
                    'page': heading['page_num'] + 1
                })
        return outline