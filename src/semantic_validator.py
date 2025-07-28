




import logging
from typing import List, Dict, Any, Set

class SemanticValidator:
    
    def __init__(self, penalty_threshold: float = -2.0):
        self.penalty_threshold = penalty_threshold
        self.logger = logging.getLogger(__name__)

    def validate_headings(self, heading_candidates: List[Dict[str, Any]], knowledge_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not heading_candidates: return []
        entity_texts = {ent['text'].strip().lower() for ent in knowledge_graph.get('entities', [])}
        validated = []
        for candidate in heading_candidates:
            text = candidate['text'].strip()
            penalty_score = self._calculate_penalty_score(text, entity_texts)
            if penalty_score > self.penalty_threshold:
                validated.append(candidate)
            else:
                self.logger.info(f"SEMANTICALLY REJECTED: '{text}' (Penalty Score: {penalty_score:.1f})")
        return validated

    def _calculate_penalty_score(self, text: str, entity_texts: Set[str]) -> float:
        """Calculates a penalty score. More negative is worse."""
        score = 0.0
        text_lower = text.lower()
        if text_lower in entity_texts:
            score -= 3.0 
        if len(text.split()) > 15:
            score -= 2.0 
        if text.strip().endswith(('.', ',', ';')):
            score -= 2.5 
        return score