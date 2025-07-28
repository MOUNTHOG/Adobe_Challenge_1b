
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans

class FontClusterer:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.logger = logging.getLogger(__name__)

    def cluster_fonts(self, spans: List[Dict[str, Any]]) -> Dict[Tuple, str]:
        if not spans: return {}
        try:
            style_counts = Counter()
            for span in spans:
                is_bold = "bold" in span.get('font_name', '').lower() or (span.get('flags', 0) & 2**4)
                key = (span.get('font_name', 'default'), round(span.get('font_size', 10.0), 1), is_bold)
                style_counts[key] += len(span['text'])

            if not style_counts: return {}
            bold_styles = {s: c for s, c in style_counts.items() if s[2]}
            regular_styles = {s: c for s, c in style_counts.items() if not s[2]}

            body_style = max(regular_styles, key=regular_styles.get) if regular_styles else min(bold_styles, key=lambda s: s[1]) if bold_styles else list(style_counts.keys())[0]

            style_to_level = {body_style: 'Body'}
            for style in regular_styles:
                if style != body_style: style_to_level[style] = 'Body'

            heading_candidates = [s for s in bold_styles if s[1] >= body_style[1] and s != body_style]
            larger_candidates = [s for s in heading_candidates if s[1] > body_style[1]]
            same_size_candidates = [s for s in heading_candidates if s[1] == body_style[1]]

            for style in same_size_candidates:
                style_to_level[style] = 'H3'

            if larger_candidates:
                font_sizes = np.array([s[1] for s in larger_candidates]).reshape(-1, 1)
                k = min(self.n_clusters - 1, len(np.unique(font_sizes)))
                if k > 0:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(font_sizes)
                    sorted_centers_indices = np.argsort(kmeans.cluster_centers_.flatten())[::-1]
                    level_map = {idx: f'H{i+1}' for i, idx in enumerate(sorted_centers_indices)}
                    for i, style in enumerate(larger_candidates):
                        cluster_label = kmeans.labels_[i]
                        style_to_level[style] = level_map[cluster_label]

            final_labels = {}
            for (name, size, _), level in style_to_level.items():
                final_labels[(name, round(size, 1))] = level
            
            self.logger.info(f"Final font clusters identified: {list(set(final_labels.values()))}")
            return final_labels
        except Exception as e:
            self.logger.error(f"Error clustering fonts: {e}")
            return {}