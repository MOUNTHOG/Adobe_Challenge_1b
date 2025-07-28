


import re
import logging
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class SummaryGenerator:
    

    def __init__(self, num_sentences: int = 5, min_sentence_len: int = 5):
    
        self.num_sentences = num_sentences
        self.min_sentence_len = min_sentence_len
        self.logger = logging.getLogger(__name__)

    def _split_sentences(self, text: str) -> List[str]:
       
       
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

        cleaned_sentences = []
        for s in sentences:
            s_clean = re.sub(r'\s+', ' ', s).strip()
           
            if len(s_clean.split()) >= self.min_sentence_len and not re.match(r'^(figure|table|fig|tab)\s*\d+', s_clean.lower()):
                cleaned_sentences.append(s_clean)
        return cleaned_sentences

    def generate_summary(self, text: str) -> str:
        
        if not text or len(text.strip()) < 100:
            self.logger.warning("Text is too short for summary generation.")
            return text

        try:
           
            sentences = self._split_sentences(text)
            if len(sentences) <= self.num_sentences:
                return ' '.join(sentences)

           
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)

           
            sim_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(sim_matrix, 0) 

            graph = nx.from_numpy_array(sim_matrix)

           
            scores = nx.pagerank(graph)

           
            ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

            
            top_sentences = sorted(ranked_sentences[:self.num_sentences], key=lambda x: x[2])
            summary = ' '.join(s[1] for s in top_sentences)

            self.logger.info(f"Generated TextRank summary from {len(sentences)} sentences.")
            return summary

        except Exception as e:
            self.logger.error(f"Error generating graph-based summary: {str(e)}")
            # Fallback to a simple summary if the graph algorithm fails.
            fallback_sentences = text.split('.')[:self.num_sentences]
            return '. '.join(s.strip() for s in fallback_sentences if s.strip()) + '.'
