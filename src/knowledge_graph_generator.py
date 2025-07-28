"""
This module is responsible for generating a knowledge graph by extracting
entities and their relationships from the document text.
"""
import logging
from typing import List, Dict, Any
import spacy

class KnowledgeGraphGenerator:
    """
    Extracts entities and infers relationships to build a knowledge graph.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("Spacy model 'en_core_web_sm' not found.")
            self.logger.info("Please run 'python -m spacy download en_core_web_sm' to install it.")
            self.nlp = None

    def generate_knowledge_graph(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        
        if not self.nlp or not text:
            return {"entities": [], "relationships": []}

        doc = self.nlp(text)

       
        entities = []
        for ent in doc.ents:
            
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "DATE"]:
                entities.append({"text": ent.text, "type": ent.label_})

       
        relationships = []
        for sent in doc.sents:
           
            subject = None
            obj = None
            relation = None

            for token in sent:
                if "subj" in token.dep_:
                    subject = token
                elif "obj" in token.dep_:
                    obj = token
                elif "ROOT" in token.dep_ and token.pos_ == "VERB":
                    relation = token
            
            if subject and obj and relation:
                
                subject_ent = self._find_entity_for_token(subject, doc.ents)
                object_ent = self._find_entity_for_token(obj, doc.ents)

                if subject_ent and object_ent:
                    relationships.append({
                        "subject": subject_ent.text,
                        "relation": relation.lemma_, 
                        "object": object_ent.text
                    })

        self.logger.info(f"Generated knowledge graph with {len(entities)} entities "
                       f"and {len(relationships)} relationships.")
        
        return {"entities": entities, "relationships": relationships}

    def _find_entity_for_token(self, token, entities):
        for ent in entities:
            if token.i >= ent.start and token.i < ent.end:
                return ent
        return None
