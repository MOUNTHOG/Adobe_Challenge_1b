# enhanced_heading_extractor.py - Modified for renamed directories and structure

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Set, Tuple
import numpy as np
import fitz # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

class EnhancedHeadingExtractor:
    def __init__(self, base_input_dir: str = "./collections", base_output_dir: str = "./headings"):
        self.vectorizer = TfidfVectorizer(
            max_features=2000, # Increased for better vocabulary coverage
            stop_words='english',
            ngram_range=(1, 4), # Include 4-grams for better phrase matching
            min_df=1,
            max_df=0.85,
            token_pattern=r'\b[a-zA-Z]{2,}\b' # Better tokenization
        )
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir
        print("Enhanced PDF extractor initialized")
        print(f"   Input base directory (collections): {base_input_dir}")
        print(f"   Headings directory: {base_output_dir}")

    # Rest of your class methods remain exactly the same...
    # (keeping all the existing methods unchanged)


    def load_extracted_headings(self, pdf_files: List[str], collection_name: str) -> Dict[str, List[Dict]]:
        """Load outline headings from combined JSON file in headings directory"""
        headings_per_doc = {}
        
        # Look for combined file in the headings directory
        combined_json = os.path.join(self.base_output_dir, f"{collection_name}_combined_headings.json")

        if os.path.exists(combined_json):
            print(f"📖 Loading COMBINED headings from: {combined_json}")
            try:
                with open(combined_json, 'r', encoding='utf-8') as f:
                    combined_data = json.load(f)
            except Exception as e:
                print(f"❌ Failed to read combined file: {e}")
                return {}

            if combined_data and "outline" in combined_data:
                # Create PDF name mapping
                pdf_map = {
                    os.path.basename(p): p
                    for p in pdf_files
                }

                # Group outline entries by pdf_name
                grouped: Dict[str, List[Dict]] = {}
                for h in combined_data["outline"]:
                    pdf_name = h.get("pdf_name")
                    if not pdf_name:
                        continue
                    grouped.setdefault(pdf_name, []).append(h)

                # Parse each PDF's headings
                for pdf_name, outlines in grouped.items():
                    pdf_path = pdf_map.get(pdf_name)
                    if not pdf_path:
                        print(f"⚠️ Outline refers to unknown PDF: {pdf_name}")
                        continue

                    parsed = self._parse_headings_enhanced(
                        {"outline": outlines},
                        pdf_path,
                        pdf_name
                    )
                    headings_per_doc[pdf_name] = parsed
                    print(f"✅ Loaded {len(parsed)} headings from {pdf_name}")

                return headings_per_doc
        
        # If combined file not found
        print(f"❌ Combined headings file not found: {combined_json}")
        print("📁 Available files in headings directory:")
        if os.path.exists(self.base_output_dir):
            for file in os.listdir(self.base_output_dir):
                if file.endswith('.json'):
                    print(f"   - {file}")
        
        return {}

    def _parse_headings_enhanced(self, heading_data: Dict, pdf_path: str, pdf_name: str) -> List[Dict]:
        """Enhanced heading parsing with better content extraction"""
        sections = []
        
        if 'outline' in heading_data:
            for i, heading_info in enumerate(heading_data['outline']):
                try:
                    title = heading_info.get('text', '').strip()
                    page = heading_info.get('page', 1)
                    level = heading_info.get('level', 'H3')
                    
                    if not title or len(title) < 3:
                        continue
                        
                    print(f"📋 Processing heading {i+1}: '{title}' (page {page})")
                    
                    # Enhanced content extraction with multiple strategies
                    actual_content = self._extract_enhanced_pdf_content(pdf_path, title, page, i)
                    
                    if actual_content and len(actual_content.strip()) > 20:
                        sections.append({
                            'title': title,
                            'content': actual_content,
                            'page': page,
                            'document': pdf_name,
                            'heading_level': level,
                            'confidence': self._calculate_content_confidence(actual_content),
                            'is_heading_only': False,
                            'heading_index': i
                        })
                        print(f"✅ Content extracted: {len(actual_content)} chars")
                    else:
                        print(f"⚠️ No substantial content extracted for '{title}'")
                        
                except Exception as e:
                    print(f"❌ Error parsing heading '{heading_info}': {e}")
                    continue
        
        print(f"📊 Successfully parsed {len(sections)} headings with content from {pdf_name}")
        return sections

    def _extract_enhanced_pdf_content(self, pdf_path: str, heading: str, target_page: int, heading_index: int) -> str:
        """Enhanced PDF content extraction with multiple improved strategies"""
        try:
            if not os.path.exists(pdf_path):
                return ""
                
            with fitz.open(pdf_path) as pdf_doc:
                if target_page > len(pdf_doc):
                    target_page = min(target_page, len(pdf_doc))
                
                # Strategy 1: Enhanced heading-based extraction
                for page_offset in [0, 1, -1, 2]: # Try current, next, previous, and next+1 pages
                    page_idx = target_page - 1 + page_offset
                    if 0 <= page_idx < len(pdf_doc):
                        content = self._extract_content_from_page(pdf_doc[page_idx], heading, page_offset == 0)
                        if content and len(content.strip()) > 100:
                            return content
                
                # Strategy 2: Multi-page content extraction for long sections
                if target_page < len(pdf_doc):
                    multi_page_content = self._extract_multi_page_content(pdf_doc, target_page, heading)
                    if multi_page_content and len(multi_page_content.strip()) > 100:
                        return multi_page_content
                
                # Strategy 3: Context-aware extraction using surrounding headings
                context_content = self._extract_contextual_content(pdf_doc, target_page, heading)
                if context_content and len(context_content.strip()) > 50:
                    return context_content
                
                return ""
                
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""

    def _extract_content_from_page(self, page, heading: str, is_target_page: bool) -> str:
        """Extract content from a specific page with enhanced strategies"""
        text = page.get_text()
        if not text.strip():
            return ""
        
        # Strategy 1: Look for exact heading match and extract following content
        content = self._find_content_after_exact_heading(text, heading)
        if content:
            return content
        
        # Strategy 2: Fuzzy heading match (for slight variations)
        content = self._find_content_after_fuzzy_heading(text, heading)
        if content:
            return content
        
        # Strategy 3: Keyword-based extraction with improved scoring
        content = self._extract_by_enhanced_keywords(text, heading)
        if content:
            return content
        
        # Strategy 4: If target page, get substantial content anyway
        if is_target_page:
            content = self._extract_substantial_content(text)
            if content:
                return content
        
        return ""

    def _find_content_after_exact_heading(self, text: str, heading: str) -> str:
        """Find content after exact heading match"""
        lines = text.split('\n')
        content_lines = []
        found_heading = False
        heading_clean = heading.lower().strip()
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check for exact heading match
            if heading_clean in line_clean.lower():
                found_heading = True
                continue
            
            # Collect content after heading
            if found_heading:
                # Stop if we hit another heading (heuristics)
                if (len(line_clean) < 150 and 
                    line_clean.isupper() and 
                    len(line_clean.split()) <= 10):
                    break
                
                if (len(line_clean) < 100 and 
                    not line_clean.endswith('.') and 
                    line_clean.istitle() and 
                    len(content_lines) > 0):
                    break
                
                if len(line_clean) > 30: # Substantial content
                    content_lines.append(line_clean)
                
                if len(' '.join(content_lines)) > 800: # Enough content
                    break
        
        if content_lines:
            return self._clean_and_limit_content(' '.join(content_lines))
        
        return ""

    def _find_content_after_fuzzy_heading(self, text: str, heading: str) -> str:
        """Find content with fuzzy heading matching"""
        lines = text.split('\n')
        heading_words = set(word.lower() for word in heading.split() if len(word) > 2)
        
        best_match_line = -1
        best_match_score = 0
        
        # Find the line that best matches the heading
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if len(line_clean) < 5:
                continue
            
            line_words = set(word.lower() for word in line_clean.split() if len(word) > 2)
            if heading_words and line_words:
                overlap = len(heading_words.intersection(line_words))
                similarity = overlap / len(heading_words)
                
                if similarity > best_match_score and similarity >= 0.6:
                    best_match_score = similarity
                    best_match_line = i
        
        # Extract content after the best matching line
        if best_match_line >= 0:
            content_lines = []
            for i in range(best_match_line + 1, len(lines)):
                line_clean = lines[i].strip()
                if not line_clean:
                    continue
                
                # Stop conditions (another heading)
                if (len(line_clean) < 100 and 
                    line_clean.isupper() and 
                    len(content_lines) > 0):
                    break
                
                if len(line_clean) > 30:
                    content_lines.append(line_clean)
                
                if len(' '.join(content_lines)) > 600:
                    break
            
            if content_lines:
                return self._clean_and_limit_content(' '.join(content_lines))
        
        return ""

    def _extract_by_enhanced_keywords(self, text: str, heading: str) -> str:
        """Enhanced keyword-based extraction with better scoring"""
        heading_words = [word.lower() for word in heading.split() if len(word) > 3]
        if not heading_words:
            return ""
        
        # Split into paragraphs and score them
        paragraphs = re.split(r'\n\s*\n', text)
        scored_paragraphs = []
        
        for para in paragraphs:
            para_clean = para.strip()
            if len(para_clean) < 80: # Skip very short paragraphs
                continue
            
            para_lower = para_clean.lower()
            
            # Calculate score based on multiple factors
            score = 0
            
            # Keyword matching score
            word_matches = sum(1 for word in heading_words if word in para_lower)
            keyword_score = word_matches / len(heading_words) if heading_words else 0
            score += keyword_score * 3.0
            
            # Content quality score
            sentence_count = para_clean.count('.') + para_clean.count('!') + para_clean.count('?')
            quality_score = min(sentence_count / 5.0, 1.0) # Normalize to max 1.0
            score += quality_score
            
            # Length score (prefer substantial content)
            length_score = min(len(para_clean) / 400.0, 1.0) # Normalize to max 1.0
            score += length_score
            
            # Avoid header-like content
            if para_clean.isupper() or (len(para_clean) < 100 and para_clean.istitle()):
                score -= 2.0
            
            if score > 1.0: # Minimum threshold
                scored_paragraphs.append((score, para_clean))
        
        # Sort by score and select best paragraphs
        scored_paragraphs.sort(reverse=True)
        
        if scored_paragraphs:
            selected_content = []
            total_length = 0
            
            for score, para in scored_paragraphs:
                if total_length + len(para) > 1000: # Content limit
                    break
                selected_content.append(para)
                total_length += len(para)
                
                if len(selected_content) >= 3: # Max 3 paragraphs
                    break
            
            if selected_content:
                return self._clean_and_limit_content(' '.join(selected_content))
        
        return ""

    def _extract_multi_page_content(self, pdf_doc, start_page: int, heading: str) -> str:
        """Extract content spanning multiple pages"""
        content_parts = []
        heading_words = set(word.lower() for word in heading.split() if len(word) > 2)
        
        # Try up to 3 pages starting from the heading page
        for page_offset in range(3):
            page_idx = start_page - 1 + page_offset
            if page_idx >= len(pdf_doc):
                break
            
            page = pdf_doc[page_idx]
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Look for content related to the heading
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                para_clean = para.strip()
                if len(para_clean) < 100:
                    continue
                
                para_words = set(word.lower() for word in para_clean.split() if len(word) > 2)
                
                # Check relevance to heading
                if heading_words and para_words:
                    overlap = len(heading_words.intersection(para_words))
                    relevance = overlap / len(heading_words) if heading_words else 0
                    
                    if relevance >= 0.3: # 30% word overlap
                        content_parts.append(para_clean)
                        
                        if len(' '.join(content_parts)) > 800:
                            break
            
            if len(' '.join(content_parts)) > 600: # Enough content collected
                break
        
        if content_parts:
            return self._clean_and_limit_content(' '.join(content_parts))
        
        return ""

    def _extract_contextual_content(self, pdf_doc, target_page: int, heading: str) -> str:
        """Extract content using context from surrounding pages"""
        if target_page <= 0 or target_page > len(pdf_doc):
            return ""
        
        try:
            page = pdf_doc[target_page - 1]
            text = page.get_text()
            
            if not text.strip():
                return ""
            
            # Get the most substantial and relevant content from the page
            content = self._extract_substantial_content(text)
            return content
            
        except Exception:
            return ""

    def _extract_substantial_content(self, text: str) -> str:
        """Extract the most substantial content from text"""
        # Split into sentences and paragraphs
        sentences = re.split(r'[.!?]+', text)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Score paragraphs for quality
        good_paragraphs = []
        for para in paragraphs:
            para_clean = para.strip()
            if len(para_clean) < 100:
                continue
            
            # Quality indicators
            sentence_count = para_clean.count('.') + para_clean.count('!') + para_clean.count('?')
            
            # Skip headers and titles
            if (para_clean.isupper() or 
                (len(para_clean) < 200 and para_clean.istitle()) or 
                sentence_count < 2):
                continue
            
            # Calculate quality score
            quality_score = (
                min(len(para_clean) / 300.0, 2.0) + # Length score
                min(sentence_count / 3.0, 2.0) + # Sentence count score
                (1.0 if not para_clean.isupper() else 0.0) # Not all caps
            )
            
            good_paragraphs.append((quality_score, para_clean))
        
        # Sort by quality and select best
        good_paragraphs.sort(reverse=True)
        
        if good_paragraphs:
            selected_content = []
            total_length = 0
            
            for score, para in good_paragraphs:
                if total_length + len(para) > 800:
                    break
                selected_content.append(para)
                total_length += len(para)
                
                if len(selected_content) >= 2: # Max 2 paragraphs
                    break
            
            if selected_content:
                return self._clean_and_limit_content(' '.join(selected_content))
        
        # Fallback: get best sentences
        good_sentences = []
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) > 50 and not sentence_clean.isupper():
                good_sentences.append(sentence_clean)
                
                if len(good_sentences) >= 5:
                    break
        
        if good_sentences:
            content = '. '.join(good_sentences) + '.'
            return self._clean_and_limit_content(content)
        
        return ""

    def _clean_and_limit_content(self, content: str) -> str:
        """Clean and limit content length"""
        # Clean up spacing and formatting
        content = ' '.join(content.split())
        content = re.sub(r'\s+', ' ', content)
        
        # Ensure proper sentence spacing
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        
        # Limit length while preserving sentence boundaries
        if len(content) > 800:
            # Find a good break point near the limit
            break_point = content.rfind('.', 0, 800)
            if break_point > 400: # Make sure we have substantial content
                content = content[:break_point + 1]
            else:
                content = content[:800] + "..."
        
        return content.strip()

    def _calculate_content_confidence(self, content: str) -> float:
        """Calculate confidence score for extracted content"""
        if not content or len(content.strip()) < 20:
            return 0.1
        
        score = 0.5 # Base score
        
        # Length bonus
        length_bonus = min(len(content) / 500.0, 0.3)
        score += length_bonus
        
        # Sentence structure bonus
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        if sentence_count >= 2:
            score += 0.2
        
        # Quality indicators
        if not content.isupper(): # Not all caps
            score += 0.1
        
        if len(content.split()) >= 20: # Substantial word count
            score += 0.1
        
        return min(score, 1.0)

    def _extract_enhanced_terms(self, text: str) -> Set[str]:
        """Extract terms with better NLP techniques"""
        terms = set()
        
        # Extract individual words (3+ chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        terms.update(words)
        
        # Extract meaningful phrases
        # 2-grams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            terms.add(bigram)
        
        # 3-grams for important concepts
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            terms.add(trigram)
        
        # Extract numbers with context
        number_contexts = re.findall(r'(\d+)\s+(\w+)', text.lower())
        for num, context in number_contexts:
            terms.add(f"{num} {context}")
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', text)
        for phrase in quoted_phrases:
            if len(phrase.strip()) > 3:
                terms.add(phrase.lower().strip())
        
        return terms

    def _calculate_enhanced_relevance_score(self, section: Dict, persona_terms: Set[str], 
                                         job_terms: Set[str], context_terms: Set[str]) -> float:
        """Enhanced relevance scoring with better term weighting"""
        score = 0.0
        title_lower = section['title'].lower()
        content_lower = section['content'].lower()
        
        # Title scoring (higher weight)
        title_score = 0.0
        for term in persona_terms:
            if term in title_lower:
                title_score += 0.6
        
        for term in job_terms:
            if term in title_lower:
                title_score += 0.8
        
        for term in context_terms:
            if term in title_lower:
                title_score += 0.4
        
        score += title_score
        
        # Content scoring (lower weight but more comprehensive)
        content_score = 0.0
        for term in persona_terms:
            if term in content_lower:
                content_score += 0.2
        
        for term in job_terms:
            if term in content_lower:
                content_score += 0.3
        
        for term in context_terms:
            if term in content_lower:
                content_score += 0.1
        
        score += content_score
        
        # Content quality bonus
        content_length = len(section['content'])
        if content_length > 200:
            quality_bonus = min(content_length / 1000.0, 0.5)
            score += quality_bonus
        
        # Confidence bonus
        confidence_bonus = section.get('confidence', 0.5) * 0.3
        score += confidence_bonus
        
        # Heading level bonus
        level_bonus = {'H1': 0.3, 'H2': 0.2, 'H3': 0.1}.get(section.get('heading_level', 'H3'), 0.0)
        score += level_bonus
        
        return score

    def rank_sections_enhanced(self, all_sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Enhanced ranking with better term extraction and scoring"""
        if not all_sections:
            return []
        
        print(f"🔍 Enhanced analysis: '{persona}' + '{job}'")
        
        # Remove duplicates with better logic
        unique_sections = self._remove_enhanced_duplicates(all_sections)
        print(f"📊 Removed {len(all_sections) - len(unique_sections)} duplicate sections")
        
        # Enhanced term extraction
        persona_terms = self._extract_enhanced_terms(persona)
        job_terms = self._extract_enhanced_terms(job)
        
        # Extract context terms from job
        context_terms = self._extract_context_terms(job)
        
        print(f"📝 Persona terms: {sorted(list(persona_terms))[:8]}")
        print(f"🎯 Job terms: {sorted(list(job_terms))[:8]}")
        print(f"🔗 Context terms: {sorted(list(context_terms))[:8]}")
        
        # Prepare for vectorization
        all_terms = list(persona_terms.union(job_terms).union(context_terms))
        query = ' '.join(all_terms)
        
        section_texts = []
        for section in unique_sections:
            # Enhanced text representation
            text = f"{section['title']} {section['title']} {section['content']}"
            section_texts.append(text)
        
        try:
            # Enhanced vectorization and similarity calculation
            all_texts = [query] + section_texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            query_vector = tfidf_matrix[0:1]
            section_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, section_vectors)[0]
            
            # Calculate enhanced scores
            for i, section in enumerate(unique_sections):
                semantic_similarity = similarities[i]
                content_confidence = section.get('confidence', 0.5)
                enhanced_relevance = self._calculate_enhanced_relevance_score(
                    section, persona_terms, job_terms, context_terms
                )
                
                # Weighted combination with enhanced scoring
                final_score = (
                    semantic_similarity * 0.35 +
                    content_confidence * 0.15 +
                    enhanced_relevance * 0.5
                )
                
                section['similarity_score'] = float(final_score)
                section['semantic_score'] = float(semantic_similarity)
                section['enhanced_relevance'] = float(enhanced_relevance)
                section['content_confidence'] = float(content_confidence)
                
        except Exception as e:
            print(f"⚠️ Vectorization error, using fallback scoring: {e}")
            # Enhanced fallback scoring
            for section in unique_sections:
                confidence = section.get('confidence', 0.5)
                enhanced_relevance = self._calculate_enhanced_relevance_score(
                    section, persona_terms, job_terms, context_terms
                )
                
                section['similarity_score'] = (confidence * 0.3 + enhanced_relevance * 0.7)
        
        # Sort and rank
        unique_sections.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        for i, section in enumerate(unique_sections):
            section['importance_rank'] = i + 1
        
        return unique_sections

    def _remove_enhanced_duplicates(self, sections: List[Dict]) -> List[Dict]:
        """Enhanced duplicate removal with better similarity detection"""
        unique_sections = []
        
        for section in sections:
            is_duplicate = False
            current_title = section['title'].lower().strip()
            current_content = section['content'][:100].lower() # First 100 chars for comparison
            
            for existing in unique_sections:
                existing_title = existing['title'].lower().strip()
                existing_content = existing['content'][:100].lower()
                
                # Title similarity check
                title_similarity = self._calculate_text_similarity(current_title, existing_title)
                
                # Content similarity check
                content_similarity = self._calculate_text_similarity(current_content, existing_content)
                
                # Consider duplicate if titles are very similar OR content is very similar
                if title_similarity > 0.85 or content_similarity > 0.9:
                    # Keep the one with higher confidence
                    if section.get('confidence', 0) > existing.get('confidence', 0):
                        unique_sections.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_sections.append(section)
        
        return unique_sections

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _extract_context_terms(self, job: str) -> Set[str]:
        """Extract context-specific terms from job description"""
        context_terms = set()
        job_lower = job.lower()
        
        # Extract action words
        action_words = re.findall(r'\b(?:plan|organize|prepare|create|develop|analyze|review|study|find|search|get|make|build|design|learn|understand|explore|discover|identify)\b', job_lower)
        context_terms.update(action_words)
        
        # Extract duration/time expressions
        time_expressions = re.findall(r'\b(?:\d+[-\s]?(?:day|week|month|hour|minute)s?|weekend|vacation|trip|journey)\b', job_lower)
        context_terms.update(time_expressions)
        
        # Extract group/social context
        social_context = re.findall(r'\b(?:friends|family|colleagues|students|team|group|party|couple|solo|alone)\b', job_lower)
        context_terms.update(social_context)
        
        return context_terms

    def load_input_json(self, collection_path: str) -> Dict:
        """Load input.json from collection directory"""
        input_json_path = os.path.join(collection_path, 'input.json')
        
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"input.json not found in {collection_path}")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_collection(self, collection_name: str) -> Dict:
        """Process a single collection with its input.json configuration"""
        start_time = time.time()
        
        # Construct paths for this collection
        collection_path = os.path.join(self.base_input_dir, collection_name)
        pdfs_dir = os.path.join(collection_path, "pdfs")
        
        print(f"\n🗂️ Processing collection: {collection_name}")
        print(f"   Collection path: {collection_path}")
        print(f"   PDFs directory: {pdfs_dir}")
        
        # Load input.json configuration
        try:
            input_config = self.load_input_json(collection_path)
            persona = input_config.get("persona", {}).get("role", "User")
            job_to_be_done = input_config.get("job_to_be_done", {}).get("task", "Extract information")
            documents = input_config.get("documents", [])
            
            print(f"👤 Persona: {persona}")
            print(f"🎯 Job: {job_to_be_done}")
            print(f"📄 Documents specified: {len(documents)}")
            
        except Exception as e:
            print(f"❌ Error loading input.json: {e}")
            return {
                'metadata': {
                    'error': f'input_json_load_error: {str(e)}',
                    'collection': collection_name,
                    'persona': 'Unknown',
                    'job_to_be_done': 'Unknown'
                },
                'extracted_sections': [],
                'subsection_analysis': []
            }
        
        # Get PDF files from pdfs directory
        if not os.path.exists(pdfs_dir):
            print(f"❌ PDFs directory not found: {pdfs_dir}")
            return {
                'metadata': {
                    'error': 'pdfs_directory_not_found',
                    'collection': collection_name,
                    'persona': persona,
                    'job_to_be_done': job_to_be_done
                },
                'extracted_sections': [],
                'subsection_analysis': []
            }
        
        # Get PDFs based on documents list in input.json
        pdf_files = []
        for doc in documents:
            pdf_filename = doc.get("filename", "")
            if pdf_filename.endswith('.pdf'):
                pdf_path = os.path.join(pdfs_dir, pdf_filename)
                if os.path.exists(pdf_path):
                    pdf_files.append(pdf_path)
                else:
                    print(f"⚠️ PDF file not found: {pdf_filename}")
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {pdfs_dir}")
            return {
                'metadata': {
                    'error': 'no_pdf_files_found',
                    'collection': collection_name,
                    'persona': persona,
                    'job_to_be_done': job_to_be_done
                },
                'extracted_sections': [],
                'subsection_analysis': []
            }
        
        print(f"🚀 Processing {len(pdf_files)} PDF files from {collection_name}...")
        
        # Load headings from headings directory using collection name
        headings_per_doc = self.load_extracted_headings(pdf_files, collection_name)
        
        # Combine all sections
        all_sections = []
        for doc_name, sections in headings_per_doc.items():
            all_sections.extend(sections)
        
        print(f"📊 Total sections extracted: {len(all_sections)}")
        
        if not all_sections:
            print("⚠️ No sections found with substantial content")
            return {
                'metadata': {
                    'error': 'no_substantial_content_found',
                    'collection': collection_name,
                    'input_documents': [os.path.basename(f) for f in pdf_files],
                    'persona': persona,
                    'job_to_be_done': job_to_be_done
                },
                'extracted_sections': [],
                'subsection_analysis': []
            }
        
        # Enhanced ranking
        ranked_sections = self.rank_sections_enhanced(all_sections, persona, job_to_be_done)
        
        # Create enhanced subsection analysis
        unique_refined_texts = []
        seen_content_hashes = set()
        
        for section in ranked_sections[:15]: # Consider more sections initially
            content = section['content']
            title = section['title']
            
            # Create content hash for better duplicate detection
            content_hash = hash(content[:200]) # Hash first 200 chars
            
            # Quality checks
            if (content_hash not in seen_content_hashes and 
                len(content.strip()) > 80 and # Minimum length
                not content.startswith(("No substantial", "Content related", "Error")) and
                section.get('confidence', 0) > 0.3): # Minimum confidence
                
                seen_content_hashes.add(content_hash)
                unique_refined_texts.append({
                    'document': section['document'],
                    'refined_text': content,
                    'page_number': section['page']
                })
                
                if len(unique_refined_texts) >= 5: # Limit to top 5
                    break
        
        # Generate enhanced result - UPDATED FORMAT
        result = {
            'metadata': {
                'input_documents': [os.path.basename(f) for f in pdf_files],
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': [
                {
                    'document': section['document'],
                    'section_title': section['title'],
                    'importance_rank': section['importance_rank'],
                    'page_number': section['page']
                }
                for section in ranked_sections[:5]
            ],
            'subsection_analysis': unique_refined_texts
        }
        
        processing_time = time.time() - start_time
        print(f"✅ Enhanced processing completed in {processing_time:.2f} seconds")
        
        # Enhanced summary
        print(f"🎯 Top sections with enhanced scoring:")
        for i, section in enumerate(ranked_sections[:5]):
            print(f"   {i+1}. {section['title']}")
            # print(f"      Relevance: {section.get('similarity_score', 0):.3f} | Confidence: {section.get('confidence', 0):.3f}")
        
        successful_extractions = len([s for s in ranked_sections if s.get('confidence', 0) > 0.5])
        print(f"📈 High-confidence extractions: {successful_extractions}/{len(ranked_sections)}")
        
        return result


    def process_all_collections(self) -> Dict[str, Dict]:
        """Process all collections found in the collections directory"""
        print(f"🏁 Starting batch processing of collections...")
        
        # Find all collection directories (those containing input.json)
        collection_dirs = []
        if not os.path.exists(self.base_input_dir):
            print(f"❌ Collections directory not found: {self.base_input_dir}")
            return {}
        
        for item in os.listdir(self.base_input_dir):
            item_path = os.path.join(self.base_input_dir, item)
            if os.path.isdir(item_path):
                input_json_path = os.path.join(item_path, 'input.json')
                if os.path.exists(input_json_path):
                    collection_dirs.append(item)
        
        if not collection_dirs:
            print(f"❌ No collections found with input.json files")
            return {}
        
        print(f"📁 Found {len(collection_dirs)} collections: {collection_dirs}")
        
        all_results = {}
        
        for collection_name in collection_dirs:
            print(f"\n{'='*60}")
            print(f"📁 Processing collection: {collection_name}")
            print(f"{'='*60}")
            
            try:
                result = self.process_collection(collection_name)
                all_results[collection_name] = result
                
                # Save individual result in the collection's directory (NOT headings directory)
                collection_output_dir = os.path.join(self.base_input_dir, collection_name)
                
                output_filename = os.path.join(collection_output_dir, f'{collection_name}_analysis.json')
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Saved result to: {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing {collection_name}: {e}")
                all_results[collection_name] = {
                    'metadata': {
                        'error': str(e),
                        'collection': collection_name,
                        'persona': 'Unknown',
                        'job_to_be_done': 'Unknown'
                    },
                    'extracted_sections': [],
                    'subsection_analysis': []
                }
        
        
def main():
    import sys
    
    # Parse command line arguments for collections and headings directories
    collections_dir = "./collections"
    headings_dir = "./headings"
    
    if len(sys.argv) >= 2:
        collections_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        headings_dir = sys.argv[2]
    
    print(f"🚀 Enhanced Heading Extractor")
    print(f"   Collections directory: {collections_dir}")
    print(f"   Headings directory: {headings_dir}")
    
    try:
        # Initialize extractor
        extractor = EnhancedHeadingExtractor(collections_dir, headings_dir)
        
        # Process all collections
        results = extractor.process_all_collections()
        
        if results:
            print("🎯 Enhanced extraction completed successfully!")
        else:
            print(" ")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
