
import re
import unicodedata

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = unicodedata.normalize('NFKC', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    text = text.strip()
    
    return text

def is_likely_header_footer(text: str) -> bool:
    
    text_lower = text.lower().strip()
    
    patterns = [
        r'^\d+$',  
        r'^page \d+',  
        r'^\d+ of \d+$',  
        r'^chapter \d+',  
        r'^section \d+',  
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return True
    
    if len(text.strip()) < 3:
        return True
        
    return False

def normalize_font_name(font_name: str) -> str:
    if not font_name:
        return "Unknown"
    
    normalized = re.sub(r'[^a-zA-Z0-9]', '', font_name.lower())
    
    mappings = {
        'timesnewroman': 'times',
        'timesroman': 'times',
        'arial': 'arial',
        'helvetica': 'helvetica',
        'calibri': 'calibri',
    }
    
    for pattern, replacement in mappings.items():
        if pattern in normalized:
            return replacement
    
    return normalized

def extract_numbering(text: str) -> tuple:
    patterns = [
        (r'^(\d+\.?\d*\.?\d*)\s+', lambda m: (m.group(1), m.group(1).count('.') + 1)),
        (r'^([A-Z]\.?\d*\.?)\s+', lambda m: (m.group(1), m.group(1).count('.') + 1)),
        (r'^([IVX]+\.)\s+', lambda m: (m.group(1), 1)),
        (r'^(\d+)\)\s+', lambda m: (m.group(1) + ')', 1)),
        (r'^([a-z])\)\s+', lambda m: (m.group(1) + ')', 2)),
    ]
    
    for pattern, extractor in patterns:
        match = re.match(pattern, text)
        if match:
            return extractor(match)
    
    return None, 0