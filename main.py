import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.pdf_classifier import PDFClassifier
from src.ocr_processor import OCRProcessor
from src.text_extractor import TextExtractor
from src.title_extractor import TitleExtractor
from src.summary_generator import SummaryGenerator
from src.font_clusterer import FontClusterer
from src.heading_detector import HeadingDetector
from src.semantic_validator import SemanticValidator
from src.heading_filter import HeadingFilter
from src.hierarchy_builder import HierarchyBuilder
from src.knowledge_graph_generator import KnowledgeGraphGenerator
from src.output_generator import OutputGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_single_pdf(pdf_path: Path) -> Dict[str, Any]:
    pdf_classifier = PDFClassifier()
    ocr_processor = OCRProcessor()
    text_extractor = TextExtractor()
    title_extractor = TitleExtractor()
    summary_generator = SummaryGenerator()
    font_clusterer = FontClusterer()
    heading_detector = HeadingDetector()
    semantic_validator = SemanticValidator()
    heading_filter = HeadingFilter()
    hierarchy_builder = HierarchyBuilder()
    kg_generator = KnowledgeGraphGenerator()
    output_generator = OutputGenerator()

    doc, classifications = pdf_classifier.classify_pages(pdf_path)
    _, _, detected_lang = text_extractor.extract_text_with_layout(doc, classifications, {})
    ocr_results = ocr_processor.process_scanned_pages(doc, classifications, lang=detected_lang)
    all_spans, full_text, _ = text_extractor.extract_text_with_layout(doc, classifications, ocr_results)

    first_page_spans = [s for s in all_spans if s.get('page_num') == 0]
    title_string, title_spans = title_extractor.extract_title_block(first_page_spans)

    title_span_ids = {id(span) for span in title_spans}
    non_title_spans = [span for span in all_spans if id(span) not in title_span_ids]

    summary = summary_generator.generate_summary(full_text)
    knowledge_graph = kg_generator.generate_knowledge_graph(full_text)
    
    font_clusters = font_clusterer.cluster_fonts(non_title_spans)
    heading_candidates = heading_detector.detect_candidates(non_title_spans, font_clusters)
    validated_headings = semantic_validator.validate_headings(heading_candidates, knowledge_graph)
    filtered_headings = heading_filter.filter_headings(validated_headings)
    structured_headings = hierarchy_builder.build_hierarchy(filtered_headings)
    cleaned_headings = hierarchy_builder.cleanup_headings(structured_headings)

    final_output = output_generator.generate_output(
        title=title_string,
        headings=cleaned_headings
    )
    
    doc.close()
    return final_output

def main():
    input_dir = Path("collections")  # Changed from "input" to "collections"
    output_dir = Path("headings")    # Keep as "headings" based on previous discussion
    output_dir.mkdir(exist_ok=True)
    
    # Find all collection directories (directories containing pdfs/ subdirectory)
    collection_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir() and (item / "pdfs").exists():
            collection_dirs.append(item)
    
    if not collection_dirs:
        logging.error("No collections found. Each collection should be a directory containing pdfs/ subdirectory")
        return
    
    logging.info(f"Found {len(collection_dirs)} collections to process")
    
    for collection_dir in collection_dirs:
        logging.info(f"Processing collection: {collection_dir.name}")
        
        # Get all PDF files from the pdfs directory
        pdfs_dir = collection_dir / "pdfs"
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {pdfs_dir}")
            continue
        
        logging.info(f"Found {len(pdf_files)} PDF files in collection {collection_dir.name}")
        
        # Process all PDFs in the collection and combine results
        all_headings = []
        pdf_titles = {}
        processed_count = 0
        
        with ThreadPoolExecutor() as executor:
            future_to_pdf = {executor.submit(process_single_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    
                    # Store individual PDF title
                    pdf_titles[pdf_path.name] = result.get("title", f"Title for {pdf_path.name}")
                    
                    # Extract headings and add PDF name to each heading
                    headings = result.get("outline", [])
                    for heading in headings:
                        heading_with_pdf = {
                            "level": heading.get("level", "H1"),
                            "text": heading.get("text", ""),
                            "page": heading.get("page", 1),
                            "pdf_name": pdf_path.name
                        }
                        all_headings.append(heading_with_pdf)
                    
                    logging.info(f"Successfully processed {pdf_path.name} - found {len(headings)} headings")
                    processed_count += 1
                    
                except Exception as e:
                    logging.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        
        # Create combined output for this collection
        combined_output = {
            "titles": pdf_titles,
            "outline": all_headings
        }
        
        # Save the combined output
        output_path = output_dir / f"{collection_dir.name}_combined_headings.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_output, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully processed collection {collection_dir.name} ({processed_count}/{len(pdf_files)} files) with {len(all_headings)} total headings and saved to {output_path}")

if __name__ == "__main__":
    main()
