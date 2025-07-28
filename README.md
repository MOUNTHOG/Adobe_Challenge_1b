# PDF Processing Pipeline

## Overview

This repository contains a comprehensive PDF processing pipeline that extracts headings from PDFs, analyzes their content considering persona and job context, and generates rich, structured JSON output. The pipeline comprises two core scripts (`main.py` and `all.py`) and an orchestrator (`final.py`) to run them sequentially. The solution is fully containerized with Docker for reliable, platform-compatible execution on AMD64 Linux.

---

## Key Features and Highlights

- **Multi-PDF Collection Support:** Processes multiple collections of PDFs organized in directories.
- **Heading Extraction:** Extracts and combines headings from PDFs into consolidated JSON files.
- **Context-Aware Analysis:** Leverages persona and job-to-be-done information from input JSON configurations to perform semantic relevance scoring.
- **Enhanced Content Refinement:** Extracts refined textual content corresponding to each heading for in-depth analysis.
- **Parallel Processing:** Uses multithreading for efficient PDF processing.
- **Offline Capability:** Fully self-contained Docker image; no internet connection required at runtime.
- **AMD64 Compatibility:** Explicitly built for amd64 (x86_64) Linux architectures with no GPU dependencies.
- **Clean and Extensible Codebase:** Modular structure with clear separation between extraction and analysis components.

---

## Repository Structure

```
./
├── collections/ # Input directory containing PDF collections
│ ├── collection 1/
│ │ ├── input.json # Config with persona, job, and documents
| | ├── Collection 1_analysis.json # Expected output for collection
│ │ └── pdfs/
│ │ ├── file1.pdf
│ │ └── file2.pdf
│ ├── collection2/
│ │ └── ...
│ └── ...
├── headings/ # Output directory for combined headings 
│ ├── collection1_combined_headings.json 
│ ├── collection2_combined_headings.json 
│ └── ...
├── src/ 
│ ├── pdf_classifier.py
│ ├── ocr_processor.py
│ └── ... (other modules)
├── main.py
├── all.py 
├── final.py 
├── requirements.txt 
├── Dockerfile 
└── README.md 
```

---

## Solution Approach (Stepwise)

1. **Preparation:**
   - Organize PDFs into collection directories under `collections/`.
   - Place a configuration JSON (`input.json`) per collection specifying persona, job-to-be-done, and list of PDFs.

2. **Heading Extraction (`main.py`):**
   - Iterates over collections and their PDFs.
   - Extracts title and headings from each PDF using a detailed text and font-clustering pipeline.
   - Combines all headings per collection into a single JSON file with heading metadata.

3. **Contextual Analysis (`all.py`):**
   - Reads combined heading JSON files from the `headings/` directory.
   - Loads persona and job info from each collection's `input.json`.
   - Extracts and refines content linked to headings.
   - Scores and ranks sections based on semantic relevance to persona and job.
   - Produces detailed analysis JSON output saved back into the collection directory.

4. **Orchestration (`final.py`):**
   - Runs `main.py`, then `all.py` sequentially.
   - Handles logging, error detection, and execution timing.
   - Ensures smooth end-to-end pipeline execution.

---

## Build and Run Guide (Using Docker on Linux/AMD64)

### Build Docker Image

```
docker build --platform=linux/amd64 -t pdf-processor .
```

### Run the Pipeline

Run container with mounted volumes for input/output:

```
docker run --platform=linux/amd64
-v $(pwd)/collections:/app/collections:
-v $(pwd)/headings:/app/headings
pdf-processor
```
