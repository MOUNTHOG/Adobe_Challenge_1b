# Challenge 1b : PDF Processing Solution

## Overview
This system, built for the Adobe India Hackathon, analyzes PDFs based on a given persona and their job to be done. It extracts structured headings, maps relevant sections, and ranks them using semantic and heuristic scoring.It solves the problem of intelligent content prioritization across large document collections.

The pipeline is orchestrated by `final.py` and consists of two main stages:
- **Heading Extraction (`main.py`)**: Scans PDFs to identify and extract structural elements such as titles and headings.
- **Content Analysis (`all.py`)**: Uses extracted headings to locate relevant content and rank it using heuristic and semantic methods.
The solution is fully containerized with Docker for reliable, platform-compatible execution on AMD64 Linux.

---

## Key Features and Highlights

- **Multi-PDF Collection Support:** Processes multiple collections of PDFs organized in directories.
- **Heading Extraction:** Extracts and combines headings from PDFs into consolidated JSON files.
- **Context-Aware Analysis:** Leverages persona and job-to-be-done information from input JSON configurations to perform semantic relevance scoring.
- **Enhanced Content Refinement:** Extracts refined textual content corresponding to each heading for in-depth analysis.
- **Parallel Processing:** Uses multithreading for efficient PDF headings extraction.
- **Offline Capability:** Fully self-contained Docker image; no internet connection required at runtime.
- **AMD64 Compatibility:** Explicitly built for amd64 (x86_64) Linux architectures with no GPU dependencies.
- **Clean and Extensible Codebase:** Modular structure with clear separation between extraction and analysis components.

---

## Repository Structure

```
.
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

The pipeline adopts a layered strategy designed for robustness, especially when working with unstructured or noisy PDFs.

### Key Features

- **Metadata-Driven Analysis**: Begins with structured metadata (from `_combined_headings.json`) for more reliable extraction.
- **Multi-Strategy Content Extraction**: Uses a fallback system combining exact matching, fuzzy text comparison, keyword proximity and general best scoring.
- **Hybrid Ranking System**:
  - Heuristic Scoring: Based on keyword density and structure.
  - Semantic Similarity: Uses TF-IDF and cosine similarity to score relevance.
- **Confidence Scoring**: Longer, better-structured paragraphs are scored higher.
![WhatsApp Image 2025-07-29 at 03 38 22_a2e14b53](https://github.com/user-attachments/assets/60b9ac5e-e787-49cb-9408-563a75703721)


---

### Description of Folders

- `collections/`: Root directory for PDF document collections.
- `pdfs/`: Contains the actual PDF files inside each collection.
- `input.json`: Specifies `persona`, `job_to_be_done`, and document list.
- `headings/`: Stores extracted heading metadata after running `main.py`.
- `src/`: Contains core logic and utilities used by the pipeline.

---

## How to Use

### Set Up Your Collection

Inside `collections/`, create a subfolder for each document set. Add:

- A folder called `pdfs/` containing your PDF files.
- An `input.json` file for configuration.

Example `input.json`:

```
json
{
  "persona": {
    "role": "adventure traveler"
  },
  "job_to_be_done": {
    "task": "Find information on hiking trails and national parks for a weekend trip"
  },
  "documents": [
    {"filename": "document1.pdf"},
    {"filename": "document2.pdf"}
  ]
}
```
---

## Build and Run Guide (Using Docker on Linux/AMD64)

### Build Docker Image

```
docker build --platform=linux/amd64 -t pdf-processor .
```

### Run the Pipeline

Run container with mounted volumes for input/output for linux/bash:

```
docker run --rm --name pdf-container -v "$(pwd)/collections:/app/collections" -v "$(pwd)/headings:/app/headings" pdf-processor
```

Run container with mounted volumes for input/output for powershell:

```
docker run --rm --name pdf-container -v ${PWD}\collections:/app/collections -v ${PWD}\headings:/app/headings pdf-processor
```

##Output Description 

### Analysis 

After running all.py, the final output is saved to:
```
collections/{collection_name}/{collection_name}_analysis.json
```
it incudes : 
```
json
{
  "metadata": {
    "persona": {...},
    "job_to_be_done": {...},
    "documents": [...]
  },
  "extracted_sections": [
    "Most Relevant Section Titles"
  ],
  "subsection_analysis": [
    {
      "section": "Section Title",
      "content": "Relevant paragraph or content",
      "score": 0.91
    }
  ]
}

```
