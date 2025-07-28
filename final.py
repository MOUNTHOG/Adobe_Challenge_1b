#!/usr/bin/env python3
"""
final.py - Sequential execution script that runs main.py followed by all.py

This script orchestrates the complete PDF processing pipeline:
1. Runs main.py to extract headings from PDFs and create combined heading files
2. Runs all.py to analyze the headings and generate final analysis results
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_script(script_name, description):
    """
    Run a Python script and handle output/errors
    
    Args:
        script_name (str): Name of the Python script to run
        description (str): Description of what the script does
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting {description}")
    print(f"   Running: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print the output in real-time style
        if result.stdout:
            print("📄 Script Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ Script Warnings/Errors:")
            print(result.stderr)
        
        # Check if script completed successfully
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"✅ {description} completed successfully!")
            print(f"   Execution time: {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"❌ {description} failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print(f"❌ Error: {script_name} not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {script_name}: {e}")
        return False

def check_prerequisites():
    """
    Check if required files and directories exist before starting
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    print("🔍 Checking prerequisites...")
    
    required_files = ['main.py', 'all.py']
    required_dirs = ['collections', 'src']
    
    missing_items = []
    
    # Check for required Python scripts
    for file in required_files:
        if not os.path.exists(file):
            missing_items.append(f"File: {file}")
    
    # Check for required directories
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_items.append(f"Directory: {dir_name}")
    
    # Check if collections directory has any collections
    if os.path.exists('collections'):
        collections = [item for item in os.listdir('collections') 
                      if os.path.isdir(os.path.join('collections', item)) 
                      and os.path.exists(os.path.join('collections', item, 'pdfs'))]
        if not collections:
            missing_items.append("No collections found in 'collections' directory (each collection should have a 'pdfs' subdirectory)")
    
    if missing_items:
        print("❌ Prerequisites not met:")
        for item in missing_items:
            print(f"   - Missing: {item}")
        return False
    
    print("✅ All prerequisites met!")
    return True

def create_directories():
    """Create necessary output directories if they don't exist"""
    print("📁 Creating necessary directories...")
    
    directories = ['headings']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"   Created directory: {dir_name}")
        else:
            print(f"   Directory already exists: {dir_name}")

def display_results():
    """Display summary of generated files"""
    print(f"\n{'='*80}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    # Check headings directory
    headings_dir = Path('headings')
    if headings_dir.exists():
        heading_files = list(headings_dir.glob('*_combined_headings.json'))
        analysis_files = list(headings_dir.glob('*_analysis.json'))
        
        print(f"📁 Headings Directory ({headings_dir}):")
        print(f"   Combined heading files: {len(heading_files)}")
        for file in heading_files:
            print(f"     - {file.name}")
        
        print(f"   Analysis files: {len(analysis_files)}")
        for file in analysis_files:
            print(f"     - {file.name}")
    
    # Check collections directory for analysis results
    collections_dir = Path('collections')
    if collections_dir.exists():
        collection_analysis_files = []
        for collection_dir in collections_dir.iterdir():
            if collection_dir.is_dir():
                analysis_file = collection_dir / f'{collection_dir.name}_analysis.json'
                if analysis_file.exists():
                    collection_analysis_files.append(analysis_file)
        
        if collection_analysis_files:
            print(f"📁 Collections Directory Analysis Results:")
            for file in collection_analysis_files:
                print(f"     - {file}")
    
    print(f"\n✅ Processing pipeline completed successfully!")
    print(f"🎯 Check the files above for your results.")

def main():
    """Main execution function"""
    print("🎯 PDF Processing Pipeline - Final Orchestrator")
    print("=" * 80)
    print("This script will run the complete PDF processing pipeline:")
    print("1. main.py - Extract headings from PDFs and create combined files")
    print("2. all.py - Analyze headings and generate final results")
    print("=" * 80)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Cannot proceed due to missing prerequisites.")
        print("Please ensure all required files and directories are present.")
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    pipeline_start_time = time.time()
    
    # Step 1: Run main.py
    success_main = run_script(
        'main.py', 
        'PDF Heading Extraction (main.py)'
    )
    
    if not success_main:
        print("\n❌ Pipeline failed at Step 1 (main.py)")
        print("Cannot proceed to Step 2 without successful heading extraction.")
        sys.exit(1)
    
    # Brief pause between steps
    print("\n⏳ Preparing for analysis step...")
    time.sleep(2)
    
    # Step 2: Run all.py
    success_all = run_script(
        'all.py',
        'Heading Analysis and Final Processing (all.py)'
    )
    
    if not success_all:
        print("\n❌ Pipeline failed at Step 2 (all.py)")
        print("Heading extraction completed, but analysis failed.")
        sys.exit(1)
    
    # Calculate total pipeline time
    total_time = time.time() - pipeline_start_time
    
    # Display results summary
    display_results()
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Total execution time: {total_time:.2f} seconds")
    print(f"   Both main.py and all.py executed successfully.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Pipeline interrupted by user (Ctrl+C)")
        print("Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
