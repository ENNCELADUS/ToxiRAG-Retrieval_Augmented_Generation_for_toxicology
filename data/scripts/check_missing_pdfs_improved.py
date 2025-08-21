#!/usr/bin/env python3
"""
Improved script to check which PDF files in Downloads folder have titles not present in the markdown file.
This version uses PDF filenames instead of extracting titles from PDF content.
"""

import os
import re
from pathlib import Path

def extract_compound_names_from_md(md_file_path):
    """Extract all compound names from the markdown file."""
    compound_names = set()
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all compound names using regex
        # Look for patterns like "化合物名称: X" or "compound_name: X"
        patterns = [
            r'化合物名称\s*:\s*(.+?)(?:\n|$)',
            r'compound_name\s*:\s*(.+?)(?:\n|$)',
            r'标题\s*:\s*(.+?)(?:\n|$)',
            r'title\s*:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean_name = match.strip().strip('*').strip()
                if clean_name and len(clean_name) > 1:
                    compound_names.add(clean_name)
        
        # Also look for main section headers that might be compound names
        # Look for lines starting with # followed by what looks like a compound name
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# ') and not line.startswith('# 1.') and not line.startswith('# 2.'):
                # Extract the title after #
                title = line[2:].strip()
                if title and len(title) > 2 and not title.startswith('论文信息') and not title.startswith('文献基础信息'):
                    compound_names.add(title)
        
        return compound_names
    
    except Exception as e:
        print(f"Error reading markdown file: {e}")
        return set()

def extract_title_from_filename(filename):
    """Extract title from PDF filename."""
    # Remove .pdf extension
    title = filename.replace('.pdf', '')
    
    # Remove author name if present (usually after underscore)
    if '_' in title:
        # Split by underscore and take the first part
        parts = title.split('_')
        if len(parts) > 1:
            # Check if the last part looks like an author name (Chinese characters)
            last_part = parts[-1]
            if re.search(r'[\u4e00-\u9fff]', last_part) and len(last_part) <= 10:
                title = '_'.join(parts[:-1])
    
    return title

def get_pdf_files(downloads_path):
    """Get all PDF files from Downloads folder."""
    pdf_files = []
    try:
        for file in Path(downloads_path).glob('*.pdf'):
            pdf_files.append(file)
        return pdf_files
    except Exception as e:
        print(f"Error accessing Downloads folder: {e}")
        return []

def normalize_text(text):
    """Normalize text for comparison by removing special characters and converting to lowercase."""
    if not text:
        return ""
    # Remove special characters, keep Chinese characters, letters, and numbers
    normalized = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
    return normalized.lower().strip()

def check_missing_pdfs(md_file_path, downloads_path):
    """Main function to check which PDFs are missing from the markdown file."""
    print("Extracting compound names from markdown file...")
    md_compounds = extract_compound_names_from_md(md_file_path)
    
    print(f"Found {len(md_compounds)} compound names in markdown file:")
    for compound in sorted(md_compounds):
        print(f"  - {compound}")
    
    print(f"\nScanning PDF files in {downloads_path}...")
    pdf_files = get_pdf_files(downloads_path)
    
    if not pdf_files:
        print("No PDF files found in Downloads folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF files.")
    
    missing_pdfs = []
    matched_pdfs = []
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        pdf_title = extract_title_from_filename(pdf_file.name)
        
        if pdf_title:
            print(f"  Extracted title from filename: {pdf_title}")
            
            # Normalize both titles for comparison
            normalized_pdf_title = normalize_text(pdf_title)
            found_match = False
            
            for md_compound in md_compounds:
                normalized_md_compound = normalize_text(md_compound)
                
                # Check if there's a match (either way)
                if (normalized_pdf_title in normalized_md_compound or 
                    normalized_md_compound in normalized_pdf_title or
                    normalized_pdf_title == normalized_md_compound):
                    print(f"  ✓ Matched with: {md_compound}")
                    matched_pdfs.append((pdf_file.name, pdf_title, md_compound))
                    found_match = True
                    break
            
            if not found_match:
                print(f"  ✗ No match found")
                missing_pdfs.append((pdf_file.name, pdf_title))
        else:
            print(f"  ✗ Could not extract title")
            missing_pdfs.append((pdf_file.name, "Could not extract title"))
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Matched PDFs: {len(matched_pdfs)}")
    print(f"Missing PDFs: {len(missing_pdfs)}")
    
    if matched_pdfs:
        print(f"\nMATCHED PDFs:")
        for filename, pdf_title, md_compound in matched_pdfs:
            print(f"  ✓ {filename}")
            print(f"    PDF Title: {pdf_title}")
            print(f"    MD Compound: {md_compound}")
            print()
    
    if missing_pdfs:
        print(f"\nMISSING PDFs (not found in markdown file):")
        for filename, pdf_title in missing_pdfs:
            print(f"  ✗ {filename}")
            print(f"    Title: {pdf_title}")
            print()
    else:
        print(f"\nAll PDFs were matched with compounds in the markdown file!")

if __name__ == "__main__":
    # Configuration
    md_file_path = "急毒.md"
    downloads_path = "/home/richard/Downloads"
    
    # Check if files exist
    if not os.path.exists(md_file_path):
        print(f"Error: Markdown file '{md_file_path}' not found!")
        exit(1)
    
    if not os.path.exists(downloads_path):
        print(f"Error: Downloads folder '{downloads_path}' not found!")
        exit(1)
    
    # Run the check
    check_missing_pdfs(md_file_path, downloads_path) 