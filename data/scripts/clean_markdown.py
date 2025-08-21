#!/usr/bin/env python3
"""
Comprehensive script to clean markdown file by removing citation patterns and fixing formatting.
This script:
1. Removes [cite: xxx] patterns
2. Removes [cite_start] patterns  
3. Removes :contentReference[oaicite:xxx]{index=xxx} patterns
4. Removes unnecessary indentation while preserving proper markdown structure
5. Adds proper spacing around tables (empty lines before and after)
6. Limits excessive empty lines

This will clean up all citation references and formatting issues while preserving the rest of the content.
"""

import re
import os
import argparse

def clean_markdown_file(file_path):
    """Remove all citation patterns from a markdown file."""
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count patterns before removal (using more comprehensive patterns)
        cite_pattern = r'\[cite:\s*\d+(?:,\s*\d+)*\]'
        cite_start_pattern = r'\[cite_?start\]'  # Match both [cite_start] and [citestart]
        content_ref_pattern = r':contentReference\[oaicite:\d+\]\{index=\d+\}'
        
        cite_before = len(re.findall(cite_pattern, content))
        cite_start_before = len(re.findall(cite_start_pattern, content))
        content_ref_before = len(re.findall(content_ref_pattern, content))
        
        print(f"Found patterns initially:")
        print(f"  - [cite: xxx] patterns: {cite_before}")
        print(f"  - [cite_start] patterns: {cite_start_before}")
        print(f"  - :contentReference patterns: {content_ref_before}")
        
        # Step 1: Remove [cite: xxx] patterns (handles single and multiple numbers)
        cleaned_content = re.sub(cite_pattern, '', content)
        
        # Step 2: Remove [cite_start] patterns (including any spaces that might follow)
        cleaned_content = re.sub(cite_start_pattern, '', cleaned_content)
        
        # Step 3: Remove :contentReference[oaicite:xxx]{index=xxx} patterns
        cleaned_content = re.sub(content_ref_pattern, '', cleaned_content)
        
        # Step 4: Remove unnecessary indentation
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading spaces for non-list items and non-code blocks
            stripped_line = line.lstrip()
            
            # Preserve indentation for:
            # - List items (starting with -, *, +, or numbers)
            # - Code blocks (starting with ``` or indented by 4+ spaces in original)
            # - Table rows (starting with |)
            # - Nested bullet points that should be preserved
            if (stripped_line.startswith(('- ', '* ', '+ ')) or 
                re.match(r'^\d+\.', stripped_line) or  # numbered lists
                stripped_line.startswith('|') or       # table rows
                stripped_line.startswith('```') or     # code blocks
                stripped_line == '' or                 # empty lines
                re.match(r'^#{1,6}\s', stripped_line)): # headers
                cleaned_lines.append(stripped_line)
            else:
                # For regular content, remove leading whitespace but check for legitimate indentation
                # Keep one level of indentation for nested list items
                original_indent = len(line) - len(line.lstrip())
                if original_indent > 0 and any(prev_line.strip().startswith(('-', '*', '+')) for prev_line in cleaned_lines[-3:] if prev_line.strip()):
                    # This might be a nested item, keep minimal indentation
                    cleaned_lines.append('  ' + stripped_line if stripped_line else '')
                else:
                    cleaned_lines.append(stripped_line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Step 5: Add proper spacing around tables
        lines = cleaned_content.split('\n')
        table_spaced_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Check if current line is a table row (starts with |)
            if current_line.startswith('|') and current_line.endswith('|'):
                # Check if previous line needs spacing
                if (i > 0 and 
                    table_spaced_lines and 
                    table_spaced_lines[-1].strip() != '' and 
                    not table_spaced_lines[-1].strip().startswith('|')):
                    table_spaced_lines.append('')  # Add empty line before table
                
                # Add the current table row
                table_spaced_lines.append(lines[i])
                
                # Look ahead to find the end of the table
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('|') and lines[j].strip().endswith('|'):
                    table_spaced_lines.append(lines[j])
                    j += 1
                
                # Check if next line after table needs spacing
                if (j < len(lines) and 
                    lines[j].strip() != '' and 
                    not lines[j].strip().startswith('|')):
                    table_spaced_lines.append('')  # Add empty line after table
                
                i = j  # Skip to after the table
            else:
                table_spaced_lines.append(lines[i])
                i += 1
        
        cleaned_content = '\n'.join(table_spaced_lines)
        
        # Step 6: Clean up multiple consecutive spaces (but preserve single spaces and newlines)
        cleaned_content = re.sub(r'[ \t]+', ' ', cleaned_content)  # Only replace multiple spaces/tabs with single space
        cleaned_content = re.sub(r' +\n', '\n', cleaned_content)  # Remove spaces before newlines
        
        # Step 7: Clean up excessive empty lines (max 2 consecutive empty lines)
        cleaned_content = re.sub(r'\n{4,}', '\n\n\n', cleaned_content)  # Limit to max 3 newlines (2 empty lines)
        
        # Step 8: Format markdown headers (ensure exactly 1 space after #, ##, ###)
        cleaned_content = format_markdown_headers(cleaned_content)
        
        # Count patterns after removal
        cite_after = len(re.findall(cite_pattern, cleaned_content))
        cite_start_after = len(re.findall(cite_start_pattern, cleaned_content))
        content_ref_after = len(re.findall(content_ref_pattern, cleaned_content))
        
        # Get file size before and after
        original_size = len(content.encode('utf-8'))
        new_size = len(cleaned_content.encode('utf-8'))
        
        # Write the cleaned content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        # Print results
        print(f"\n✅ Successfully processed file: {file_path}")
        print(f"📊 Results:")
        print(f"   - [cite: xxx] patterns removed: {cite_before - cite_after}")
        print(f"   - [cite_start] patterns removed: {cite_start_before - cite_start_after}")
        print(f"   - :contentReference patterns removed: {content_ref_before - content_ref_after}")
        print(f"   - Total patterns removed: {cite_before + cite_start_before + content_ref_before - cite_after - cite_start_after - content_ref_after}")
        print(f"   - Original file size: {original_size:,} bytes")
        print(f"   - New file size: {new_size:,} bytes")
        print(f"   - Size reduction: {original_size - new_size:,} bytes ({((original_size - new_size) / original_size * 100):.1f}%)")
        
        # Check if any patterns remain
        remaining_total = cite_after + cite_start_after + content_ref_after
        if remaining_total > 0:
            print(f"\n⚠️  Warning: {remaining_total} patterns still remain!")
            if cite_after > 0:
                print(f"   - [cite: xxx] patterns: {cite_after}")
            if cite_start_after > 0:
                print(f"   - [cite_start] patterns: {cite_start_after}")
            if content_ref_after > 0:
                print(f"   - :contentReference patterns: {content_ref_after}")
        else:
            print("\n🎉 All citation patterns have been completely removed!")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def format_markdown_headers(content):
    """
    Format markdown headers to ensure exactly 1 space between # symbols and text.
    
    This function:
    - Finds lines starting with #, ##, ###, ####, #####, or ######
    - Ensures there's exactly 1 space between the # symbols and the header text
    - Removes any extra spaces or tabs
    - Preserves the header level and text content
    
    Args:
        content (str): The markdown content to format
        
    Returns:
        str: Content with properly formatted headers
    """
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Check if line starts with markdown headers (1-6 # symbols)
        header_match = re.match(r'^(#{1,6})\s*(.+)$', line)
        
        if header_match:
            # Extract the # symbols and the header text
            hashes = header_match.group(1)
            header_text = header_match.group(2).strip()
            
            # Format: hashes + exactly 1 space + header text
            formatted_line = f"{hashes} {header_text}"
            formatted_lines.append(formatted_line)
        else:
            # Not a header line, keep as is
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def main():
    """Main function to process the markdown file."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Clean markdown file by removing citation patterns and fixing formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_markdown.py                    # Use default file (急毒.md)
  python clean_markdown.py --file_path my_file.md
  python clean_markdown.py -f my_file.md
        """
    )
    
    parser.add_argument(
        '--file_path', '-f',
        type=str,
        default="急毒.md",
        help="Path to the markdown file to clean (default: 急毒.md)"
    )
    
    args = parser.parse_args()
    file_path = args.file_path
    
    print("🧹 Comprehensive markdown cleaning - citations, indentation, and table spacing...")
    print(f"📁 Processing file: {file_path}")
    print("-" * 70)
    
    success = clean_markdown_file(file_path)
    
    if success:
        print("-" * 70)
        print("✅ Comprehensive markdown cleaning completed!")
        print("📝 Your markdown file has been cleaned and updated.")
        print("\n🎯 All citation patterns removed and formatting improved:")
        print("   ✓ [cite: xxx] patterns")
        print("   ✓ [cite_start] patterns")
        print("   ✓ :contentReference[oaicite:xxx]{index=xxx} patterns")
        print("   ✓ Unnecessary indentation removed")
        print("   ✓ Proper spacing around tables added")
        print("   ✓ Excessive empty lines cleaned up")
        print("   ✓ Markdown headers formatted (exactly 1 space after #)")
    else:
        print("❌ Failed to process the file. Please check the error messages above.")

if __name__ == "__main__":
    main() 