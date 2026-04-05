import os
import re
from pathlib import Path

def brutal_markdown_scrubber(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.strip():
        return

    # 1. Nuke the Top Nav Garbage
    header_match = re.search(r'^(# |## )', content, re.MULTILINE)
    if header_match:
        content = content[header_match.start():]

    # 2. Guillotine the Footer
    footer_patterns = [
        r'#### Stay updated', 
        r'## Quick Links', 
        r'Subscribe for our press releases'
    ]
    
    for pattern in footer_patterns:
        footer_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if footer_match:
            content = content[:footer_match.start()]

    # 3. Purge Image Tags
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

    # 4. Eradicate Arabic Text (The English-Only Enforcer)
    # Strips standard Arabic block and presentation forms
    content = re.sub(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]+', '', content)

    # 5. Clean up the Whitespace Apocalypse
    # Stripping images and Arabic text will leave bizarre floating punctuation and massive gaps.
    # This condenses horizontal spaces and vertical line breaks.
    content = re.sub(r' {2,}', ' ', content) # Condense multiple spaces
    content = re.sub(r'\n{3,}', '\n\n', content).strip()

    # Save to the clean directory
    out_path = Path(output_dir) / Path(file_path).name
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Purged and Cleaned: {Path(file_path).name}")

def process_corpus(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    md_files = Path(input_dir).glob('*.md')
    
    for file in md_files:
        brutal_markdown_scrubber(file, output_dir)

# Execution
if __name__ == "__main__":
    process_corpus('data', 'cleaned_data')