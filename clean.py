import os
import re
from collections import Counter

# --- Configuration ---
INPUT_DIR = "data"
OUTPUT_DIR = "clean_data"
# If a line appears in >60% of files (normalized), it's considered boilerplate noise.
DF_THRESHOLD = 0.60 

def clean_markdown_regex(content):
    """
    Step 1: Structural cleaning using Regex.
    Target: Navigation patterns, empty links, and repetitive web artifacts.
    """
    # Remove lines that are just Markdown lists of links (Navigation menus)
    content = re.sub(r'^\s*[\*\-]\s*\[.*?\]\(.*?\)\s*$', '', content, flags=re.MULTILINE)
    
    # Remove standalone image/icon links like [ ](url)
    content = re.sub(r'\[\s*\]\(.*?\)', '', content)
    
    # Remove "Switch to..." breadcrumb metadata
    content = re.sub(r'\(.*?"Switch to.*?"\)', '', content)
    
    return content

def normalize_line(line):
    """
    Step 2: Semantic normalization for statistical analysis.
    Finds the 'soul' of the line by stripping formatting.
    """
    # Extract text from links: [Industrial Sector](url) -> Industrial Sector
    clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
    
    # Strip punctuation and lower-case
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', clean_text).lower().strip()
    
    # Collapse internal whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text

def exterminate_boilerplate():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Could not find '{INPUT_DIR}'.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.md')]
    if not files:
        print("No Markdown files found.")
        return

    print(f"🚀 Processing {len(files)} files...")
    
    line_counts = Counter()
    file_contents = {}

    # PASS 1: Regex Clean + Build Frequency Map
    for file in files:
        with open(os.path.join(INPUT_DIR, file), 'r', encoding='utf-8') as f:
            # Apply regex cleaning to the whole file first
            raw_text = f.read()
            regex_cleaned_text = clean_markdown_regex(raw_text)
            
            lines = regex_cleaned_text.splitlines()
            file_contents[file] = lines
            
            # Count document frequency of normalized lines
            unique_norm_lines = set()
            for line in lines:
                norm = normalize_line(line)
                if len(norm) > 2: 
                    unique_norm_lines.add(norm)
                    
            for norm in unique_norm_lines:
                line_counts[norm] += 1

    # Define the 'Toxic' threshold
    cutoff = len(files) * DF_THRESHOLD
    toxic_set = {line for line, count in line_counts.items() if count >= cutoff}
    
    print(f"Found {len(toxic_set)} boilerplate strings appearing in >{DF_THRESHOLD*100}% of docs.")

    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    # PASS 2: Final Purge and Write
    for file, lines in file_contents.items():
        output_path = os.path.join(OUTPUT_DIR, file)
        with open(output_path, 'w', encoding='utf-8') as f:
            consecutive_blanks = 0
            
            for line in lines:
                norm_line = normalize_line(line)
                
                # OBLITERATE if it matches the toxic statistical set
                if len(norm_line) > 2 and norm_line in toxic_set:
                    continue
                
                # Handle whitespace to keep the output tidy
                stripped = line.strip()
                if not stripped:
                    consecutive_blanks += 1
                    if consecutive_blanks <= 1: # Allow only one blank line
                        f.write('\n')
                else:
                    consecutive_blanks = 0
                    f.write(line + '\n')

    print(f"✨ Success. Cleaned files are in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    exterminate_boilerplate()