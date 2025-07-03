#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import glob
import sys
import time

POSTS_DIR = '_posts'

def inline_latex_to_display(content):
    """Convert inline LaTeX formulas (enclosed by $...$) to display math
    (enclosed by $$...$$)."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        # Replace single $ with double $$ for inline LaTeX
        line = line.replace('$$', '$')
        lines[i] = line.replace('$', '$$')
    return '\n'.join(lines)

def replace_percent_signs(content):
    """Replace percent signs (\\%) with original percent signs
    in LaTeX content."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        # Replace \% with %
        lines[i] = line.replace('\\%', '%')
    return '\n'.join(lines)

def insert_empty_line_in_codeblock(content):
    """Insert an empty line at the beginning of every code block.
    However, if there's already an empty line at the beginning, do nothing."""
    lines = content.splitlines()
    in_code_block = False
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if line.startswith('```'):
            if not in_code_block:
                # Start of a code block
                if lines[i + 1].strip() != '': # Not an empty line
                    new_lines.append('')
            in_code_block = not in_code_block
    return '\n'.join(new_lines)

def process_markdown(md_path):
    """Process the markdown file."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Convert inline LaTeX to display math
    content = inline_latex_to_display(content)
    content = replace_percent_signs(content)
    content = insert_empty_line_in_codeblock(content)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    # print(f"Processed markdown: {md_path}")\

def main():
    parser = argparse.ArgumentParser(description="Encrypt a Jekyll post with staticrypt")
    parser.add_argument('md_file', help="Markdown filename in the local directory (e.g., my-post.md)")
    args = parser.parse_args()

    src_md = args.md_file
    if not os.path.isfile(src_md):
        print(f"Error: {src_md} not found.", file=sys.stderr)
        sys.exit(1)

    # Move markdown to _posts
    file_name = os.path.basename(src_md)
    file_name_date = time.strftime('%Y-%m-%d-', time.localtime()) + file_name # Prepend date to filename
    dest_md = os.path.join(POSTS_DIR, file_name_date)
    shutil.copy2(src_md, dest_md)
    print(f"Moved {src_md} -> {dest_md}")
    
    # Process the markdown file
    process_markdown(dest_md)

    print("All done!")

if __name__ == '__main__':
    main()