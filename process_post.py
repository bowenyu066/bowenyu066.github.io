#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import glob
import sys
import time
import re

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

def replace_mlir_with_cpp(content):
    """For code blocks that contain MLIR code, replace the language
    identifier from `mlir` to `cpp`."""
    # lines = content.splitlines()
    # new_lines = []
    # in_code_block = False
    # for line in lines:
    #     if line.strip().startswith('```'):
    #         if not in_code_block:
    #             language = line.strip().replace('```', '')
    #             if language == 'mlir':
    #                 new_lines.append('```cpp')
    #             else:
    #                 new_lines.append(line)
    #         else:
    #             new_lines.append(line)
    #         in_code_block = not in_code_block
    #     elif line.strip().startswith('> ```'):
    #         if not in_code_block:
    #             language = line.strip().replace('> ```', '').strip()
    #             if language == 'mlir':
    #                 new_lines.append('> ```cpp')
    #             else:
    #                 new_lines.append(line)
    #         else:
    #             new_lines.append(line)
    #         in_code_block = not in_code_block
    #     else:
    #         new_lines.append(line)
    # return '\n'.join(new_lines)
    # This pattern finds lines starting a code block with 'mlir'
    # and replaces 'mlir' with 'cpp', keeping the prefix.
    pattern = r'^(\s*(?:>\s*)?```)mlir\b'
    replacement = r'\1cpp'
    return re.sub(pattern, replacement, content, flags=re.MULTILINE)

def insert_empty_line_in_codeblock(content):
    """Insert an empty line at the beginning of every code block.
    However, if there's already an empty line at the beginning, do nothing."""
    lines = content.splitlines()
    in_code_block = False
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        line = line.strip()
        if line.startswith('```'):
            if not in_code_block:
                # Start of a code block
                if lines[i + 1].strip() != '': # Not an empty line
                    new_lines.append('')
            in_code_block = not in_code_block
        elif line.startswith('> ```'):
            if not in_code_block:
                # Start of a code block
                if lines[i + 1].strip() != '>': # Not an empty line
                    new_lines.append('>')
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
    content = replace_mlir_with_cpp(content)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    # print(f"Processed markdown: {md_path}")\

def main():
    parser = argparse.ArgumentParser(description="Process a markdown file and move it to the _posts directory")
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
    # Remove all possible original files (including those with different date prefixes)
    for f in os.listdir(POSTS_DIR):
        if f.endswith(file_name) and len(f) == len('YYYY-MM-DD-') + len(file_name):
            # Match files with exact pattern: YYYY-MM-DD-{file_name}
            prefix = f[:10]  # First 10 characters: YYYY-MM-DD
            if len(prefix) == 10 and prefix[4] == '-' and prefix[7] == '-' and prefix[:4].isdigit() and prefix[5:7].isdigit() and prefix[8:10].isdigit():
                os.remove(os.path.join(POSTS_DIR, f))
                print(f"Removed old file: {f}")
    shutil.copy2(src_md, dest_md)
    print(f"Moved {src_md} -> {dest_md}")
    
    # Process the markdown file
    process_markdown(dest_md)

    print("All done!")

if __name__ == '__main__':
    main()