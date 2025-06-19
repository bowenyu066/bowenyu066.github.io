#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import glob
import sys
import getpass

LOCAL_DIR = 'files'
POSTS_DIR = '_posts'
SITE_DIR = '_site'
ENCRYPTED_DIR = 'encrypted'

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
    r"""Replace percent signs (\\%) with original percent signs
    in LaTeX content."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        # Replace \% with %
        lines[i] = line.replace(r'\%', '%')
    return '\n'.join(lines)

def process_markdown(md_path):
    """Process the markdown file before encryption. Right now, the only
    processing is to convert all inline LaTeX formulas (enclosed by $...$)
    into double dollar signs ($$...$$) for display math."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Convert inline LaTeX to display math
    content = inline_latex_to_display(content)
    content = replace_percent_signs(content)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    # print(f"Processed markdown: {md_path}")

def parse_front_matter(md_path):
    """Extract YAML front matter lines from the markdown file."""
    fm_lines = []
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines or not lines[0].strip() == '---':
        return []
    for line in lines[1:]:
        fm_lines.append(line)
        if line.strip() == '---':
            break
    return fm_lines

def find_generated_html(base_name):
    """Locate the built HTML in the _site directory."""
    p1 = os.path.join(SITE_DIR, 'posts', '*', '*', base_name, 'index.html')
    p2 = os.path.join(SITE_DIR, 'posts', '*', '*', f'{base_name}.html')
    matches = glob.glob(p1) + glob.glob(p2)
    if not matches:
        print(f"Error: Generated HTML file {base_name} not found.", file=sys.stderr)
        sys.exit(1)
    return matches[0]

def main():
    parser = argparse.ArgumentParser(description="Encrypt a Jekyll post with staticrypt")
    parser.add_argument('md_file', help="Markdown filename in the local directory (e.g., my-post.md)")
    # parser.add_argument('password', help="Password to use for encryption")
    password = getpass.getpass("🔒 Enter encryption password: ")
    args = parser.parse_args()

    src_md = os.path.join(LOCAL_DIR, args.md_file)
    if not os.path.isfile(src_md):
        print(f"Error: {src_md} not found.", file=sys.stderr)
        sys.exit(1)

    # Read front matter for stub
    front_matter = parse_front_matter(src_md)

    # Move markdown to _posts
    dest_md = os.path.join(POSTS_DIR, args.md_file)
    shutil.copy2(src_md, dest_md)
    print(f"Moved {src_md} -> {dest_md}")
    
    # Process the markdown file
    process_markdown(dest_md)

    try:
        # Build the site
        print("Running Jekyll build...")
        subprocess.run(['bundle', 'exec', 'jekyll', 'build'], check=True)

        # Find HTML
        base_name = os.path.splitext(args.md_file)[0]
        print(base_name)
        html_path = find_generated_html(base_name)
        html_file_name = os.path.basename(html_path)
        print(f"Found HTML at {html_path}, name: {html_file_name}")

        # Ensure encrypted dir exists
        os.makedirs(ENCRYPTED_DIR, exist_ok=True)
        out_html = os.path.join(ENCRYPTED_DIR, f'{base_name}.html')
        if os.path.exists(out_html):
            print(f"Warning: {out_html} already exists. It will be overwritten.")
            os.remove(out_html)

        # Run staticrypt
        print("Running staticrypt...")
        subprocess.run([
            'staticrypt', html_path,
            '-p', password,
            '-o', out_html
        ], check=True)
        os.rename(os.path.join(ENCRYPTED_DIR, f'{html_file_name}'), out_html)
        print(f"Encrypted HTML -> {out_html}")
    
    finally:
        # Delete original markdown from _posts
        os.remove(dest_md)
        print(f"Removed unencrypted post: {dest_md}")

    # Write stub markdown linking to encrypted file
    # stub_lines = ['---\n'] + front_matter + ['\n']
    # stub_lines.append('This post is password-protected. Click below to view:\n\n')
    # stub_lines.append(f'👉 [Access Encrypted Version](/encrypted/{base_name}.html)\n')
    # with open(dest_md, 'w', encoding='utf-8') as f:
    #     f.writelines(stub_lines)
    # print(f"Wrote stub post: {dest_md}")

    print("All done!")

if __name__ == '__main__':
    main()