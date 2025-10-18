#!/usr/bin/env python
"""
batch_convert.py

Recursively converts all PDF files in specified directories to grayscale images
and extracts generator signatures using binary_lib.convert_pdf.
"""

import os
import sys
from binary_lib import convert_pdf

# List of dataset directories (expand ~)
folders = [
    os.path.expanduser('~/Dataset/Google_Docs'),
    os.path.expanduser('~/Dataset/MSOffice'),
    os.path.expanduser('~/Dataset/Python'),
    os.path.expanduser('~/Dataset/LaTeX'),
    os.path.expanduser('~/Dataset/LibreOffice'),
]

def find_pdfs(root_dir):
    """
    Recursively yield all .pdf file paths under root_dir.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.pdf'):
                yield os.path.join(dirpath, fname)

def batch_convert():
    """
    Loop through each dataset folder, find PDFs, and convert them.
    """
    total = 0
    for folder in folders:
        if not os.path.isdir(folder):
            sys.stderr.write("Warning: directory not found: %s\n" % folder)
            continue
        for pdf_path in find_pdfs(folder):
            try:
                img_path, sigs = convert_pdf(pdf_path)
                print("✓ Converted: %s → %s" % (pdf_path, img_path))
            except Exception as e:
                sys.stderr.write("Error processing %s: %s\n" % (pdf_path, e))
            total += 1
    print("\nBatch conversion completed. Processed %d files." % total)

if __name__ == '__main__':
    batch_convert()