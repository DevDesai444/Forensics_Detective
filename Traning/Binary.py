#!/usr/bin/env python
"""
Binary.py

Module for converting a PDF file into a grayscale image (byte-to-pixel mapping)
and extracting embedded generator signatures, saving them to JSON.

Functions:
    convert_pdf_to_image(pdf_path, width=None, square=True) -> output_image_path
    extract_pdf_signatures(pdf_path) -> output_json_path

Usage (as script):
    python Binary.py /path/to/document.pdf
"""

import sys
import os
import json
import math

try:
    from PIL import Image
    import numpy as np
except ImportError:
    sys.stderr.write("Error: install Pillow and numpy: pip install Pillow numpy\n")
    sys.exit(1)


def read_pdf_bytes(path):
    """Read raw bytes from PDF."""
    with open(path, 'rb') as f:
        return f.read()


def calculate_dimensions(length, width=None, square=True):
    """Compute image width & height."""
    if width:
        w = width
        h = int(math.ceil(length / float(w)))
    elif square:
        side = int(math.ceil(math.sqrt(length)))
        w = h = side
    else:
        w = int(math.ceil(math.sqrt(length * 1.618)))
        h = int(math.ceil(length / float(w)))
    return w, h


def bytes_to_image(data, width, height):
    """Map bytes to grayscale image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    total = width * height
    if arr.size < total:
        arr = np.pad(arr, (0, total - arr.size), 'constant')
    else:
        arr = arr[:total]
    img = arr.reshape((height, width))
    return Image.fromarray(img, 'L')


def save_image(img, pdf_path):
    """Save image next to PDF and return its path."""
    out = os.path.splitext(pdf_path)[0] + '.png'
    img.save(out)
    return out


def detect_generator_signatures(data):
    """Detect common PDF generator signatures."""
    sigs = {'creator': None, 'producer': None, 'creation_tool': None}
    creators = {
        b'Microsoft Word':'Microsoft Word', b'Adobe Acrobat':'Adobe Acrobat',
        b'PDFCreator':'PDFCreator',    b'pdfTeX':'LaTeX/pdfTeX',
        b'iText':'iText',               b'Foxit':'Foxit',
        b'LibreOffice':'LibreOffice',   b'Google Docs':'Google Docs',
        b'Ghostscript':'Ghostscript',   b'wkhtmltopdf':'wkhtmltopdf',
        b'WeasyPrint':'WeasyPrint',     b'ReportLab':'ReportLab'
    }
    for bs, name in creators.items():
        idx = data.find(bs)
        if idx != -1:
            if sigs['creator'] is None:
                sigs['creator'] = name
            win = data[max(0, idx-200):idx+200]
            if b'/Producer' in win and sigs['producer'] is None:
                sigs['producer'] = name
    if data.find(b'/Creator') != -1 and sigs['creation_tool'] is None:
        sigs['creation_tool'] = 'Found /Creator tag'
    return sigs


def save_signatures(sigs, pdf_path):
    """Save signatures JSON next to PDF and return its path."""
    out = os.path.splitext(pdf_path)[0] + '.signatures.json'
    with open(out, 'w') as f:
        json.dump(sigs, f, indent=2)
    return out


def convert_pdf_to_image(pdf_path, width=None, square=True):
    """
    Convert PDF to grayscale image.
    Returns the path of the saved PNG.
    """
    data = read_pdf_bytes(pdf_path)
    w, h = calculate_dimensions(len(data), width=width, square=square)
    img = bytes_to_image(data, w, h)
    return save_image(img, pdf_path)


def extract_pdf_signatures(pdf_path):
    """
    Extract PDF generator signatures.
    Returns the path of the saved JSON.
    """
    data = read_pdf_bytes(pdf_path)
    sigs = detect_generator_signatures(data)
    return save_signatures(sigs, pdf_path)


def process_pdf(pdf_path, width=None, square=True):
    """
    Convenience function: convert to image and extract signatures.
    Returns tuple (image_path, json_path).
    """
    img_path = convert_pdf_to_image(pdf_path, width=width, square=square)
    json_path = extract_pdf_signatures(pdf_path)
    return img_path, json_path


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python Binary.py /path/to/document.pdf\n")
        sys.exit(1)

    pdf = sys.argv[1]
    if not os.path.isfile(pdf):
        sys.stderr.write("Error: File not found: %s\n" % pdf)
        sys.exit(1)

    img_path, sig_path = process_pdf(pdf)
    sys.stdout.write("✓ Image saved: %s\n" % img_path)
    sys.stdout.write("✓ Signatures saved: %s\n" % sig_path)

# Dataset/Google_Docs/..
# Dataset/MSOffice/..
# Dataset/Python/..
# Dataset/LaTeX/..
# Dataset/LibreOffice/..

if __name__ == '__main__':
    main()