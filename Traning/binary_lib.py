#!/usr/bin/env python
"""
binary_lib.py

Library for PDF binary-to-image conversion and embedded signature extraction.

Functions:
    read_pdf_bytes(path) -> bytes
    calculate_dimensions(length, width=None, square=True) -> (width, height)
    bytes_to_image(data, width, height) -> PIL.Image
    detect_generator_signatures(data) -> dict
    save_image(img, pdf_path) -> output_image_path
    save_signatures(sigs, pdf_path) -> output_json_path
    convert_pdf_to_image(pdf_path, width=None, square=True) -> output_image_path
    extract_pdf_signatures(pdf_path) -> output_json_path
    process_pdf(pdf_path, width=None, square=True) -> (image_path, json_path)
"""

import os
import json
import math

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise ImportError("Install dependencies: pip install Pillow numpy")

def read_pdf_bytes(path):
    """Read raw bytes from PDF."""
    with open(path, 'rb') as f:
        return f.read()

def calculate_dimensions(length, width=None, square=True):
    """Compute image width & height based on byte length."""
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
    """Map bytes to a grayscale PIL Image of given dimensions."""
    arr = np.frombuffer(data, dtype=np.uint8)
    total = width * height
    if arr.size < total:
        arr = np.pad(arr, (0, total - arr.size), 'constant')
    else:
        arr = arr[:total]
    img_array = arr.reshape((height, width))
    return Image.fromarray(img_array, 'L')

def detect_generator_signatures(data):
    """Detect common PDF generator signatures in raw byte data."""
    sigs = {'creator': None, 'producer': None, 'creation_tool': None}
    creators = {
        b'Microsoft Word': 'Microsoft Word',
        b'Adobe Acrobat': 'Adobe Acrobat',
        b'PDFCreator': 'PDFCreator',
        b'pdfTeX': 'LaTeX/pdfTeX',
        b'iText': 'iText',
        b'Foxit': 'Foxit',
        b'LibreOffice': 'LibreOffice',
        b'Google Docs': 'Google Docs',
        b'Ghostscript': 'Ghostscript',
        b'wkhtmltopdf': 'wkhtmltopdf',
        b'WeasyPrint': 'WeasyPrint',
        b'ReportLab': 'ReportLab'
    }
    for marker, name in creators.items():
        idx = data.find(marker)
        if idx != -1:
            if sigs['creator'] is None:
                sigs['creator'] = name
            window = data[max(0, idx - 200): idx + 200]
            if b'/Producer' in window and sigs['producer'] is None:
                sigs['producer'] = name
    if data.find(b'/Creator') != -1 and sigs['creation_tool'] is None:
        sigs['creation_tool'] = 'Found /Creator tag'
    return sigs

def save_image(img, pdf_path):
    """Save PIL Image next to PDF file and return output path."""
    out_path = os.path.splitext(pdf_path)[0] + '.png'
    img.save(out_path)
    return out_path

def save_signatures(sigs, pdf_path):
    """Save signatures dict next to PDF as JSON and return output path."""
    out_path = os.path.splitext(pdf_path)[0] + '.signatures.json'
    with open(out_path, 'w') as f:
        json.dump(sigs, f, indent=2)
    return out_path

def convert_pdf_to_image(pdf_path, width=None, square=True):
    """
    Convert PDF to grayscale image.
    Returns path of saved PNG.
    """
    data = read_pdf_bytes(pdf_path)
    w, h = calculate_dimensions(len(data), width=width, square=square)
    img = bytes_to_image(data, w, h)
    return save_image(img, pdf_path)

def extract_pdf_signatures(pdf_path):
    """
    Extract generator signatures from PDF.
    Returns path of saved JSON.
    """
    data = read_pdf_bytes(pdf_path)
    sigs = detect_generator_signatures(data)
    return save_signatures(sigs, pdf_path)

def process_pdf(pdf_path, width=None, square=True):
    """
    Convert PDF to image and extract signatures.
    Returns tuple (image_path, json_path).
    """
    img_path = convert_pdf_to_image(pdf_path, width=width, square=square)
    json_path = extract_pdf_signatures(pdf_path)
    return img_path, json_path
