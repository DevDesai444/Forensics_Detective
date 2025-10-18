import os
import random
from faker import Faker
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# Configuration
OUTPUT_DIR = "/Users/DEVDESAI1/Desktop/University_at_Buffalo/EAS510/Forensics_Detective/src/Mother_Pdfs"
NUM_PDFS = 1000
PAGE_SIZE = LETTER
FONT_NAME = "Helvetica"
TITLE_FONT_SIZE = 20
BODY_FONT_SIZE = 12
LINE_HEIGHT = 14

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Faker for random topics and text
fake = Faker()

def make_pdf(file_path: str, title: str, body_text: str):
    """
    Create a single PDF file with a title and body text.
    """
    c = canvas.Canvas(file_path, pagesize=PAGE_SIZE)
    width, height = PAGE_SIZE

    # Draw title
    c.setFont(FONT_NAME, TITLE_FONT_SIZE)
    c.drawCentredString(width / 2, height - 72, title)

    # Draw body
    c.setFont(FONT_NAME, BODY_FONT_SIZE)
    y = height - 72 - 2 * LINE_HEIGHT
    for line in body_text.split("\n"):
        if y < 72:
            c.showPage()
            c.setFont(FONT_NAME, BODY_FONT_SIZE)
            y = height - 72
        c.drawString(72, y, line)
        y -= LINE_HEIGHT

    c.save()

def generate_random_body(paragraphs: int = 5, sentences_per_paragraph: int = 6) -> str:
    """
    Generate a block of text consisting of a given number of paragraphs,
    each with a given number of sentences.
    """
    paras = []
    for _ in range(paragraphs):
        sentences = [fake.sentence(nb_words=random.randint(8, 15)) for _ in range(sentences_per_paragraph)]
        paras.append(" ".join(sentences))
    return "\n\n".join(paras)

def main():
    for i in range(1, NUM_PDFS + 1):
        # Generate a random title (topic)
        title = fake.catch_phrase()
        # Sanitize filename
        safe_title = "_".join(title.lower().split())
        filename = f"{safe_title}.pdf"
        file_path = os.path.join(OUTPUT_DIR, filename)

        # Generate random body text
        body = generate_random_body(paragraphs=7, sentences_per_paragraph=8)

        # Create PDF
        make_pdf(file_path, title, body)
        print(f"Created: {file_path}")

if __name__ == "__main__":
    main()