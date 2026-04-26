import io
import re

import pymupdf
import pytesseract
from PIL import Image

doc = pymupdf.open("data_lake/corpus.pdf")

chapter_pattern = re.compile(
    r"CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN"
    r"|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN"
    r"|EIGHTEEN|NINETEEN|TWENTY|THIRTY|FORTY|\d+)",
    re.IGNORECASE,
)

# scan first 200 pages — should catch book 1 chapters
hits = []
for i in range(200):
    page = doc[i]
    mat = pymupdf.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)

    if chapter_pattern.search(text):
        # show what the chapter heading looks like in the raw OCR
        lines = [l for l in text.split("\n") if chapter_pattern.search(l)]
        hits.append({"page": i + 1, "lines": lines})
        print(f"Page {i + 1}: {lines}")

print(f"\nFound {len(hits)} chapter headings in first 200 pages")
