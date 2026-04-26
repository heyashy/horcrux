import io
import time

import pymupdf
import pytesseract
from PIL import Image

doc = pymupdf.open("data_lake/corpus.pdf")

# time 5 pages
start = time.time()
for i in range(20, 25):
    page = doc[i]
    mat = pymupdf.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)

elapsed = time.time() - start
per_page = elapsed / 5

# estimate total
total_pages = len(doc)
estimated_minutes = (total_pages * per_page) / 60

print(f"Pages in PDF: {total_pages}")
print(f"Per page: {per_page:.1f}s")
print(f"Estimated total: {estimated_minutes:.0f} minutes")
