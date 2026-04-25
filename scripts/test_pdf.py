import pymupdf
import pytesseract
from PIL import Image
import io

doc = pymupdf.open("data_lake/corpus.pdf")

# render page as image then OCR it
page = doc[20]
mat = pymupdf.Matrix(2, 2)  # 2x zoom — better OCR accuracy
pix = page.get_pixmap(matrix=mat)

# convert to PIL image
img = Image.open(io.BytesIO(pix.tobytes("png")))

# OCR
text = pytesseract.image_to_string(img)
print(text[:500])
