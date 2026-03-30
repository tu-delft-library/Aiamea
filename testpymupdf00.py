import fitz  # pymupdf

PDF_PATH = r"E:\XRZONE_Files\PDFExtractor\pdf-ris\samples\batches\03\SEG016.pdf"
FIRST_PAGE = 0  # 0-indexed in pymupdf
LAST_PAGE = 1   # inclusive

doc = fitz.open(PDF_PATH)

for page_num in range(FIRST_PAGE, LAST_PAGE + 1):
    page = doc[page_num]
    text = page.get_text("text")  # plain text
    print(f"\n--- PAGE {page_num + 1} ---\n")
    print(text)

doc.close()