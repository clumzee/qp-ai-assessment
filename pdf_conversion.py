from PyPDF2 import PdfReader
from typing_extensions import Concatenate


def read_pdf_and_extract_text(path: str) -> str:
    pdfreader = PdfReader(path)

    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    return raw_text