import os
from pathlib import Path


def save_uploaded_file(uploaded_file, dest_path: Path):
    """Save a Streamlit UploadedFile to disk."""
    # Make sure the folder exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Now write the file
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def extract_text(file_path: Path) -> str:
    """Extract text from PDF, TXT, or MD files.
    Uses PyPDF2 for PDFs. For scanned PDFs, run OCR separately.
    """
    suf = file_path.suffix.lower()

    if suf == ".pdf":
        try:
            import PyPDF2
        except ImportError:
            raise RuntimeError("Install PyPDF2 to extract PDF text")

        text = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    elif suf in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {suf}")
