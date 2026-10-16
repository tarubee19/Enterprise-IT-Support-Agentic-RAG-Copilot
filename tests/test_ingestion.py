import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from docx import Document as DocxDocument

from app.services.ingestion import SUPPORTED, chunk_documents, load_file

SAMPLE_KB = Path(__file__).resolve().parents[1] / "data" / "sample_kb"


def test_supported_types():
    assert SUPPORTED == {".pdf", ".txt", ".md", ".docx"}


def test_load_markdown_sample():
    docs = load_file(SAMPLE_KB / "company_it_handbook.md")
    assert len(docs) == 1
    assert "VPN" in docs[0].page_content


def test_load_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Printer queue troubleshooting steps.", encoding="utf-8")
    docs = load_file(f)
    assert "Printer" in docs[0].page_content


def test_load_docx(tmp_path):
    f = tmp_path / "policy.docx"
    d = DocxDocument()
    d.add_paragraph("Guest Wi-Fi password rotates monthly.")
    d.save(str(f))
    docs = load_file(f)
    assert "Wi-Fi" in docs[0].page_content


def test_unsupported_type(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(f)


def test_chunking_respects_size():
    docs = load_file(SAMPLE_KB / "company_it_handbook.md")
    chunks = chunk_documents(docs)
    assert len(chunks) >= 2
    assert all(len(c.page_content) <= 900 for c in chunks)
