import io

import pytest

from cv_rag.loaders import clean_text, load_bytes


def test_load_markdown():
    text = load_bytes(b"Jane Doe\n\n\n\n## Skills\nPython   SQL", "jane.md")
    assert text == "Jane Doe\n\n## Skills\nPython SQL"


def test_load_docx_includes_tables():
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Jane Doe")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "Python, Spark"
    buf = io.BytesIO()
    doc.save(buf)
    text = load_bytes(buf.getvalue(), "jane.docx")
    assert "Jane Doe" in text and "Skills | Python, Spark" in text


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        load_bytes(b"x", "cv.rtf")


def test_empty_file_raises():
    with pytest.raises(ValueError, match="No text"):
        load_bytes(b"   \n ", "empty.txt")


def test_clean_text_strips_nulls_and_crlf():
    assert clean_text("a\x00b\r\nc") == "ab\nc"
