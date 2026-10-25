from pathlib import Path

from cv_rag.chunking import chunk_cv, guess_candidate_name, split_sections

SAMPLES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def test_split_sections_detects_headings():
    text = "Jane Doe\nEngineer\n\nEXPERIENCE\nAcme 2020\n\n## Technical Skills\nPython\nEducation:\nBSc"
    names = [n for n, _ in split_sections(text)]
    assert names == ["Header", "Experience", "Skills", "Education"]


def test_guess_name_from_first_line():
    assert guess_candidate_name("JANE DOE\nData Engineer", "x.pdf") == "Jane Doe"


def test_guess_name_falls_back_to_filename():
    assert guess_candidate_name("Curriculum\n12345", "john_smith_resume.pdf") == "John Smith"


def test_chunks_carry_candidate_header_and_metadata():
    text = (SAMPLES / "priya_raman_cv.md").read_text()
    chunks = chunk_cv(text, source="priya_raman_cv.md", chunk_size=2000)
    assert chunks, "expected chunks"
    assert all(c.metadata["candidate"] == "Priya Raman" for c in chunks)
    assert all(c.text.startswith("Candidate: Priya Raman | Section: ") for c in chunks)
    sections = {c.metadata["section"] for c in chunks}
    assert {"Experience", "Education", "Skills", "Certifications"} <= sections
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique and stable


def test_guess_name_for_various_headers():
    cases = {
        "Daniel Okafor\nPlatform Engineer": "Daniel Okafor",
        "SOFIA MARTINEZ\nProduct Manager": "Sofia Martinez",
    }
    for text, expected in cases.items():
        assert guess_candidate_name(text, "cv.md") == expected
