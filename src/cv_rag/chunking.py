"""CV-aware chunking.

CVs have a predictable structure (Summary, Experience, Education, Skills...).
We first split a CV into sections, then split long sections into overlapping
chunks. Every chunk is prefixed with a short header ("Candidate: X | Section: Y")
so that a chunk retrieved on its own still says whose CV it came from - this
matters a lot when many CVs share one index.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# Canonical section name -> heading variants found in real CVs.
SECTION_ALIASES: dict[str, list[str]] = {
    "Summary": ["summary", "profile", "professional summary", "about me", "objective",
                "career objective", "personal statement", "overview"],
    "Experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history", "career history", "employment"],
    "Education": ["education", "academic background", "qualifications",
                  "academic qualifications", "education and training"],
    "Skills": ["skills", "technical skills", "core skills", "key skills", "competencies",
               "core competencies", "tech stack", "technologies", "tools"],
    "Projects": ["projects", "key projects", "personal projects", "selected projects"],
    "Certifications": ["certifications", "certificates", "licenses", "licences",
                       "licenses and certifications", "courses"],
    "Awards": ["awards", "honors", "honours", "achievements", "accomplishments"],
    "Publications": ["publications", "papers", "research"],
    "Languages": ["languages"],
    "Volunteering": ["volunteering", "volunteer experience", "community"],
    "Interests": ["interests", "hobbies"],
    "Contact": ["contact", "contact information", "personal details", "personal information"],
}

_HEADING_LOOKUP = {alias: canon for canon, aliases in SECTION_ALIASES.items() for alias in aliases}


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        key = f"{self.metadata.get('source')}::{self.metadata.get('chunk_index')}::{self.text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _normalise_heading(line: str) -> str | None:
    """Return the canonical section name if `line` looks like a CV heading."""
    stripped = line.strip().strip("#*_:-=| ").strip()
    if not stripped or len(stripped) > 40:
        return None
    return _HEADING_LOOKUP.get(stripped.lower())


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split CV text into (section_name, section_text) pairs, in order."""
    sections: list[tuple[str, list[str]]] = [("Header", [])]
    for line in text.splitlines():
        heading = _normalise_heading(line)
        if heading:
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    result = [(name, "\n".join(lines).strip()) for name, lines in sections]
    return [(name, body) for name, body in result if body]


_NAME_LINE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'.\- ]+$")


def guess_candidate_name(text: str, filename: str) -> str:
    """Heuristic: a CV usually starts with the candidate's name on its own line."""
    for line in text.splitlines()[:8]:
        candidate = line.strip().strip("#*_ ").strip()
        if not candidate or _normalise_heading(candidate):
            continue
        words = candidate.split()
        if 2 <= len(words) <= 5 and _NAME_LINE.match(candidate) and len(candidate) <= 50:
            return candidate.title() if candidate.isupper() else candidate
    stem = re.sub(r"(?i)[_\-\s]*(cv|resume|curriculum[_\-\s]?vitae)[_\-\s]*", " ", Path(filename).stem)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem.title() or "Unknown Candidate"


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "• ", "- ", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_cv(
    text: str,
    source: str,
    candidate: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Turn one CV into retrieval-ready chunks with metadata."""
    candidate = candidate or guess_candidate_name(text, source)
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for piece in _split_long(body, chunk_size, chunk_overlap):
            piece = piece.strip()
            if not piece:
                continue
            header = f"Candidate: {candidate} | Section: {section}"
            chunks.append(
                Chunk(
                    text=f"{header}\n{piece}",
                    metadata={
                        "candidate": candidate,
                        "section": section,
                        "source": Path(source).name,
                        "chunk_index": len(chunks),
                    },
                )
            )
    return chunks
