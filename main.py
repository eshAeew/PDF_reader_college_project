"""Extract student marks from a PDF into CSV.

This script scans a PDF for student records containing:
- USN
- Name
- Roll number
- Subject marks
- Total (last numeric value)

Usage:
  python main.py --input PDF.pdf --output students.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


USN_PATTERN = re.compile(r"\b[A-Z0-9]{6,12}\b")
ROLL_PATTERN = re.compile(r"\b\d{1,4}\b")
NUMBER_PATTERN = re.compile(r"\b\d{1,3}\b")
NAME_CLEAN_PATTERN = re.compile(r"[^A-Za-z .'-]")


@dataclass
class StudentRecord:
    usn: str
    name: str
    roll: str
    marks: List[str]


class PdfTextExtractorError(RuntimeError):
    """Raised when no PDF text extraction method is available."""


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF using available local tools."""
    try:
        from pypdf import PdfReader  # type: ignore

        return _extract_with_pypdf(pdf_path, PdfReader)
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            return _extract_with_pypdf(pdf_path, PdfReader)
        except ImportError:
            return _extract_with_pdftotext(pdf_path)


def _extract_with_pypdf(pdf_path: Path, reader_cls) -> str:
    reader = reader_cls(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_with_pdftotext(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PdfTextExtractorError(
            "No PDF parser available. Install 'pypdf' or 'PyPDF2', "
            "or install the 'pdftotext' utility."
        ) from exc
    return result.stdout


def normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def chunk_records(lines: Sequence[str]) -> List[str]:
    """Group lines so each chunk begins with a USN line."""
    chunks: List[str] = []
    current: List[str] = []
    for line in lines:
        if USN_PATTERN.search(line):
            if current:
                chunks.append(" ".join(current))
                current = []
        current.append(line)
    if current:
        chunks.append(" ".join(current))
    return chunks


def parse_record(chunk: str) -> StudentRecord | None:
    """Parse a single record chunk into a StudentRecord."""
    usn_match = USN_PATTERN.search(chunk)
    if not usn_match:
        return None
    usn = usn_match.group(0)

    after_usn = chunk[usn_match.end() :].strip()
    tokens = after_usn.split()
    if not tokens:
        return None

    roll = ""
    name_tokens: List[str] = []
    marks: List[str] = []

    roll_idx = None
    for idx, token in enumerate(tokens):
        if ROLL_PATTERN.fullmatch(token):
            roll = token
            roll_idx = idx
            break
        name_tokens.append(token)

    if roll_idx is not None:
        marks = [t for t in tokens[roll_idx + 1 :] if NUMBER_PATTERN.fullmatch(t)]

    name_raw = " ".join(name_tokens).strip()
    name = NAME_CLEAN_PATTERN.sub("", name_raw).strip()

    if not name or not roll:
        return None

    return StudentRecord(usn=usn, name=name, roll=roll, marks=marks)


def parse_records(text: str) -> List[StudentRecord]:
    lines = normalize_lines(text)
    chunks = chunk_records(lines)
    records: List[StudentRecord] = []
    for chunk in chunks:
        record = parse_record(chunk)
        if record:
            records.append(record)
    return records


def build_headers(records: Iterable[StudentRecord]) -> List[str]:
    max_marks = max((len(record.marks) for record in records), default=0)
    subject_headers = [f"Subject_{idx + 1}" for idx in range(max_marks - 1)]
    headers = ["USN", "Name", "Roll"] + subject_headers
    if max_marks:
        headers.append("Total")
    return headers


def write_csv(output_path: Path, records: Sequence[StudentRecord]) -> None:
    headers = build_headers(records)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for record in records:
            row = [record.usn, record.name, record.roll]
            if headers[-1:] == ["Total"]:
                subject_count = len(headers) - 4
                subjects = record.marks[:subject_count]
                total = record.marks[subject_count : subject_count + 1]
                row.extend(subjects)
                row.extend([""] * (subject_count - len(subjects)))
                row.extend(total if total else [""])
            else:
                row.extend(record.marks)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract student marks to CSV.")
    parser.add_argument("--input", default="PDF.pdf", type=Path, help="PDF file path")
    parser.add_argument(
        "--output", default="students.csv", type=Path, help="CSV output path"
    )
    args = parser.parse_args()

    text = extract_text(args.input)
    records = parse_records(text)
    if not records:
        raise SystemExit("No student records found. Check PDF format.")

    write_csv(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
