"""Extract student marks from a PDF into CSV.

This script scans a PDF for student records containing:
- USN
- Name
- Roll number
- Subject totals
- Overall total (last numeric value)

Usage:
  python main.py --input PDF.pdf --output students.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


USN_PATTERN = re.compile(r"\bU[A-Z0-9]{9,12}\b")
ROLL_PATTERN = re.compile(r"\b\d{4,6}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
NAME_CLEAN_PATTERN = re.compile(r"[^A-Za-z .'-]")
TEXT_STRING_PATTERN = re.compile(rb"\((?:\\.|[^\\)])*\)")
COURSE_CODE_PATTERN = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)+-?$")
PAGE_LABELS = {
    "Bangalore University",
    "Tabulation Register for",
    "Bachelor of Computer Applications",
    "Semester :",
    "Exam Month :",
    "Discipline :",
    "Printed Date:",
    "Class",
    "Max. Total",
    "Term Grade:",
    "Letter Grade",
}


@dataclass
class StudentRecord:
    usn: str
    name: str
    roll: str
    marks: dict[str, str]


class PdfTextExtractorError(RuntimeError):
    """Raised when PDF parsing fails."""


def extract_pdf_pages(pdf_path: Path) -> List[List[str]]:
    """Extract per-page lines using a minimal PDF parser."""
    data = pdf_path.read_bytes()
    objects = _parse_objects(data)
    pages_root = _find_pages_root(objects)
    page_ids = _walk_pages(objects, pages_root)

    pages: List[List[str]] = []
    for page_id in page_ids:
        content_ids = _page_content_ids(objects[page_id])
        page_text = _extract_page_text(objects, content_ids)
        pages.append(normalize_lines(page_text))
    return pages


def _parse_objects(data: bytes) -> dict[int, bytes]:
    pattern = re.compile(rb"(\d+)\s+(\d+)\s+obj(.*?)endobj", re.S)
    return {int(m.group(1)): m.group(3) for m in pattern.finditer(data)}


def _find_pages_root(objects: dict[int, bytes]) -> int:
    for obj_num, obj_body in objects.items():
        if b"/Type /Catalog" in obj_body:
            match = re.search(rb"/Pages\s+(\d+)\s+0\s+R", obj_body)
            if match:
                return int(match.group(1))
    raise PdfTextExtractorError("Failed to locate /Pages root in PDF.")


def _walk_pages(objects: dict[int, bytes], obj_num: int) -> List[int]:
    obj_body = objects[obj_num]
    if b"/Type /Page" in obj_body and b"/Type /Pages" not in obj_body:
        return [obj_num]
    kids = re.search(rb"/Kids\s*\[(.*?)\]", obj_body, re.S)
    if not kids:
        return []
    page_ids: List[int] = []
    for kid in re.findall(rb"(\d+)\s+0\s+R", kids.group(1)):
        page_ids.extend(_walk_pages(objects, int(kid)))
    return page_ids


def _page_content_ids(page_body: bytes) -> List[int]:
    content_ids = re.findall(rb"/Contents\s+(\d+)\s+0\s+R", page_body)
    if content_ids:
        return [int(x) for x in content_ids]
    array_match = re.search(rb"/Contents\s*\[(.*?)\]", page_body, re.S)
    if array_match:
        return [int(x) for x in re.findall(rb"(\d+)\s+0\s+R", array_match.group(1))]
    return []


def _extract_page_text(objects: dict[int, bytes], content_ids: Iterable[int]) -> str:
    parts: List[str] = []
    for content_id in content_ids:
        obj_body = objects.get(content_id, b"")
        stream_match = re.search(rb"stream\r?\n(.*?)endstream", obj_body, re.S)
        if not stream_match:
            continue
        stream = stream_match.group(1)
        if b"FlateDecode" in obj_body:
            stream = _inflate_stream(stream)
        parts.extend(_decode_text_strings(stream))
    return "\n".join(parts)


def _inflate_stream(stream: bytes) -> bytes:
    import zlib

    try:
        return zlib.decompress(stream)
    except zlib.error as exc:
        raise PdfTextExtractorError("Failed to decompress PDF content stream.") from exc


def _decode_text_strings(stream: bytes) -> List[str]:
    decoded: List[str] = []
    for match in TEXT_STRING_PATTERN.finditer(stream):
        raw = match.group(0)[1:-1]
        decoded.append(_decode_pdf_string(raw))
    return decoded


def _decode_pdf_string(raw: bytes) -> str:
    text = raw.decode("latin-1", errors="ignore")
    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    return text


def normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_records(
    pages: Sequence[Sequence[str]],
    course_map: dict[str, str],
) -> List[StudentRecord]:
    records: List[StudentRecord] = []
    for page_idx, lines in enumerate(pages):
        records.extend(parse_page_records(pages, page_idx, course_map))
    return records


def parse_page_records(
    pages: Sequence[Sequence[str]],
    page_idx: int,
    course_map: dict[str, str],
) -> List[StudentRecord]:
    records: List[StudentRecord] = []
    lines = pages[page_idx]
    usn_indexes = [idx for idx, line in enumerate(lines) if line == "USN"]
    for usn_idx in usn_indexes:
        usn = lines[usn_idx + 1] if usn_idx + 1 < len(lines) else ""
        if not USN_PATTERN.fullmatch(usn):
            continue
        course_codes = find_course_codes(lines, usn_idx)
        roll = find_roll_number(lines, usn_idx)
        name = find_student_name(lines, usn_idx)
        totals = find_totals(lines, usn_idx)
        if course_codes and len(totals) < len(course_codes):
            missing = len(course_codes) - len(totals)
            totals.extend(find_continuation_totals(pages, page_idx + 1, missing))
        if not roll or not name or not totals or not course_codes:
            continue
        marks = map_marks(course_codes, totals, course_map)
        records.append(StudentRecord(usn=usn, name=name, roll=roll, marks=marks))
    return records


def find_roll_number(lines: Sequence[str], usn_idx: int) -> str:
    for idx in range(usn_idx - 1, -1, -1):
        candidate = lines[idx]
        if ROLL_PATTERN.fullmatch(candidate):
            return candidate
    return ""


def find_student_name(lines: Sequence[str], usn_idx: int) -> str:
    try:
        pr_idx = lines.index("Pr", usn_idx)
    except ValueError:
        return ""
    for idx in range(pr_idx - 1, usn_idx, -1):
        candidate = lines[idx]
        if candidate in PAGE_LABELS or candidate == "Th":
            continue
        if NUMBER_PATTERN.fullmatch(candidate):
            continue
        if "+" in candidate:
            continue
        name = NAME_CLEAN_PATTERN.sub("", candidate).strip()
        if name:
            return name
    return ""


def find_totals(lines: Sequence[str], usn_idx: int) -> List[str]:
    result_idx = None
    for idx in range(usn_idx, len(lines)):
        if lines[idx].startswith("Result:"):
            result_idx = idx
            break
    if result_idx is None:
        return []
    total_idx = None
    for idx in range(result_idx, len(lines)):
        if lines[idx] == "Total":
            total_idx = idx
            break
    if total_idx is None:
        return []
    totals: List[str] = []
    for idx in range(total_idx + 1, len(lines)):
        line = lines[idx]
        if line.startswith("Class") or line.startswith("Max. Total"):
            break
        if line.startswith("Term Grade"):
            break
        if NUMBER_PATTERN.fullmatch(line):
            totals.append(line)
    return totals


def find_course_codes(lines: Sequence[str], usn_idx: int) -> List[str]:
    try:
        sub_idx = lines.index("Sub", max(0, usn_idx - 200), usn_idx)
    except ValueError:
        return []
    raw_codes = lines[sub_idx + 1 : usn_idx]
    return combine_course_code_lines(raw_codes)


def combine_course_code_lines(lines: Sequence[str]) -> List[str]:
    combined: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line in {"Total", "Sub"}:
            idx += 1
            continue
        if is_course_code_fragment(line):
            if line.endswith("-") and idx + 1 < len(lines):
                combined.append(normalize_course_code(f"{line}{lines[idx + 1]}"))
                idx += 2
                continue
            combined.append(normalize_course_code(line))
        idx += 1
    return combined


def map_marks(
    course_codes: Sequence[str],
    totals: Sequence[str],
    course_map: dict[str, str],
) -> dict[str, str]:
    marks: dict[str, str] = {}
    for code, total in zip(course_codes, totals):
        course_name = course_map.get(code, code)
        marks[course_name] = total
    return marks


def parse_course_catalog(lines: Sequence[str]) -> tuple[dict[str, str], List[str]]:
    course_map: dict[str, str] = {}
    course_order: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.isdigit() and idx + 2 < len(lines):
            code_line = lines[idx + 1]
            name_line = lines[idx + 2]
            if is_course_code_fragment(code_line):
                if code_line.endswith("-") and idx + 3 < len(lines):
                    code = normalize_course_code(f"{code_line}{lines[idx + 2]}")
                    name = lines[idx + 3]
                    idx += 4
                else:
                    code = normalize_course_code(code_line)
                    name = name_line
                    idx += 3
                course_map[code] = name
                course_order.append(name)
                continue
        idx += 1
    return course_map, course_order


def find_continuation_totals(
    pages: Sequence[Sequence[str]], start_page_idx: int, needed: int
) -> List[str]:
    totals: List[str] = []
    for page_idx in range(start_page_idx, len(pages)):
        lines = pages[page_idx]
        if "USN" in lines or "Sub" in lines:
            break
        totals.extend(extract_totals_from_lines(lines))
        if len(totals) >= needed:
            return totals[:needed]
    return totals


def extract_totals_from_lines(lines: Sequence[str]) -> List[str]:
    totals: List[str] = []
    total_indexes = [idx for idx, line in enumerate(lines) if line == "Total"]
    for total_idx in total_indexes:
        for idx in range(total_idx + 1, len(lines)):
            line = lines[idx]
            if line == "Total":
                break
            if line.startswith("Class") or line.startswith("Max. Total"):
                break
            if line.startswith("Printed Date") or line.startswith("Page"):
                break
            if NUMBER_PATTERN.fullmatch(line):
                totals.append(line)
    return totals


def normalize_course_code(code: str) -> str:
    return code.replace(" ", "").strip()


def is_course_code_fragment(value: str) -> bool:
    normalized = normalize_course_code(value)
    return "-" in normalized and bool(COURSE_CODE_PATTERN.fullmatch(normalized))


def build_headers(course_order: Sequence[str]) -> List[str]:
    headers = ["USN", "Name", "Roll"] + list(course_order)
    if course_order:
        headers.append("Total")
    return headers


def write_csv(
    output_path: Path, records: Sequence[StudentRecord], course_order: Sequence[str]
) -> None:
    headers = build_headers(course_order)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for record in records:
            row = [record.usn, record.name, record.roll]
            if headers[-1:] == ["Total"]:
                subject_names = headers[3:-1]
                subjects = [record.marks.get(name, "") for name in subject_names]
                row.extend(subjects)
                row.append(f"{sum(to_float(value) for value in subjects):.2f}")
            else:
                row.extend(record.marks.values())
            writer.writerow(row)


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract student marks to CSV.")
    parser.add_argument("--input", default="PDF.pdf", type=Path, help="PDF file path")
    parser.add_argument(
        "--output", default="students.csv", type=Path, help="CSV output path"
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.input)
    course_map, course_order = parse_course_catalog(pages[0])
    records = parse_records(pages, course_map)
    if not records:
        raise SystemExit("No student records found. Check PDF format.")

    write_csv(args.output, records, course_order)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
