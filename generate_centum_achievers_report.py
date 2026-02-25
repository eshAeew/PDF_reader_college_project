"""Generate Subject-wise Centum Achievers PDF from students.csv.

Usage:
  python generate_centum_achievers_report.py
  python generate_centum_achievers_report.py --students students.csv --pdf PDF.pdf --output subject_wise_centum_achievers.pdf
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import main as pdf_parser


@dataclass
class CentumRow:
    sl_no: int
    name: str
    registration_no: str
    subject_name: str


def normalize_code(value: str) -> str:
    return "".join(value.upper().split())


def to_float(value: str) -> float:
    try:
        return float(value.strip()) if value else 0.0
    except ValueError:
        return 0.0


def is_close(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def subject_max_mark(subject_code: str) -> float:
    # Based on BU tabulation format in this project:
    # - practical/lab papers are out of 50
    # - SEC papers are out of 25
    # - all others are out of 100
    code = subject_code.strip().upper()
    if code.startswith("SEC-"):
        return 25.0
    if "-L" in code:
        return 50.0
    return 100.0


def load_course_name_map(pdf_path: Path) -> dict[str, str]:
    pages = pdf_parser.extract_pdf_pages(pdf_path)
    if not pages:
        return {}
    course_map, _ = pdf_parser.parse_course_catalog(pages[0])
    return {normalize_code(code): name for code, name in course_map.items()}


def parse_centum_rows(students_csv: Path, pdf_path: Path) -> tuple[list[CentumRow], int, int]:
    with students_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("students.csv is empty.")

    fields = list(rows[0].keys())
    for required in ("Name", "USN"):
        if required not in fields:
            raise ValueError(f"students.csv missing required column: {required}")

    subject_columns = [c for c in fields if c not in ("Roll", "USN", "Name", "Total")]
    course_name_map = load_course_name_map(pdf_path)
    result: list[CentumRow] = []
    sl_no = 1

    for subject in subject_columns:
        max_mark = subject_max_mark(subject)
        subject_name = course_name_map.get(normalize_code(subject), subject)
        for row in rows:
            mark_raw = (row.get(subject) or "").strip()
            if not mark_raw:
                continue
            mark = to_float(mark_raw)
            if is_close(mark, max_mark):
                result.append(
                    CentumRow(
                        sl_no=sl_no,
                        name=(row.get("Name") or "").strip(),
                        registration_no=(row.get("USN") or "").strip(),
                        subject_name=subject_name,
                    )
                )
                sl_no += 1

    return result, len(rows), len(subject_columns)


def build_pdf(output_path: Path, centum_rows: Sequence[CentumRow]) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0b67c2"),
        fontSize=16,
        leading=18,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    header_cell_style = ParagraphStyle(
        "HeaderCell",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=12,
        alignment=1,
    )
    body_cell_style = ParagraphStyle(
        "BodyCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=12,
        alignment=1,
    )

    elements = [Paragraph('V. <u>Subject-wise Centum Achievers</u>', heading_style), Spacer(1, 4)]

    header = [
        Paragraph("Sl. No", header_cell_style),
        Paragraph("Name", header_cell_style),
        Paragraph("Registration No.", header_cell_style),
        Paragraph("Subject Name", header_cell_style),
    ]
    table_data: list[list[object]] = [header]

    if centum_rows:
        for item in centum_rows:
            table_data.append(
                [
                    str(item.sl_no),
                    item.name,
                    item.registration_no,
                    Paragraph(item.subject_name, body_cell_style),
                ]
            )
    else:
        table_data.append(["-", "No centum achievers found", "-", "-"])

    col_widths = [20 * mm, 76 * mm, 50 * mm, 70 * mm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ]
        )
    )
    elements.append(table)

    doc.build(elements)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Subject-wise Centum Achievers PDF from students.csv."
    )
    parser.add_argument("--students", type=Path, default=Path("students.csv"))
    parser.add_argument("--pdf", type=Path, default=Path("PDF.pdf"))
    parser.add_argument(
        "--output", type=Path, default=Path("subject_wise_centum_achievers.pdf")
    )
    args = parser.parse_args()

    centum_rows, total_students, total_subjects = parse_centum_rows(args.students, args.pdf)
    build_pdf(args.output, centum_rows)

    print(f"Students scanned: {total_students}")
    print(f"Subject columns scanned: {total_subjects}")
    print(f"Centum achiever rows: {len(centum_rows)}")
    print(f"Wrote PDF report to {args.output}")


if __name__ == "__main__":
    main()
