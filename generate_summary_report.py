"""Generate overall summary and top performers PDF from CSV files.

Usage:
  python generate_summary_report.py
  python generate_summary_report.py --students students.csv --details details_sheet.csv --output result_summary.pdf --max-total 850
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass
class StudentRecord:
    roll: str
    usn: str
    name: str
    gender: str
    total: float
    percentage: float


@dataclass
class SummaryRow:
    label: str
    students_appeared: int
    distinction: int
    first_class: int
    second_class: int
    pass_class: int
    total_passed: int
    total_failed: int
    pass_percentage: float


def normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_column(fieldnames: Sequence[str], aliases: Iterable[str]) -> str:
    normalized = {normalize(name): name for name in fieldnames}
    for alias in aliases:
        key = normalize(alias)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find column. Tried aliases: {', '.join(aliases)}")


def to_float(value: str) -> float:
    try:
        return float(value.strip()) if value else 0.0
    except ValueError:
        return 0.0


def normalize_gender(value: str) -> str:
    v = value.strip().lower()
    if v in {"male", "m", "boy", "boys"}:
        return "boys"
    if v in {"female", "f", "girl", "girls"}:
        return "girls"
    return "other"


def read_students(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_joined_records(
    students_path: Path, details_path: Path, max_total: float
) -> tuple[list[StudentRecord], int]:
    student_rows = read_students(students_path)
    detail_rows = read_students(details_path)

    if not student_rows:
        raise ValueError("students.csv is empty.")
    if not detail_rows:
        raise ValueError("details_sheet.csv is empty.")

    student_fields = list(student_rows[0].keys())
    details_fields = list(detail_rows[0].keys())

    student_roll_col = find_column(student_fields, ["Roll"])
    student_usn_col = find_column(student_fields, ["USN", "Registration No", "Regisration No"])
    student_name_col = find_column(student_fields, ["Name", "Student Name"])
    student_total_col = find_column(student_fields, ["Total"])

    details_usn_col = find_column(details_fields, ["Regisration No", "Registration No", "USN"])
    details_gender_col = find_column(details_fields, ["Gender"])

    details_map: dict[str, dict[str, str]] = {}
    for row in detail_rows:
        usn = (row.get(details_usn_col) or "").strip().upper()
        if usn and usn not in details_map:
            details_map[usn] = row

    records: list[StudentRecord] = []
    unmatched = 0
    for row in student_rows:
        usn = (row.get(student_usn_col) or "").strip().upper()
        if not usn:
            continue
        details = details_map.get(usn)
        if details is None:
            unmatched += 1
            gender = "other"
        else:
            gender = normalize_gender(details.get(details_gender_col, ""))

        total = to_float(row.get(student_total_col, ""))
        percentage = (total / max_total) * 100 if max_total > 0 else 0.0
        records.append(
            StudentRecord(
                roll=(row.get(student_roll_col) or "").strip(),
                usn=usn,
                name=(row.get(student_name_col) or "").strip(),
                gender=gender,
                total=total,
                percentage=percentage,
            )
        )
    return records, unmatched


def classify_pass_band(percentage: float) -> str:
    if percentage >= 85:
        return "distinction"
    if percentage >= 60:
        return "first"
    if percentage >= 50:
        return "second"
    if percentage >= 40:
        return "pass"
    return "failed"


def compute_summary(label: str, records: Sequence[StudentRecord]) -> SummaryRow:
    distinction = 0
    first_class = 0
    second_class = 0
    pass_class = 0
    failed = 0

    for record in records:
        band = classify_pass_band(record.percentage)
        if band == "distinction":
            distinction += 1
        elif band == "first":
            first_class += 1
        elif band == "second":
            second_class += 1
        elif band == "pass":
            pass_class += 1
        else:
            failed += 1

    appeared = len(records)
    total_passed = distinction + first_class + second_class + pass_class
    pass_percentage = (total_passed / appeared) * 100 if appeared else 0.0

    return SummaryRow(
        label=label,
        students_appeared=appeared,
        distinction=distinction,
        first_class=first_class,
        second_class=second_class,
        pass_class=pass_class,
        total_passed=total_passed,
        total_failed=failed,
        pass_percentage=pass_percentage,
    )


def summary_table_data(rows: Sequence[SummaryRow]) -> list[list[str]]:
    header = [
        "Category",
        "Students<br/>Appeared",
        "Passed with Distinction<br/>(85%-100%)",
        "Passed with First Class<br/>(60%-85%)",
        "Passed with Second Class<br/>(50%-60%)",
        "Passed with Pass Class<br/>(40%-&lt;50%)",
        "Total<br/>Passed",
        "Total<br/>Failed",
        "Pass<br/>Percentage",
    ]
    table = [header]
    for row in rows:
        table.append(
            [
                row.label,
                str(row.students_appeared),
                str(row.distinction),
                str(row.first_class),
                str(row.second_class),
                str(row.pass_class),
                str(row.total_passed),
                str(row.total_failed),
                f"{row.pass_percentage:.2f}%",
            ]
        )
    return table


def top_performers_table_data(records: Sequence[StudentRecord]) -> list[list[str]]:
    header = ["Rank", "Name", "Registration No.", "Marks Obtained", "Percentage"]
    ranked = sorted(records, key=lambda r: (-r.total, -r.percentage, r.usn))
    top3 = ranked[:3]
    ranks = ["I", "II", "III"]
    table = [header]
    for idx in range(3):
        if idx < len(top3):
            record = top3[idx]
            table.append(
                [
                    ranks[idx],
                    record.name,
                    record.usn,
                    f"{record.total:.2f}",
                    f"{record.percentage:.2f}%",
                ]
            )
        else:
            table.append([ranks[idx], "", "", "", ""])
    return table


def build_pdf(
    output_path: Path,
    summary_rows: Sequence[SummaryRow],
    top_records: Sequence[StudentRecord],
) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
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
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading3"],
        textColor=colors.HexColor("#0b67c2"),
        fontSize=13,
        leading=16,
        spaceAfter=6,
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=11,
        alignment=1,
    )

    elements = [
        Paragraph('I. <u>Overall Summary</u>', heading_style),
    ]

    summary_data = summary_table_data(summary_rows)
    summary_data[0] = [Paragraph(cell, table_header_style) for cell in summary_data[0]]
    summary_col_widths = [68, 82, 92, 92, 96, 92, 70, 70, 95]
    summary_row_heights = [46] + [30] * len(summary_rows)
    summary_table = Table(
        summary_data,
        colWidths=summary_col_widths,
        rowHeights=summary_row_heights,
        repeatRows=1,
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph('II. <u>Top Performers</u>', section_style))

    top_data = top_performers_table_data(top_records)
    top_data[0] = [Paragraph(cell.replace(" ", "<br/>", 1), table_header_style) if cell in {"Marks Obtained", "Registration No."} else Paragraph(cell, table_header_style) for cell in top_data[0]]
    top_col_widths = [55, 180, 180, 120, 110]
    top_row_heights = [36, 30, 30, 30]
    top_table = Table(
        top_data,
        colWidths=top_col_widths,
        rowHeights=top_row_heights,
        repeatRows=1,
    )
    top_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(top_table)

    doc.build(elements)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Overall Summary and Top Performers PDF from students/details CSV files."
    )
    parser.add_argument("--students", type=Path, default=Path("students.csv"))
    parser.add_argument("--details", type=Path, default=Path("details_sheet.csv"))
    parser.add_argument("--output", type=Path, default=Path("summary_report.pdf"))
    parser.add_argument("--max-total", type=float, default=850.0)
    args = parser.parse_args()

    records, unmatched = build_joined_records(
        students_path=args.students,
        details_path=args.details,
        max_total=args.max_total,
    )
    if not records:
        raise SystemExit("No student records found after joining both CSV files.")

    boys = [r for r in records if r.gender == "boys"]
    girls = [r for r in records if r.gender == "girls"]

    summary_rows = [
        compute_summary("Boys", boys),
        compute_summary("Girls", girls),
        compute_summary("Total", records),
    ]
    build_pdf(args.output, summary_rows, records)

    print(f"Joined students: {len(records)}")
    print(f"Unmatched student USNs in details_sheet.csv: {unmatched}")
    print(f"Wrote PDF report to {args.output}")


if __name__ == "__main__":
    main()
