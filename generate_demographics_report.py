"""Generate Performance Analysis by Demographics PDF.

Uses:
- students.csv
- details_sheet.csv
- categories.txt

Output:
- demographics_report.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


GROUPS = ["General", "EWS (Out of General)", "SC", "ST", "OBC", "TOTAL"]
GENDERS = ["male", "female", "trans"]
ROW_LABELS = ["Total", "PWD", "MM", "OM"]


@dataclass
class CategoryConfig:
    category_i: set[str]
    category_ii_a: set[str]
    category_ii_b: set[str]
    category_iii_a: set[str]
    category_iii_b: set[str]
    general: set[str]
    sc: set[str]
    st: set[str]
    obc: set[str]
    ews: set[str]


@dataclass
class Student:
    usn: str
    name: str
    gender: str
    caste: str
    discipline: str
    group: str
    is_pwd: bool
    is_mm: bool
    is_om: bool
    percentage: float


def normalize(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def normalize_set(values: Iterable[object]) -> set[str]:
    result: set[str] = set()
    for value in values:
        if isinstance(value, str):
            nv = normalize(value)
            if nv:
                result.add(nv)
    return result


def find_column(fieldnames: Sequence[str], aliases: Sequence[str]) -> str:
    normalized = {normalize(name): name for name in fieldnames}
    for alias in aliases:
        key = normalize(alias)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Missing required column. Tried: {', '.join(aliases)}")


def to_float(value: str) -> float:
    try:
        return float(value.strip()) if value else 0.0
    except ValueError:
        return 0.0


def is_truthy(value: str) -> bool:
    return normalize(value) in {
        "1",
        "y",
        "yes",
        "true",
        "pwbd",
        "pwd",
        "divyang",
        "benchmarkdisability",
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_categories(path: Path) -> CategoryConfig:
    namespace: dict[str, object] = {}
    exec(path.read_text(encoding="utf-8"), namespace)

    def load_set(name: str) -> set[str]:
        value = namespace.get(name, set())
        if isinstance(value, (set, list, tuple)):
            return normalize_set(value)
        return set()

    return CategoryConfig(
        category_i=load_set("CATEGORY_I"),
        category_ii_a=load_set("CATEGORY_II_A"),
        category_ii_b=load_set("CATEGORY_II_B"),
        category_iii_a=load_set("CATEGORY_III_A"),
        category_iii_b=load_set("CATEGORY_III_B"),
        general=load_set("GENERAL_LIST"),
        sc=load_set("SC_LIST"),
        st=load_set("ST_LIST"),
        obc=load_set("OBC_LIST"),
        ews=load_set("EWS_LIST"),
    )


def normalize_gender(raw_gender: str) -> str:
    g = normalize(raw_gender)
    if g in {"male", "m", "boy", "boys"}:
        return "male"
    if g in {"female", "f", "girl", "girls"}:
        return "female"
    if "trans" in g:
        return "trans"
    return "trans"


def classify_group(caste: str, config: CategoryConfig) -> str:
    c = normalize(caste)
    if not c:
        return "General"

    if c in config.sc or c in {"sc", "scheduledcastes", "scheduledcaste"}:
        return "SC"
    if c in config.st or c in {"st", "scheduledtribes", "scheduledtribe"}:
        return "ST"
    if c in config.ews or "ews" in c:
        return "EWS (Out of General)"
    if c in config.general:
        return "General"
    if c in config.obc:
        return "OBC"

    if (
        c in config.category_i
        or c in config.category_ii_a
        or c in config.category_ii_b
        or c in config.category_iii_a
        or c in config.category_iii_b
    ):
        return "OBC"

    if "muslim" in c:
        return "OBC"
    return "General"


def minority_flags(caste: str, config: CategoryConfig) -> tuple[bool, bool]:
    c = normalize(caste)
    is_mm = "muslim" in c or c in config.category_ii_b
    if is_mm:
        return True, False

    other_minority_keywords = ("christian", "jain", "buddhist", "sikh", "parsi")
    is_om = any(keyword in c for keyword in other_minority_keywords)
    return False, is_om


def infer_pwd_columns(fieldnames: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for name in fieldnames:
        n = normalize(name)
        if any(token in n for token in ("pwd", "pwbd", "disability", "benchmark")):
            columns.append(name)
    return columns


def build_students(
    students_csv: Path, details_csv: Path, categories_txt: Path, max_total: float
) -> tuple[list[Student], int]:
    students_rows = read_csv(students_csv)
    details_rows = read_csv(details_csv)
    config = load_categories(categories_txt)

    if not students_rows:
        raise ValueError("students.csv is empty.")
    if not details_rows:
        raise ValueError("details_sheet.csv is empty.")

    students_fields = list(students_rows[0].keys())
    details_fields = list(details_rows[0].keys())

    usn_students_col = find_column(students_fields, ["USN", "Registration No", "Regisration No"])
    name_students_col = find_column(students_fields, ["Name", "Student Name"])
    total_students_col = find_column(students_fields, ["Total"])

    usn_details_col = find_column(details_fields, ["Regisration No", "Registration No", "USN"])
    gender_col = find_column(details_fields, ["Gender"])
    caste_col = find_column(details_fields, ["Caste"])
    discipline_col = find_column(details_fields, ["Discipline", "Program"])
    pwd_columns = infer_pwd_columns(details_fields)

    details_map: dict[str, dict[str, str]] = {}
    for row in details_rows:
        usn = (row.get(usn_details_col) or "").strip().upper()
        if usn and usn not in details_map:
            details_map[usn] = row

    result: list[Student] = []
    unmatched = 0
    for row in students_rows:
        usn = (row.get(usn_students_col) or "").strip().upper()
        if not usn:
            continue

        detail = details_map.get(usn)
        if detail is None:
            unmatched += 1
            gender = "trans"
            caste = ""
            discipline = ""
            is_pwd = False
        else:
            gender = normalize_gender(detail.get(gender_col, ""))
            caste = (detail.get(caste_col) or "").strip()
            discipline = (detail.get(discipline_col) or "").strip()
            is_pwd = any(is_truthy(detail.get(col, "")) for col in pwd_columns)

        group = classify_group(caste, config)
        is_mm, is_om = minority_flags(caste, config)
        percentage = (to_float(row.get(total_students_col, "")) / max_total) * 100 if max_total > 0 else 0.0

        result.append(
            Student(
                usn=usn,
                name=(row.get(name_students_col) or "").strip(),
                gender=gender,
                caste=caste,
                discipline=discipline,
                group=group,
                is_pwd=is_pwd,
                is_mm=is_mm,
                is_om=is_om,
                percentage=percentage,
            )
        )

    return result, unmatched


def empty_grid() -> dict[str, dict[str, int]]:
    return {group: {gender: 0 for gender in GENDERS} for group in GROUPS}


def count_by_group_gender(students: Sequence[Student]) -> dict[str, dict[str, int]]:
    grid = empty_grid()
    for student in students:
        group = student.group if student.group in GROUPS else "General"
        gender = student.gender if student.gender in GENDERS else "trans"
        grid[group][gender] += 1
        grid["TOTAL"][gender] += 1
    return grid


def section_counts(students: Sequence[Student]) -> dict[str, dict[str, dict[str, int]]]:
    rows: dict[str, Sequence[Student]] = {
        "Total": list(students),
        "PWD": [s for s in students if s.is_pwd],
        "MM": [s for s in students if s.is_mm],
        "OM": [s for s in students if s.is_om],
    }
    return {label: count_by_group_gender(items) for label, items in rows.items()}


def fill_data_rows(
    data: list[list[str]],
    row_start: int,
    counts: dict[str, dict[str, dict[str, int]]],
) -> None:
    for offset, label in enumerate(ROW_LABELS):
        row_index = row_start + offset
        data[row_index][2] = label
        col = 3
        for group in GROUPS:
            for gender in GENDERS:
                data[row_index][col] = str(counts[label][group][gender])
                col += 1


def choose_discipline(students: Sequence[Student]) -> str:
    disciplines = [s.discipline for s in students if s.discipline]
    if not disciplines:
        return "All"
    common, _ = Counter(disciplines).most_common(1)[0]
    return common if len(common) <= 24 else "All"


def build_demographics_table(
    appeared: dict[str, dict[str, dict[str, int]]],
    passed: dict[str, dict[str, dict[str, int]]],
    passed_60: dict[str, dict[str, dict[str, int]]],
    discipline: str,
) -> Table:
    total_cols = 21
    total_rows = 17
    data = [["" for _ in range(total_cols)] for _ in range(total_rows)]

    data[0][0] = "S.\nN O"
    data[0][1] = "Discipline"
    data[0][2] = "Category"
    data[0][3] = "Total Number of Students Appeared:"

    group_starts = [3, 6, 9, 12, 15, 18]
    for group_name, start in zip(GROUPS, group_starts):
        data[1][start] = group_name
        data[2][start] = "Male"
        data[2][start + 1] = "Female"
        data[2][start + 2] = "Trans\nGender"

    data[3][0] = "1"
    data[3][1] = discipline

    fill_data_rows(data, 3, appeared)

    data[7][2] = "Total Number of Students Passed/Awarded Degree:"
    fill_data_rows(data, 8, passed)

    data[12][2] = "Out of Total, Number of Students Passed with 60% or above:"
    fill_data_rows(data, 13, passed_60)

    col_widths = [14 * mm, 20 * mm, 17 * mm] + [12 * mm] * 18
    row_heights = [
        13 * mm,  # heading row
        12 * mm,  # group row
        16 * mm,  # gender row
        9 * mm,
        9 * mm,
        9 * mm,
        9 * mm,
        10 * mm,  # section heading
        9 * mm,
        9 * mm,
        9 * mm,
        9 * mm,
        10 * mm,  # section heading
        9 * mm,
        9 * mm,
        9 * mm,
        9 * mm,
    ]
    table = Table(data, colWidths=col_widths, rowHeights=row_heights)

    style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (2, 2), "Helvetica-Bold"),
            ("FONTNAME", (3, 0), (20, 2), "Helvetica-Bold"),
            ("FONTNAME", (2, 3), (2, 6), "Helvetica-Bold"),
            ("FONTNAME", (2, 8), (2, 11), "Helvetica-Bold"),
            ("FONTNAME", (2, 13), (2, 16), "Helvetica-Bold"),
            ("FONTNAME", (3, 0), (20, 0), "Helvetica-Bold"),
            ("FONTNAME", (2, 7), (20, 7), "Helvetica-Bold"),
            ("FONTNAME", (2, 12), (20, 12), "Helvetica-Bold"),
            ("SPAN", (3, 0), (20, 0)),
            ("SPAN", (0, 0), (0, 2)),
            ("SPAN", (1, 0), (1, 2)),
            ("SPAN", (2, 0), (2, 2)),
            ("SPAN", (3, 1), (5, 1)),
            ("SPAN", (6, 1), (8, 1)),
            ("SPAN", (9, 1), (11, 1)),
            ("SPAN", (12, 1), (14, 1)),
            ("SPAN", (15, 1), (17, 1)),
            ("SPAN", (18, 1), (20, 1)),
            ("SPAN", (0, 3), (0, 16)),
            ("SPAN", (1, 3), (1, 16)),
            ("SPAN", (2, 7), (20, 7)),
            ("SPAN", (2, 12), (20, 12)),
            ("TEXTANGLE", (0, 0), (2, 2), 90),
            ("TEXTANGLE", (3, 2), (20, 2), 90),
            ("TEXTANGLE", (0, 3), (1, 16), 90),
            ("BACKGROUND", (0, 0), (20, 2), colors.whitesmoke),
            ("BACKGROUND", (2, 7), (20, 7), colors.whitesmoke),
            ("BACKGROUND", (2, 12), (20, 12), colors.whitesmoke),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )
    table.setStyle(style)
    return table


def build_pdf(output_path: Path, table: Table) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0b67c2"),
        fontSize=14,
        leading=16,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    footnote_style = ParagraphStyle(
        "Footnote",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )

    footnote = (
        "<b>*PwBD</b>- Persons with Benchmark Disabilities <b>(Out of Total)</b>, "
        "<b>MM</b>-Muslim Minority<b>(Out of Total)</b>, "
        "<b>OM</b>-Other Minority<b>(Out of Total)</b>"
    )

    elements = [
        Paragraph('IV. <u>Performance Analysis by Demographics</u>', title_style),
        Spacer(1, 2),
        table,
        Spacer(1, 4),
        Paragraph(footnote, footnote_style),
    ]
    doc.build(elements)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demographics analysis PDF using students.csv, details_sheet.csv, and categories.txt."
    )
    parser.add_argument("--students", type=Path, default=Path("students.csv"))
    parser.add_argument("--details", type=Path, default=Path("details_sheet.csv"))
    parser.add_argument("--categories", type=Path, default=Path("categories.txt"))
    parser.add_argument("--output", type=Path, default=Path("demographics_report.pdf"))
    parser.add_argument("--max-total", type=float, default=850.0)
    args = parser.parse_args()

    students, unmatched = build_students(
        students_csv=args.students,
        details_csv=args.details,
        categories_txt=args.categories,
        max_total=args.max_total,
    )
    if not students:
        raise SystemExit("No joined students found.")

    appeared_counts = section_counts(students)
    passed_counts = section_counts([s for s in students if s.percentage >= 40.0])
    passed_60_counts = section_counts([s for s in students if s.percentage >= 60.0])
    discipline = choose_discipline(students)

    table = build_demographics_table(
        appeared=appeared_counts,
        passed=passed_counts,
        passed_60=passed_60_counts,
        discipline=discipline,
    )
    build_pdf(args.output, table)

    print(f"Joined students: {len(students)}")
    print(f"Unmatched students in details_sheet.csv: {unmatched}")
    print(f"Wrote demographics report to {args.output}")


if __name__ == "__main__":
    main()
