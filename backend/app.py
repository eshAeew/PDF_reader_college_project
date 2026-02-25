from __future__ import annotations

import csv
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from PyPDF2 import PdfMerger

# Import existing project logic from root directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import main as pdf_reader  # noqa: E402
import generate_centum_achievers_report as centum_report  # noqa: E402
import generate_demographics_report as demographics_report  # noqa: E402
import generate_summary_report as summary_report  # noqa: E402


app = Flask(__name__)
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{(ROOT_DIR / 'analysis.db').as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Keep JSON field order as generated from CSV/header parsing.
app.json.sort_keys = False
db = SQLAlchemy(app)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Analysis(db.Model):
    __tablename__ = "analysis"
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(64), nullable=False, default="created")
    error_message = db.Column(db.Text, nullable=True)
    students_json = db.Column(db.Text, nullable=True)
    summary_json = db.Column(db.Text, nullable=True)
    demographics_json = db.Column(db.Text, nullable=True)
    centum_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class StoredFile(db.Model):
    __tablename__ = "stored_file"
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("analysis.id"), nullable=False)
    role = db.Column(db.String(64), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    mime_type = db.Column(db.String(128), nullable=False)
    content = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (db.UniqueConstraint("analysis_id", "role", name="uq_analysis_role"),)


def upsert_file(
    analysis_id: int, role: str, filename: str, mime_type: str, content: bytes
) -> StoredFile:
    existing = StoredFile.query.filter_by(analysis_id=analysis_id, role=role).first()
    if existing:
        existing.filename = filename
        existing.mime_type = mime_type
        existing.content = content
        existing.created_at = utc_now()
        return existing
    item = StoredFile(
        analysis_id=analysis_id,
        role=role,
        filename=filename,
        mime_type=mime_type,
        content=content,
    )
    db.session.add(item)
    return item


def decode_csv_bytes(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig", errors="ignore")
    return list(csv.DictReader(io.StringIO(text)))


def analysis_payload(analysis: Analysis) -> dict[str, Any]:
    files = StoredFile.query.filter_by(analysis_id=analysis.id).all()
    files_map = {f.role: {"id": f.id, "filename": f.filename, "mime_type": f.mime_type} for f in files}
    return {
        "id": analysis.id,
        "status": analysis.status,
        "error_message": analysis.error_message,
        "students": json.loads(analysis.students_json) if analysis.students_json else None,
        "summary": json.loads(analysis.summary_json) if analysis.summary_json else None,
        "demographics": json.loads(analysis.demographics_json)
        if analysis.demographics_json
        else None,
        "centum": json.loads(analysis.centum_json) if analysis.centum_json else None,
        "files": files_map,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "updated_at": analysis.updated_at.isoformat() if analysis.updated_at else None,
    }


def require_analysis(analysis_id: int) -> Analysis:
    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        raise ValueError("Analysis not found.")
    return analysis


def generate_students_csv_from_pdf(pdf_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pdf_path = tmp_dir / "input.pdf"
        out_csv = tmp_dir / "students.csv"
        pdf_path.write_bytes(pdf_bytes)

        pages = pdf_reader.extract_pdf_pages(pdf_path)
        course_map, course_order = pdf_reader.parse_course_catalog(pages[0])
        records = pdf_reader.parse_records(pages, course_map)
        if not records:
            raise ValueError("No student records found in uploaded PDF.")
        pdf_reader.write_csv(out_csv, records, course_order)
        return out_csv.read_bytes()


def generate_analysis_reports(
    students_csv: bytes, details_csv: bytes, exam_pdf: bytes
) -> tuple[dict[str, bytes], dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        students_path = tmp_dir / "students.csv"
        details_path = tmp_dir / "details_sheet.csv"
        exam_pdf_path = tmp_dir / "PDF.pdf"
        categories_path = tmp_dir / "categories.txt"

        students_path.write_bytes(students_csv)
        details_path.write_bytes(details_csv)
        exam_pdf_path.write_bytes(exam_pdf)
        categories_path.write_bytes((ROOT_DIR / "categories.txt").read_bytes())

        # Summary report
        summary_pdf_path = tmp_dir / "summary_report.pdf"
        records, summary_unmatched = summary_report.build_joined_records(
            students_path=students_path,
            details_path=details_path,
            max_total=850.0,
        )
        boys = [r for r in records if r.gender == "boys"]
        girls = [r for r in records if r.gender == "girls"]
        summary_rows = [
            summary_report.compute_summary("Boys", boys),
            summary_report.compute_summary("Girls", girls),
            summary_report.compute_summary("Total", records),
        ]
        summary_report.build_pdf(summary_pdf_path, summary_rows, records)

        ranked = sorted(records, key=lambda r: (-r.total, -r.percentage, r.usn))[:3]
        summary_json = {
            "rows": [
                {
                    "label": row.label,
                    "students_appeared": row.students_appeared,
                    "distinction": row.distinction,
                    "first_class": row.first_class,
                    "second_class": row.second_class,
                    "pass_class": row.pass_class,
                    "total_passed": row.total_passed,
                    "total_failed": row.total_failed,
                    "pass_percentage": round(row.pass_percentage, 2),
                }
                for row in summary_rows
            ],
            "top_performers": [
                {
                    "rank": idx + 1,
                    "name": item.name,
                    "registration_no": item.usn,
                    "marks_obtained": round(item.total, 2),
                    "percentage": round(item.percentage, 2),
                }
                for idx, item in enumerate(ranked)
            ],
            "unmatched_details_rows": summary_unmatched,
        }

        # Demographics report
        demographics_pdf_path = tmp_dir / "demographics_report.pdf"
        students_demo, demo_unmatched = demographics_report.build_students(
            students_csv=students_path,
            details_csv=details_path,
            categories_txt=categories_path,
            max_total=850.0,
        )
        appeared_counts = demographics_report.section_counts(students_demo)
        passed_counts = demographics_report.section_counts(
            [student for student in students_demo if student.percentage >= 40.0]
        )
        passed_60_counts = demographics_report.section_counts(
            [student for student in students_demo if student.percentage >= 60.0]
        )
        discipline = demographics_report.choose_discipline(students_demo)
        table = demographics_report.build_demographics_table(
            appeared=appeared_counts,
            passed=passed_counts,
            passed_60=passed_60_counts,
            discipline=discipline,
        )
        demographics_report.build_pdf(demographics_pdf_path, table)
        demographics_json = {
            "discipline": discipline,
            "appeared": appeared_counts,
            "passed": passed_counts,
            "passed_60_or_above": passed_60_counts,
            "unmatched_details_rows": demo_unmatched,
        }

        # Centum achievers report
        centum_pdf_path = tmp_dir / "subject_wise_centum_achievers.pdf"
        centum_rows, total_students, total_subjects = centum_report.parse_centum_rows(
            students_csv=students_path,
            pdf_path=exam_pdf_path,
        )
        centum_report.build_pdf(centum_pdf_path, centum_rows)
        centum_json = {
            "total_students_scanned": total_students,
            "subject_columns_scanned": total_subjects,
            "entries": [
                {
                    "sl_no": row.sl_no,
                    "name": row.name,
                    "registration_no": row.registration_no,
                    "subject_name": row.subject_name,
                }
                for row in centum_rows
            ],
        }

        # Merge to single analysis.pdf
        analysis_pdf_path = tmp_dir / "analysis.pdf"
        merger = PdfMerger()
        for part in (summary_pdf_path, demographics_pdf_path, centum_pdf_path):
            merger.append(str(part))
        merger.write(str(analysis_pdf_path))
        merger.close()

        files = {
            "summary_pdf": summary_pdf_path.read_bytes(),
            "demographics_pdf": demographics_pdf_path.read_bytes(),
            "centum_pdf": centum_pdf_path.read_bytes(),
            "analysis_pdf": analysis_pdf_path.read_bytes(),
        }
        payload = {
            "summary": summary_json,
            "demographics": demographics_json,
            "centum": centum_json,
        }
        return files, payload


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify({"ok": True})


@app.route("/api/analyses", methods=["POST"])
def create_analysis() -> Any:
    analysis = Analysis(status="created")
    db.session.add(analysis)
    db.session.commit()
    return jsonify(analysis_payload(analysis)), 201


@app.route("/api/analyses/<int:analysis_id>", methods=["GET"])
def get_analysis(analysis_id: int) -> Any:
    try:
        analysis = require_analysis(analysis_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(analysis_payload(analysis))


@app.route("/api/analyses/<int:analysis_id>/upload-result-pdf", methods=["POST"])
def upload_result_pdf(analysis_id: int) -> Any:
    try:
        analysis = require_analysis(analysis_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    upload = request.files.get("file")
    if not upload:
        return jsonify({"error": "Missing file field 'file'."}), 400
    if not upload.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file."}), 400

    try:
        pdf_bytes = upload.read()
        students_csv_bytes = generate_students_csv_from_pdf(pdf_bytes)
        students_rows = decode_csv_bytes(students_csv_bytes)

        upsert_file(
            analysis_id=analysis.id,
            role="uploaded_pdf",
            filename=upload.filename,
            mime_type="application/pdf",
            content=pdf_bytes,
        )
        upsert_file(
            analysis_id=analysis.id,
            role="students_csv",
            filename="students.csv",
            mime_type="text/csv",
            content=students_csv_bytes,
        )

        analysis.status = "students_generated"
        analysis.error_message = None
        analysis.students_json = json.dumps(students_rows)
        db.session.commit()
        return jsonify(analysis_payload(analysis))
    except Exception as exc:  # pragma: no cover - API guard path
        analysis.status = "error"
        analysis.error_message = str(exc)
        db.session.commit()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyses/<int:analysis_id>/upload-details-csv", methods=["POST"])
def upload_details_csv(analysis_id: int) -> Any:
    try:
        analysis = require_analysis(analysis_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    upload = request.files.get("file")
    if not upload:
        return jsonify({"error": "Missing file field 'file'."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400

    students_file = StoredFile.query.filter_by(analysis_id=analysis.id, role="students_csv").first()
    uploaded_pdf_file = StoredFile.query.filter_by(analysis_id=analysis.id, role="uploaded_pdf").first()
    if not students_file or not uploaded_pdf_file:
        return (
            jsonify(
                {
                    "error": "Upload result PDF first so students.csv can be generated before details upload."
                }
            ),
            400,
        )

    try:
        details_bytes = upload.read()
        files, payload = generate_analysis_reports(
            students_csv=students_file.content,
            details_csv=details_bytes,
            exam_pdf=uploaded_pdf_file.content,
        )

        upsert_file(
            analysis_id=analysis.id,
            role="details_csv",
            filename=upload.filename,
            mime_type="text/csv",
            content=details_bytes,
        )
        upsert_file(
            analysis_id=analysis.id,
            role="summary_pdf",
            filename="summary_report.pdf",
            mime_type="application/pdf",
            content=files["summary_pdf"],
        )
        upsert_file(
            analysis_id=analysis.id,
            role="demographics_pdf",
            filename="demographics_report.pdf",
            mime_type="application/pdf",
            content=files["demographics_pdf"],
        )
        upsert_file(
            analysis_id=analysis.id,
            role="centum_pdf",
            filename="subject_wise_centum_achievers.pdf",
            mime_type="application/pdf",
            content=files["centum_pdf"],
        )
        upsert_file(
            analysis_id=analysis.id,
            role="analysis_pdf",
            filename="analysis.pdf",
            mime_type="application/pdf",
            content=files["analysis_pdf"],
        )

        analysis.status = "completed"
        analysis.error_message = None
        analysis.summary_json = json.dumps(payload["summary"])
        analysis.demographics_json = json.dumps(payload["demographics"])
        analysis.centum_json = json.dumps(payload["centum"])
        db.session.commit()
        return jsonify(analysis_payload(analysis))
    except Exception as exc:  # pragma: no cover - API guard path
        analysis.status = "error"
        analysis.error_message = str(exc)
        db.session.commit()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/files/<int:file_id>/download", methods=["GET"])
def download_file(file_id: int) -> Any:
    item = StoredFile.query.get(file_id)
    if not item:
        return jsonify({"error": "File not found."}), 404
    return send_file(
        io.BytesIO(item.content),
        mimetype=item.mime_type,
        as_attachment=True,
        download_name=item.filename,
    )


@app.route("/api/files/<int:file_id>/view", methods=["GET"])
def view_file(file_id: int) -> Any:
    item = StoredFile.query.get(file_id)
    if not item:
        return jsonify({"error": "File not found."}), 404
    return send_file(
        io.BytesIO(item.content),
        mimetype=item.mime_type,
        as_attachment=False,
        download_name=item.filename,
    )


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
