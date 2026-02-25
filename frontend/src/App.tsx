import { Fragment, useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5000";

type FileMeta = {
  id: number;
  filename: string;
  mime_type: string;
};

type SummaryRow = {
  label: string;
  students_appeared: number;
  distinction: number;
  first_class: number;
  second_class: number;
  pass_class: number;
  total_passed: number;
  total_failed: number;
  pass_percentage: number;
};

type TopPerformer = {
  rank: number;
  name: string;
  registration_no: string;
  marks_obtained: number;
  percentage: number;
};

type SummaryData = {
  rows: SummaryRow[];
  top_performers: TopPerformer[];
  unmatched_details_rows: number;
};

type CountsGrid = Record<string, Record<string, number>>;
type DemographicSection = Record<string, CountsGrid>;

type DemographicsData = {
  discipline: string;
  appeared: DemographicSection;
  passed: DemographicSection;
  passed_60_or_above: DemographicSection;
  unmatched_details_rows: number;
};

type CentumEntry = {
  sl_no: number;
  name: string;
  registration_no: string;
  subject_name: string;
};

type CentumData = {
  total_students_scanned: number;
  subject_columns_scanned: number;
  entries: CentumEntry[];
};

type Analysis = {
  id: number;
  status: string;
  error_message: string | null;
  students: Array<Record<string, string>> | null;
  summary: SummaryData | null;
  demographics: DemographicsData | null;
  centum: CentumData | null;
  files: Record<string, FileMeta>;
};

const DEMOGRAPHIC_GROUPS = [
  "General",
  "EWS (Out of General)",
  "SC",
  "ST",
  "OBC",
  "TOTAL"
] as const;

const DEMOGRAPHIC_GENDERS = ["male", "female", "trans"] as const;
const DEMOGRAPHIC_ROWS = ["Total", "PWD", "MM", "OM"] as const;
const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];

async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const error = (data as { error?: string }).error ?? "Request failed";
    throw new Error(error);
  }
  return data as T;
}

function statusLabel(status: string): string {
  if (status === "created") return "Waiting For Files";
  if (status === "students_generated") return "students.csv Ready";
  if (status === "completed") return "Analysis Complete";
  if (status === "error") return "Error";
  return status;
}

function statusClass(status: string): string {
  if (status === "created") return "is-created";
  if (status === "students_generated") return "is-progress";
  if (status === "completed") return "is-complete";
  if (status === "error") return "is-error";
  return "is-created";
}

function metricValue(value: number | string): string {
  if (typeof value === "number") return value.toLocaleString();
  return value;
}

export default function App() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [detailsFile, setDetailsFile] = useState<File | null>(null);
  const [studentSearch, setStudentSearch] = useState("");
  const [pageSize, setPageSize] = useState(20);
  const [studentPage, setStudentPage] = useState(1);

  useEffect(() => {
    void createAnalysis();
  }, []);

  useEffect(() => {
    setStudentPage(1);
  }, [studentSearch, pageSize, analysis?.students]);

  async function createAnalysis(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      const created = await apiJson<Analysis>(`${API_BASE}/api/analyses`, {
        method: "POST"
      });
      setAnalysis(created);
      setPdfFile(null);
      setDetailsFile(null);
      setStudentSearch("");
      setStudentPage(1);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadResultPdf(): Promise<void> {
    if (!analysis || !pdfFile) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", pdfFile);
      const updated = await apiJson<Analysis>(
        `${API_BASE}/api/analyses/${analysis.id}/upload-result-pdf`,
        {
          method: "POST",
          body: form
        }
      );
      setAnalysis(updated);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadDetailsCsv(): Promise<void> {
    if (!analysis || !detailsFile) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", detailsFile);
      const updated = await apiJson<Analysis>(
        `${API_BASE}/api/analyses/${analysis.id}/upload-details-csv`,
        {
          method: "POST",
          body: form
        }
      );
      setAnalysis(updated);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const studentsHeaders = useMemo(() => {
    if (!analysis?.students || analysis.students.length === 0) return [];
    return Object.keys(analysis.students[0]);
  }, [analysis?.students]);

  const filteredStudents = useMemo(() => {
    if (!analysis?.students) return [];
    const q = studentSearch.trim().toLowerCase();
    if (!q) return analysis.students;
    return analysis.students.filter((row) =>
      Object.values(row).some((value) => value?.toLowerCase().includes(q))
    );
  }, [analysis?.students, studentSearch]);

  const totalStudentPages = Math.max(1, Math.ceil(filteredStudents.length / pageSize));
  const safeStudentPage = Math.min(studentPage, totalStudentPages);
  const pagedStudents = useMemo(() => {
    const start = (safeStudentPage - 1) * pageSize;
    return filteredStudents.slice(start, start + pageSize);
  }, [filteredStudents, pageSize, safeStudentPage]);

  const analysisPdfFile = analysis?.files.analysis_pdf;
  const studentsCsvFile = analysis?.files.students_csv;

  const summaryTotal = analysis?.summary?.rows.find((row) => row.label === "Total");
  const kpis = {
    totalStudents: analysis?.students?.length ?? 0,
    totalPassed: summaryTotal?.total_passed ?? 0,
    passPercentage: summaryTotal?.pass_percentage ?? 0,
    centumEntries: analysis?.centum?.entries.length ?? 0
  };

  const demographicsSections = analysis?.demographics
    ? [
        {
          title: "Total Number of Students Appeared:",
          data: analysis.demographics.appeared
        },
        {
          title: "Total Number of Students Passed/Awarded Degree:",
          data: analysis.demographics.passed
        },
        {
          title: "Out of Total, Number of Students Passed with 60% or above:",
          data: analysis.demographics.passed_60_or_above
        }
      ]
    : [];

  const demographicsBodyRows = demographicsSections.length * (1 + DEMOGRAPHIC_ROWS.length);

  const getDemoCount = (
    section: DemographicSection,
    rowLabel: (typeof DEMOGRAPHIC_ROWS)[number],
    group: (typeof DEMOGRAPHIC_GROUPS)[number],
    gender: (typeof DEMOGRAPHIC_GENDERS)[number]
  ): number => {
    return section?.[rowLabel]?.[group]?.[gender] ?? 0;
  };

  const canUploadDetails = Boolean(
    analysis && analysis.status !== "created" && analysis.status !== "error"
  );

  return (
    <div className="app-shell">
      <div className="visual-layer">
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
      </div>

      <main className="content">
        <header className="hero card">
          <div className="hero-copy">
            <p className="eyebrow">Academic Intelligence Suite</p>
            <h1>Student Result Analytics</h1>
            <p className="hero-subtitle">
              Upload your exam PDF and details sheet to generate one polished analysis report with
              summary insights, demographics, centum achievers, and downloadable outputs.
            </p>
            <div className="hero-actions">
              <button className="btn btn-ghost" onClick={() => void createAnalysis()} disabled={busy}>
                New Analysis
              </button>
              {analysisPdfFile && (
                <a
                  href={`${API_BASE}/api/files/${analysisPdfFile.id}/download`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <button className="btn btn-primary">Download analysis.pdf</button>
                </a>
              )}
            </div>
          </div>

          <div className="kpi-grid">
            <article className="kpi-card">
              <span className="kpi-title">Students</span>
              <strong>{metricValue(kpis.totalStudents)}</strong>
            </article>
            <article className="kpi-card">
              <span className="kpi-title">Total Passed</span>
              <strong>{metricValue(kpis.totalPassed)}</strong>
            </article>
            <article className="kpi-card">
              <span className="kpi-title">Pass %</span>
              <strong>{kpis.passPercentage.toFixed(2)}%</strong>
            </article>
            <article className="kpi-card">
              <span className="kpi-title">Centum Entries</span>
              <strong>{metricValue(kpis.centumEntries)}</strong>
            </article>
          </div>
        </header>

        {analysis && (
          <section className="status-strip card">
            <div className="status-item">
              <span>Analysis ID</span>
              <strong>{analysis.id}</strong>
            </div>
            <div className="status-item">
              <span>Status</span>
              <span className={`status-badge ${statusClass(analysis.status)}`}>
                {statusLabel(analysis.status)}
              </span>
            </div>
            <div className="status-item">
              <span>Backend</span>
              <strong>{API_BASE}</strong>
            </div>
          </section>
        )}

        {error && <div className="alert-error">{error}</div>}

        <section className="card workflow">
          <h2>Workflow</h2>
          <div className="workflow-grid">
            <article className={`workflow-step ${analysis?.status !== "created" ? "done" : ""}`}>
              <div className="step-index">01</div>
              <h3>Upload Result PDF</h3>
              <p>Upload exam PDF file and auto-generate students.csv.</p>
              <input
                className="file-input"
                type="file"
                accept=".pdf,application/pdf"
                onChange={(e) => setPdfFile(e.target.files?.[0] ?? null)}
              />
              <div className="file-caption">{pdfFile ? pdfFile.name : "No file selected"}</div>
              <button
                className="btn btn-primary"
                onClick={() => void uploadResultPdf()}
                disabled={!analysis || !pdfFile || busy}
              >
                {busy ? "Processing..." : "Generate students.csv"}
              </button>
            </article>

            <article className={`workflow-step ${analysis?.status === "completed" ? "done" : ""}`}>
              <div className="step-index">02</div>
              <h3>Upload details_sheet.csv</h3>
              <p>Generate summary, demographics, centum reports and merged analysis.pdf.</p>
              <input
                className="file-input"
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => setDetailsFile(e.target.files?.[0] ?? null)}
              />
              <div className="file-caption">{detailsFile ? detailsFile.name : "No file selected"}</div>
              <button
                className="btn btn-accent"
                onClick={() => void uploadDetailsCsv()}
                disabled={!analysis || !detailsFile || !canUploadDetails || busy}
              >
                {busy ? "Generating..." : "Generate Final Analysis"}
              </button>
            </article>
          </div>
        </section>

        {analysis?.students && (
          <section className="card">
            <div className="section-header">
              <div>
                <h2>students.csv Preview</h2>
                <p>
                  Showing {pagedStudents.length} rows on page {safeStudentPage} of {totalStudentPages}.
                  Total filtered rows: {filteredStudents.length}.
                </p>
              </div>
              <div className="section-actions">
                {studentsCsvFile && (
                  <a
                    href={`${API_BASE}/api/files/${studentsCsvFile.id}/download`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <button className="btn btn-outline">Download students.csv</button>
                  </a>
                )}
              </div>
            </div>

            <div className="student-toolbar">
              <input
                className="search-input"
                placeholder="Search by Roll, USN, Name, or marks..."
                value={studentSearch}
                onChange={(e) => setStudentSearch(e.target.value)}
              />
              <div className="toolbar-controls">
                <label>
                  Rows:
                  <select
                    value={pageSize}
                    onChange={(e) => setPageSize(Number(e.target.value))}
                    className="select-input"
                  >
                    {PAGE_SIZE_OPTIONS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="btn btn-ghost"
                  onClick={() => setStudentPage((p) => Math.max(1, p - 1))}
                  disabled={safeStudentPage <= 1}
                >
                  Prev
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => setStudentPage((p) => Math.min(totalStudentPages, p + 1))}
                  disabled={safeStudentPage >= totalStudentPages}
                >
                  Next
                </button>
              </div>
            </div>

            <div className="table-shell">
              <table className="data-table students-table">
                <thead>
                  <tr>
                    {studentsHeaders.map((header, index) => (
                      <th
                        key={header}
                        className={index < 3 ? `sticky-col sticky-head sticky-col-${index}` : ""}
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pagedStudents.length === 0 ? (
                    <tr>
                      <td colSpan={studentsHeaders.length}>No rows match your search.</td>
                    </tr>
                  ) : (
                    pagedStudents.map((row, idx) => (
                      <tr key={`${safeStudentPage}-${idx}`}>
                        {studentsHeaders.map((header, index) => (
                          <td
                            key={`${idx}-${header}`}
                            className={index < 3 ? `sticky-col sticky-col-${index}` : ""}
                          >
                            {row[header]}
                          </td>
                        ))}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {analysis?.summary && (
          <section className="card">
            <div className="section-header">
              <div>
                <h2>Summary Report Data</h2>
                <p>Automatically computed from uploaded students and details sheet.</p>
              </div>
            </div>

            <div className="table-shell">
              <table className="data-table compact">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Students Appeared</th>
                    <th>Distinction</th>
                    <th>First Class</th>
                    <th>Second Class</th>
                    <th>Pass Class</th>
                    <th>Total Passed</th>
                    <th>Total Failed</th>
                    <th>Pass %</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.summary.rows.map((row) => (
                    <tr key={row.label}>
                      <td>{row.label}</td>
                      <td>{row.students_appeared}</td>
                      <td>{row.distinction}</td>
                      <td>{row.first_class}</td>
                      <td>{row.second_class}</td>
                      <td>{row.pass_class}</td>
                      <td>{row.total_passed}</td>
                      <td>{row.total_failed}</td>
                      <td>{row.pass_percentage.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3 className="subheading">Top Performers</h3>
            <div className="table-shell">
              <table className="data-table compact">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Name</th>
                    <th>Registration No.</th>
                    <th>Marks Obtained</th>
                    <th>Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.summary.top_performers.map((item) => (
                    <tr key={item.rank}>
                      <td>{item.rank}</td>
                      <td>{item.name}</td>
                      <td>{item.registration_no}</td>
                      <td>{item.marks_obtained.toFixed(2)}</td>
                      <td>{item.percentage.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {analysis?.demographics && (
          <section className="card">
            <div className="section-header">
              <div>
                <h2>Demographics Data</h2>
                <p>
                  Discipline: <b>{analysis.demographics.discipline}</b>
                </p>
              </div>
            </div>

            <div className="table-shell">
              <table className="data-table demography-table">
                <thead>
                  <tr>
                    <th rowSpan={3}>S. No</th>
                    <th rowSpan={3}>Discipline</th>
                    <th rowSpan={3}>Category</th>
                    <th colSpan={18}>Total Number of Students Appeared</th>
                  </tr>
                  <tr>
                    {DEMOGRAPHIC_GROUPS.map((group) => (
                      <th key={group} colSpan={3}>
                        {group}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    {DEMOGRAPHIC_GROUPS.flatMap((group) =>
                      DEMOGRAPHIC_GENDERS.map((gender) => (
                        <th key={`${group}-${gender}`}>
                          {gender === "trans" ? "Trans Gender" : gender === "male" ? "Male" : "Female"}
                        </th>
                      ))
                    )}
                  </tr>
                </thead>
                <tbody>
                  {demographicsSections.map((section, sectionIdx) => (
                    <Fragment key={`section-block-${sectionIdx}`}>
                      <tr className="section-title-row">
                        {sectionIdx === 0 && (
                          <>
                            <td rowSpan={demographicsBodyRows}>1</td>
                            <td rowSpan={demographicsBodyRows}>
                              {analysis.demographics?.discipline || "GENERAL"}
                            </td>
                          </>
                        )}
                        <td colSpan={19} className="section-title-cell">
                          {section.title}
                        </td>
                      </tr>
                      {DEMOGRAPHIC_ROWS.map((rowLabel) => (
                        <tr key={`${sectionIdx}-${rowLabel}`}>
                          <td className="category-cell">{rowLabel}</td>
                          {DEMOGRAPHIC_GROUPS.flatMap((group) =>
                            DEMOGRAPHIC_GENDERS.map((gender) => (
                              <td key={`${sectionIdx}-${rowLabel}-${group}-${gender}`}>
                                {getDemoCount(section.data, rowLabel, group, gender)}
                              </td>
                            ))
                          )}
                        </tr>
                      ))}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="footnote">
              <b>*PwBD</b> - Persons with Benchmark Disabilities (Out of Total), <b>MM</b> - Muslim
              Minority (Out of Total), <b>OM</b> - Other Minority (Out of Total)
            </p>
          </section>
        )}

        {analysis?.centum && (
          <section className="card">
            <div className="section-header">
              <div>
                <h2>Subject-wise Centum Achievers</h2>
                <p>Total entries: {analysis.centum.entries.length}</p>
              </div>
            </div>

            <div className="table-shell">
              <table className="data-table compact">
                <thead>
                  <tr>
                    <th>Sl. No</th>
                    <th>Name</th>
                    <th>Registration No.</th>
                    <th>Subject Name</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.centum.entries.map((entry) => (
                    <tr key={`${entry.sl_no}-${entry.registration_no}-${entry.subject_name}`}>
                      <td>{entry.sl_no}</td>
                      <td>{entry.name}</td>
                      <td>{entry.registration_no}</td>
                      <td>{entry.subject_name}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {analysisPdfFile && (
          <section className="card">
            <div className="section-header">
              <div>
                <h2>Final Combined PDF</h2>
                <p>Single merged report preview and download.</p>
              </div>
              <div className="section-actions">
                <a
                  href={`${API_BASE}/api/files/${analysisPdfFile.id}/download`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <button className="btn btn-primary">Download analysis.pdf</button>
                </a>
              </div>
            </div>

            <iframe
              className="pdf-frame"
              title="analysis-pdf"
              src={`${API_BASE}/api/files/${analysisPdfFile.id}/view`}
            />
          </section>
        )}
      </main>
    </div>
  );
}
