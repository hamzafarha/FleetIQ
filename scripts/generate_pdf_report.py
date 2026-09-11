"""Generate a comprehensive, professionally styled PDF report for SmartTaxi ETA Project."""
import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PDF_OUTPUT_PATH = Path("reports/SmartTaxi_ETA_Project_Comprehensive_Handbook.pdf")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "SmartTaxi — Smart Arrival Time Estimation (ETA) | Project Handbook")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — ESPRIT / TSE CONSULTANT INT / SMARTTAXI")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf():
    PDF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Brand Colors
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Medium blue
    dark_text = colors.HexColor("#2D3748")       # Charcoal
    bg_light = colors.HexColor("#F7FAFC")        # Off-white
    border_color = colors.HexColor("#CBD5E0")

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_text,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CustomCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1A202C"),
    )

    # Dedicated Table Cell Styles (NEVER mutate these at runtime)
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=dark_text,
    )

    td_bold_style = ParagraphStyle(
        "TableCellBold",
        parent=td_style,
        fontName="Helvetica-Bold",
        textColor=primary_color,
    )

    td_code_style = ParagraphStyle(
        "TableCellCode",
        parent=td_style,
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # --- COVER / HEADER ---
    story.append(Paragraph("SmartTaxi — Smart Arrival Time Estimation (ETA)", title_style))
    story.append(Paragraph("Comprehensive Technical Handbook: Architecture, Pipeline, Models, Fallback & Commands", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceAfter=10))

    meta_data = [
        [Paragraph("<b>Author:</b> Hamza (ETA & ML Lead)", td_style), Paragraph("<b>Context:</b> SmartTaxi Central Ecosystem", td_style)],
        [Paragraph("<b>Status:</b> Production Ready & Benchmark Validated", td_style), Paragraph("<b>Target Environment:</b> Docker / AWS ECS Fargate", td_style)],
        [Paragraph("<b>Backend Integration:</b> ASP.NET Core .NET 10", td_style), Paragraph("<b>Date:</b> September 2026 (Integration Release)", td_style)],
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # --- 1. EXECUTIVE SUMMARY ---
    story.append(Paragraph("1. Executive Summary & Mission Overview", h1_style))
    story.append(Paragraph(
        "The Smart Arrival Time Estimation (ETA) microservice is an independent, production-grade Machine Learning service "
        "built for the SmartTaxi platform. In an urban mobility ecosystem, precise arrival estimations are critical for taxi dispatch "
        "optimization, customer trust, and driver efficiency. Built to interface with the centralized ASP.NET Core backend (Houda Ghenmi), "
        "the service functions as an advisory, stateless prediction engine.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Architectural Rules:</b><br/>"
        "• <b>Zero Database Writes:</b> The ML engine never writes to the <code>Ride</code> table or PostgreSQL database.<br/>"
        "• <b>Authoritative Timestamps:</b> The backend owns all ride lifecycle transitions and timestamps.<br/>"
        "• <b>No Silent Simulation:</b> In the absence of real-time weather or traffic feeds, the service never fabricates synthetic values.<br/>"
        "• <b>High-Availability Fallback:</b> Calibrated urban speed heuristics automatically handle anomalies or server downtime.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # --- 2. DECOUPLED TARGETS ---
    story.append(Paragraph("2. Decoupled Business Targets", h1_style))
    story.append(Paragraph(
        "To eliminate ambiguity, the ETA service decouples prediction into two dedicated endpoints:",
        body_style
    ))

    targets_data = [
        [
            Paragraph("Target Name", th_style),
            Paragraph("Business Purpose", th_style),
            Paragraph("Active Endpoint", th_style),
            Paragraph("Serving Strategy", th_style)
        ],
        [
            Paragraph("<b>Target 1: Driver Pickup ETA</b>", td_bold_style),
            Paragraph("Time required for the driver to reach customer pickup location after dispatch.", td_style),
            Paragraph("<code>POST /predict/driver-pickup</code>", td_code_style),
            Paragraph("Calibrated urban routing fallback (25 km/h urban, 18 km/h rush-hour). Telemetry ML model ready.", td_style)
        ],
        [
            Paragraph("<b>Target 2: Passenger Trip Duration</b>", td_bold_style),
            Paragraph("Estimated duration of the passenger ride from pickup to destination (Case B).", td_style),
            Paragraph("<code>POST /predict/trip-duration</code>", td_code_style),
            Paragraph("Champion ML Model (Random Forest) trained on historical records with automated failover.", td_style)
        ],
    ]
    t_targets = Table(targets_data, colWidths=[110, 145, 125, 124])
    t_targets.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("BOX", (0, 0), (-1, -1), 1, primary_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_targets)
    story.append(Spacer(1, 10))

    # --- 3. TECHNOLOGY STACK ---
    story.append(Paragraph("3. Technology Stack & Frameworks", h1_style))
    tech_data = [
        [Paragraph("Layer", th_style), Paragraph("Technology", th_style), Paragraph("Version / Spec", th_style), Paragraph("Key Role", th_style)],
        [Paragraph("Language", td_bold_style), Paragraph("Python", td_style), Paragraph("3.12 (Local) / 3.11 (Docker)", td_style), Paragraph("High performance, typed, vectorized runtime", td_style)],
        [Paragraph("Data Processing", td_bold_style), Paragraph("Pandas, NumPy, PyArrow", td_style), Paragraph("2.2.2 / 1.26.4 / 25.0.1", td_style), Paragraph("Vectorized math, Parquet columnar storage", td_style)],
        [Paragraph("Spatial Geodesy", td_bold_style), Paragraph("PyProj, PyShp", td_style), Paragraph("3.8.0 / 3.1.6", td_style), Paragraph("WGS84 reprojection, zone centroid mapping", td_style)],
        [Paragraph("Machine Learning", td_bold_style), Paragraph("Scikit-Learn, LightGBM, XGBoost, CatBoost", td_style), Paragraph("1.4.2 / 4.3.0 / 2.0.3 / 1.2.10", td_style), Paragraph("Multi-model tabular regression comparison", td_style)],
        [Paragraph("Model Serialization", td_bold_style), Paragraph("Joblib", td_style), Paragraph("1.4.2", td_style), Paragraph("Champion model bundle persistence", td_style)],
        [Paragraph("Serving Microservice", td_bold_style), Paragraph("FastAPI, Uvicorn, Pydantic", td_style), Paragraph("0.141.1 / 0.52.4 / 2.13.5", td_style), Paragraph("Asynchronous REST endpoints, strict typing", td_style)],
        [Paragraph("User Interface", td_bold_style), Paragraph("Streamlit", td_style), Paragraph("1.63.0", td_style), Paragraph("Interactive live map demo & visual dashboard", td_style)],
        [Paragraph("Testing Suite", td_bold_style), Paragraph("Pytest, TestClient", td_style), Paragraph("8.2.0", td_style), Paragraph("Automated unit and integration verification (24 tests)", td_style)],
        [Paragraph("Containerization", td_bold_style), Paragraph("Docker & AWS ECS", td_style), Paragraph("Fargate Task Definitions", td_style), Paragraph("Containerized multi-stage builds with healthchecks", td_style)],
    ]
    t_tech = Table(tech_data, colWidths=[90, 130, 120, 164])
    t_tech.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
        ("BOX", (0, 0), (-1, -1), 1, secondary_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 10))

    # --- 4. END-TO-END PIPELINE STEPS ---
    story.append(Paragraph("4. End-to-End Pipeline Architecture (How It Works)", h1_style))
    story.append(Paragraph(
        "The project implements a 12-step reproducible pipeline spanning raw data ingestion to production serving:",
        body_style
    ))

    pipeline_steps = [
        ("Step 1: Dataset Audit", "notebooks/01_dataset_audit.ipynb", "Audits 15.7M raw TLC records; demonstrates trip timestamps are sufficient for Trip Duration while driver dispatch data requires decoupling."),
        ("Step 2: Cleaning & Centroids", "src/data/clean_nyc_tlc.py", "Filters duration anomalies (<=0 or >4h); reprojects NYC shapefile polygons to derive WGS84 centroids for 265 taxi zones."),
        ("Step 3: Dataset Design & Split", "src/data/make_dataset.py", "Implements strict chronological splitting (70% train [Jan-Jul], 15% val [Jul-Oct], 15% test [Oct]) to eliminate future data leakage."),
        ("Step 4: Feature Engineering", "src/features/build_features.py", "Builds Haversine, Manhattan distance proxies, cyclical hour encodings (sin/cos), weekend and rush-hour features."),
        ("Step 5: Exploratory Analysis", "notebooks/05_eda.ipynb", "Analyzes duration vs. distance relationships; identifies severe urban congestion slowdowns during peak hours."),
        ("Step 6: Baseline Heuristic", "src/models/baseline.py", "Fits historical average urban speed segmented into 24 hourly buckets (Test MAE: 7.91 min, R²: 0.258)."),
        ("Step 7: Tabular ML Training", "src/models/train.py", "Trains and compares Random Forest, XGBoost, LightGBM, and CatBoost on the exact same chronologically partitioned dataset."),
        ("Step 8: Evaluation & Gates", "src/evaluation/metrics.py", "Evaluates models against business thresholds (MAE, RMSE, R²); conducts slice analysis across trip lengths and peak hours."),
        ("Step 9: Production API", "app.py / deployment/app.py", "FastAPI microservice serving live ML predictions with sub-25ms latency and automatic heuristic failover."),
        ("Step 10: Streamlit Interface", "app_streamlit.py", "Interactive dashboard with map visualization, driver pickup calculator, model benchmark viewer, and stream testing."),
        ("Step 11: Streaming Simulator", "simulate_stream.py", "Generates real-time ride events to load-test the API, verify latency SLAs (<50ms), and inspect quality flags."),
        ("Step 12: Pipeline Automation", "scripts/run_pipeline.py", "Single-command execution running data splitting, ML training, figures generation, pytest suite, and API verification in ~33s."),
    ]

    for title, fpath, desc in pipeline_steps:
        p_text = f"• <b>{title}</b> (<code>{fpath}</code>): {desc}"
        story.append(Paragraph(p_text, bullet_style))

    story.append(Spacer(1, 10))

    # --- 5. BENCHMARK & EVALUATION RESULTS ---
    story.append(Paragraph("5. Model Benchmark Results & Evaluation", h1_style))
    story.append(Paragraph(
        "Evaluation on the held-out out-of-time test set (29,726 trips, October 2025) demonstrates substantial gains "
        "of Machine Learning over the baseline heuristic:",
        body_style
    ))

    bench_data = [
        [
            Paragraph("Model Name", th_style),
            Paragraph("MAE (min)", th_style),
            Paragraph("MAE (sec)", th_style),
            Paragraph("RMSE (min)", th_style),
            Paragraph("R² Score", th_style),
            Paragraph("MAPE (%)", th_style),
            Paragraph("Latency (ms)", th_style),
        ],
        [
            Paragraph("Baseline (Avg Speed)", td_style),
            Paragraph("7.91 min", td_style),
            Paragraph("474.4 s", td_style),
            Paragraph("13.14 min", td_style),
            Paragraph("0.258", td_style),
            Paragraph("41.9%", td_style),
            Paragraph("<b>0.02 ms</b>", td_bold_style),
        ],
        [
            Paragraph("<b>Random Forest (Champion)</b>", td_bold_style),
            Paragraph("<b>5.20 min</b>", td_bold_style),
            Paragraph("<b>312.1 s</b>", td_bold_style),
            Paragraph("<b>8.54 min</b>", td_bold_style),
            Paragraph("<b>0.687</b>", td_bold_style),
            Paragraph("<b>32.1%</b>", td_bold_style),
            Paragraph("21.37 ms", td_style),
        ],
        [
            Paragraph("LightGBM Regressor", td_style),
            Paragraph("5.28 min", td_style),
            Paragraph("316.8 s", td_style),
            Paragraph("8.63 min", td_style),
            Paragraph("0.680", td_style),
            Paragraph("33.0%", td_style),
            Paragraph("<b>1.10 ms</b>", td_bold_style),
        ],
        [
            Paragraph("XGBoost Regressor", td_style),
            Paragraph("5.30 min", td_style),
            Paragraph("317.7 s", td_style),
            Paragraph("8.65 min", td_style),
            Paragraph("0.678", td_style),
            Paragraph("33.2%", td_style),
            Paragraph("1.88 ms", td_style),
        ],
        [
            Paragraph("CatBoost Regressor", td_style),
            Paragraph("5.40 min", td_style),
            Paragraph("324.1 s", td_style),
            Paragraph("8.77 min", td_style),
            Paragraph("0.670", td_style),
            Paragraph("34.2%", td_style),
            Paragraph("1.25 ms", td_style),
        ],
    ]
    t_bench = Table(bench_data, colWidths=[130, 60, 60, 64, 60, 60, 70])
    t_bench.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("BOX", (0, 0), (-1, -1), 1, primary_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Key Insights:</b><br/>"
        "• <b>Error Reduction:</b> Machine Learning achieves a <b>34% reduction</b> in overall MAE and a massive <b>62% reduction</b> on long trips (from 29.65 min down to 11.31 min).<br/>"
        "• <b>Explained Variance:</b> R² increases from 0.258 to <b>0.687</b> (+165% improvement).<br/>"
        "• <b>Model Choice:</b> Random Forest was selected as champion for maximum accuracy, while LightGBM serves as a high-throughput scaling alternative (1.1 ms).",
        body_style
    ))
    story.append(Spacer(1, 8))

    # --- 6. EMBEDDED FIGURES ---
    fig1 = Path("reports/figures/model_comparison_benchmark.png")
    fig2 = Path("reports/figures/feature_importance.png")
    if fig1.exists() and fig2.exists():
        story.append(Paragraph("Visual Analytics & Feature Importances:", h2_style))
        img_table = [
            [
                Image(str(fig1), width=3.4 * inch, height=1.55 * inch),
                Image(str(fig2), width=3.4 * inch, height=1.55 * inch),
            ]
        ]
        t_imgs = Table(img_table, colWidths=[252, 252])
        t_imgs.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(t_imgs)
        story.append(Spacer(1, 10))

    # --- 7. COMPLETE COMMAND HANDBOOK ---
    story.append(Paragraph("6. Complete Command Reference Handbook", h1_style))
    story.append(Paragraph(
        "Essential terminal commands for local execution, testing, serving, and deployment:",
        body_style
    ))

    cmd_blocks = [
        ("Environment Setup & Activation", [
            "# Create and activate virtual environment",
            "python -m venv .venv",
            ".venv\\Scripts\\Activate.ps1",
            "pip install -r requirements.txt",
        ]),
        ("End-to-End Pipeline Execution", [
            "# Run all 5 stages in ~33 seconds (data, models, figures, tests, api)",
            ".venv\\Scripts\\python.exe scripts/run_pipeline.py",
            "# Or via Makefile shortcut",
            "make run-pipeline",
        ]),
        ("Automated Testing Suite (24 Tests)", [
            "# Run complete test suite with detailed output",
            ".venv\\Scripts\\pytest.exe -v",
            "# Run specific test files",
            ".venv\\Scripts\\pytest.exe tests/test_api.py -v",
        ]),
        ("Production API Microservice (FastAPI / Uvicorn)", [
            "# Start FastAPI server with hot-reload on port 8000",
            ".venv\\Scripts\\uvicorn.exe app:app --host 0.0.0.0 --port 8000 --reload",
            "# Interactive Swagger UI: http://127.0.0.1:8000/docs",
            "# Healthcheck: curl http://127.0.0.1:8000/health",
        ]),
        ("Interactive Streamlit Dashboard UI", [
            "# Start Streamlit app with interactive map on port 8501",
            ".venv\\Scripts\\streamlit.exe run app_streamlit.py --server.port 8501",
            "# Web UI URL: http://localhost:8501",
        ]),
        ("Real-time Event Streaming Simulator", [
            "# Send 15 consecutive simulated taxi ride requests",
            ".venv\\Scripts\\python.exe simulate_stream.py --num-trips 15 --interval 0.2",
        ]),
        ("Docker Containerization & ECS Deployment", [
            "# Build API image",
            "docker build -t smart-eta-api:latest -f Dockerfile .",
            "# Run API container",
            "docker run -d -p 8000:8000 --name eta-api smart-eta-api:latest",
            "# Build and run Streamlit UI container",
            "docker build -t smart-eta-streamlit:latest -f Dockerfile.streamlit .",
            "docker run -d -p 8501:8501 -e API_URL='http://host.docker.internal:8000' smart-eta-streamlit:latest",
        ]),
    ]

    for ctitle, clines in cmd_blocks:
        code_text = "<br/>".join(clines)
        c_table = Table([[Paragraph(f"<b>{ctitle}</b>", td_bold_style)], [Paragraph(code_text, td_code_style)]], colWidths=[504])
        c_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), bg_light),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(KeepTogether(c_table))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 8))

    # --- 8. CONCLUSION & INTEGRATION CHECKLIST ---
    story.append(Paragraph("7. Integration Readiness & Verification Checklist", h1_style))
    checklist_data = [
        [Paragraph("Check Item", th_style), Paragraph("Acceptance Criteria", th_style), Paragraph("Verification Status", th_style)],
        [Paragraph("Target Decoupling", td_bold_style), Paragraph("Driver pickup & trip duration endpoints clearly separated", td_style), Paragraph("PASSED (2 endpoints)", td_bold_style)],
        [Paragraph("Data Quality & Splitting", td_bold_style), Paragraph("Zero future data leakage; chronological time-based split", td_style), Paragraph("PASSED (Train 70% / Val 15% / Test 15%)", td_bold_style)],
        [Paragraph("Model Benchmarking", td_bold_style), Paragraph("Comparison of 5 tabular models; champion serialized", td_style), Paragraph("PASSED (Random Forest Champion)", td_bold_style)],
        [Paragraph("Business Acceptance", td_bold_style), Paragraph("R² >= 0.60, MAE sub-urban <= 4.0m, latency SLA < 50ms", td_style), Paragraph("PASSED (R²=0.687, Latency=21ms)", td_bold_style)],
        [Paragraph("Backend Governance", td_bold_style), Paragraph("Zero direct writes to DB; no silent fake traffic simulation", td_style), Paragraph("PASSED (Strict read-only & flags)", td_bold_style)],
        [Paragraph("High-Availability Fallback", td_bold_style), Paragraph("Automated failover to urban routing speed on model failure", td_style), Paragraph("PASSED (Fallback routing tested)", td_bold_style)],
        [Paragraph("Automated Testing Suite", td_bold_style), Paragraph("100% pass rate on unit & integration test harness", td_style), Paragraph("PASSED (24/24 tests green)", td_bold_style)],
    ]
    t_check = Table(checklist_data, colWidths=[130, 240, 134])
    t_check.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("BOX", (0, 0), (-1, -1), 1, primary_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_check)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully compiled PDF handbook to: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
