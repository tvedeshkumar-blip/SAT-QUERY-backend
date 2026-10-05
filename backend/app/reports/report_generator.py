import io
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("satquery.reports")

def generate_json_report(analysis_response: Dict[str, Any]) -> str:
    """Generates formatted JSON report string."""
    return json.dumps(analysis_response, indent=2)

def generate_pdf_report(analysis_response: Dict[str, Any]) -> bytes:
    """
    Generates a professional PDF analysis report using ReportLab.
    Includes SatQuery AI metadata, task classification, model execution trace, answer, and evidence summary.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#0284c7')
        )
        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#475569')
        )
        h2_style = ParagraphStyle(
            'H2',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#0f172a')
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155')
        )

        elements = []

        # Header Title
        elements.append(Paragraph("SATQUERY AI — MULTIMODAL REMOTE SENSING ANALYSIS REPORT", title_style))
        elements.append(Paragraph(f"Report ID: {analysis_response.get('id')} | Generated: {analysis_response.get('created_at')}", subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0284c7'), spaceAfter=15))

        # Metadata Table
        meta = analysis_response.get("metadata", {})
        meta_data = [
            [Paragraph("<b>Task Classification:</b>", body_style), Paragraph(str(analysis_response.get("task")).upper(), body_style)],
            [Paragraph("<b>Operational Mode:</b>", body_style), Paragraph(str(analysis_response.get("mode")), body_style)],
            [Paragraph("<b>Specialist Models:</b>", body_style), Paragraph(", ".join(analysis_response.get("models", [])), body_style)],
            [Paragraph("<b>Confidence Score:</b>", body_style), Paragraph(str(analysis_response.get("confidence_label")), body_style)],
            [Paragraph("<b>Coordinate Reference System (CRS):</b>", body_style), Paragraph(str(meta.get("crs") or "Not Specified / Local Frame"), body_style)],
            [Paragraph("<b>Execution Time:</b>", body_style), Paragraph(f"{analysis_response.get('execution_time_ms')} ms", body_style)],
        ]
        t = Table(meta_data, colWidths=[200, 340])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # Primary Answer Section
        elements.append(Paragraph("1. AGENT SYNTHESIS & FINDINGS", h2_style))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(analysis_response.get("answer", "N/A"), body_style))
        elements.append(Spacer(1, 15))

        # Visual Evidence Summary
        elements.append(Paragraph("2. VISUAL EVIDENCE ARTIFACTS", h2_style))
        elements.append(Spacer(1, 5))
        evidence = analysis_response.get("evidence", [])
        if evidence:
            ev_table_data = [["Evidence ID", "Type", "Title", "Details"]]
            for ev in evidence:
                stats = ev.get("statistics") or {}
                stat_str = ", ".join([f"{k}: {v}" for k, v in stats.items()]) if stats else "Visual Overlay"
                ev_table_data.append([
                    ev.get("id"),
                    ev.get("type"),
                    ev.get("title"),
                    stat_str
                ])
            et = Table(ev_table_data, colWidths=[100, 80, 180, 180])
            et.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(et)
        else:
            elements.append(Paragraph("No visual evidence artifacts generated for this query.", body_style))

        elements.append(Spacer(1, 15))

        # Agent Execution Trace Summary
        elements.append(Paragraph("3. AGENT EXECUTION TRACE", h2_style))
        elements.append(Spacer(1, 5))
        trace_steps = analysis_response.get("trace", {}).get("steps", [])
        if trace_steps:
            tr_data = [["Step", "Timestamp", "Detail"]]
            for ts in trace_steps:
                tr_data.append([ts.get("step"), ts.get("timestamp")[-12:-1], ts.get("detail")])
            trt = Table(tr_data, colWidths=[140, 90, 310])
            trt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(trt)

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"ReportLab PDF generation error ({e}); constructing text PDF fallback.")
        # Minimal text-based PDF fallback string
        pdf_fallback = f"%PDF-1.4\nSATQUERY AI REPORT\nTask: {analysis_response.get('task')}\nAnswer: {analysis_response.get('answer')}"
        return pdf_fallback.encode('utf-8')
