import re
from typing import Any
from fastapi import APIRouter, Response, Body, HTTPException
from app.reports.report_generator import (
    generate_pdf_report,
    generate_json_report,
    generate_markdown_report,
)
from app.reports.scientific_report import ScientificReportBuilder
from app.schemas.report import ScientificReportSchema

router = APIRouter()


def sanitize_filename(base: Any, ext: str, prefix: str = "satquery_report") -> str:
    """
    Sanitizes report export filename against path traversal, control characters,
    and invalid filesystem symbols.
    """
    raw_str = str(base or "export").strip()
    # Strip path separators and traversal attempts
    raw_str = raw_str.replace("/", "_").replace("\\", "_").replace("..", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_str).strip("._")
    if not cleaned:
        cleaned = "export"
    cleaned = cleaned[:40]
    return f"{prefix}_{cleaned}.{ext}"


@router.post("/reports/pdf")
def download_pdf_report(payload: dict = Body(...)):
    filename = sanitize_filename(payload.get("id"), "pdf")
    pdf_bytes = generate_pdf_report(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/reports/json")
def download_json_report(payload: dict = Body(...)):
    filename = sanitize_filename(payload.get("id"), "json")
    json_str = generate_json_report(payload)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/reports/markdown")
def download_markdown_report(payload: dict = Body(...)):
    """
    Stage 8: Downloads reproducible audit-grade scientific report in Markdown format.
    """
    try:
        filename = sanitize_filename(payload.get("id"), "md")
        md_content = generate_markdown_report(payload)
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate scientific markdown report: {str(e)}")


@router.post("/reports/scientific", response_model=ScientificReportSchema)
def get_scientific_report_schema(payload: dict = Body(...)):
    """
    Stage 8: Returns validated structured ScientificReportSchema for an analysis response.
    """
    try:
        if "reproducible_report" in payload and payload["reproducible_report"]:
            rep = payload["reproducible_report"]
            if isinstance(rep, dict):
                return ScientificReportSchema(**rep)
            if isinstance(rep, ScientificReportSchema):
                return rep
        return ScientificReportBuilder.build_report(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build scientific report: {str(e)}")
