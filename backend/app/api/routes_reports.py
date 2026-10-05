from fastapi import APIRouter, Response, Body
from app.reports.report_generator import generate_pdf_report, generate_json_report

router = APIRouter()

@router.post("/reports/pdf")
def download_pdf_report(payload: dict = Body(...)):
    pdf_bytes = generate_pdf_report(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=satquery_report_{payload.get('id', 'export')}.pdf"}
    )

@router.post("/reports/json")
def download_json_report(payload: dict = Body(...)):
    json_str = generate_json_report(payload)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=satquery_report_{payload.get('id', 'export')}.json"}
    )
