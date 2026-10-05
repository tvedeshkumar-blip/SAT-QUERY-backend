import pytest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.reports.report_generator import generate_pdf_report, generate_json_report

def test_json_report():
    data = {
        "id": "sat_test_123",
        "task": "vqa",
        "answer": "Land cover consists of agricultural crops and water canals.",
        "models": ["SatQueryVQA-RSAdapter"],
        "execution_time_ms": 120.5
    }
    json_str = generate_json_report(data)
    parsed = json.loads(json_str)
    assert parsed["id"] == "sat_test_123"
    assert parsed["task"] == "vqa"

def test_pdf_report():
    data = {
        "id": "sat_test_456",
        "task": "change_vqa",
        "mode": "bitemporal",
        "models": ["SatChangeVQA-RSNet", "SatChangeDetector-BiTemporal"],
        "answer": "Land surface modification detected over 12.63% of the scene area.",
        "confidence_label": "Not available",
        "execution_time_ms": 185.0,
        "created_at": "2026-09-26T13:00:00Z",
        "metadata": {"crs": "EPSG:32644"},
        "evidence": [
            {"id": "ev1", "type": "change_map", "title": "Change Map", "statistics": {"changed_area_percent": "12.63%"}}
        ],
        "trace": {
            "steps": [
                {"step": "QUERY_RECEIVED", "timestamp": "2026-09-26T13:00:00Z", "detail": "Query received"}
            ]
        }
    }
    pdf_bytes = generate_pdf_report(data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")
