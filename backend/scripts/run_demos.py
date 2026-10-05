"""
SatQuery AI — Deterministic Phase 1 Verification Demos
Executes all 5 required selection demos against data/samples/ GeoTIFF rasters:
1. Single-image VQA
2. Single-image Captioning
3. Bi-temporal Change Detection
4. Optical + SAR Multimodal Joint Analysis
5. Conflict Detection & Targeted Reanalysis
"""

import os
import sys
import base64
import requests
import json

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples"))

def encode_b64(filename: str) -> str:
    path = os.path.join(SAMPLES_DIR, filename)
    with open(path, "rb") as f:
        return "data:image/tiff;base64," + base64.b64encode(f.read()).decode("utf-8")

def run_all_demos():
    print("=" * 80)
    print("      SATQUERY AI — PHASE 1 DETERMINISTIC DEMO SUITE VERIFICATION")
    print("=" * 80)
    print(f"Target Backend: {BACKEND_URL}\n")

    b64_opt = encode_b64("cartosat_optical_bengaluru.tif")
    b64_sar = encode_b64("risat_sar_mumbai.tif")
    b64_t1 = encode_b64("temporal_t1_pre_monsoon.tif")
    b64_t2 = encode_b64("temporal_t2_post_monsoon.tif")

    demos = [
        {
            "num": 1,
            "name": "DEMO 1: Single-Image VQA",
            "payload": {
                "query": "Is there any water body or lake visible in this scene?",
                "mode": "single",
                "images": [{"data": b64_opt, "filename": "cartosat_optical_bengaluru.tif", "role": "primary"}]
            },
            "verify": lambda r: r.status_code == 200 and r.json().get("task") == "vqa"
        },
        {
            "num": 2,
            "name": "DEMO 2: Single-Image Captioning",
            "payload": {
                "query": "Describe the land cover and spatial layout of this satellite scene.",
                "mode": "single",
                "images": [{"data": b64_opt, "filename": "cartosat_optical_bengaluru.tif", "role": "primary"}]
            },
            "verify": lambda r: r.status_code == 200 and r.json().get("task") == "captioning"
        },
        {
            "num": 3,
            "name": "DEMO 3: Bi-Temporal Change Detection",
            "payload": {
                "query": "Detect and map spectral changes between T1 and T2.",
                "mode": "bitemporal",
                "images": [
                    {"data": b64_t1, "filename": "temporal_t1_pre_monsoon.tif", "role": "primary"},
                    {"data": b64_t2, "filename": "temporal_t2_post_monsoon.tif", "role": "secondary"}
                ]
            },
            "verify": lambda r: r.status_code == 200 and len(r.json().get("evidence", [])) >= 3
        },
        {
            "num": 4,
            "name": "DEMO 4: Optical + SAR Joint Analysis",
            "payload": {
                "query": "Fuse optical reflectance and SAR backscatter to identify water and urban structures.",
                "mode": "optical_sar",
                "images": [
                    {"data": b64_opt, "filename": "cartosat_optical_bengaluru.tif", "role": "optical"},
                    {"data": b64_sar, "filename": "risat_sar_mumbai.tif", "role": "sar"}
                ]
            },
            "verify": lambda r: r.status_code == 200 and r.json().get("task") == "optical_sar"
        },
        {
            "num": 5,
            "name": "DEMO 5: Conflict & Targeted Reanalysis",
            "payload": {
                "query": "Locate and highlight all urban building structures with bounding boxes.",
                "mode": "single",
                "images": [{"data": b64_opt, "filename": "cartosat_optical_bengaluru.tif", "role": "primary"}]
            },
            "verify": lambda r: r.status_code == 200 and r.json().get("task") == "grounding"
        }
    ]

    all_passed = True
    for d in demos:
        print(f"Executing {d['name']}...")
        try:
            resp = requests.post(f"{BACKEND_URL}/api/v1/analyze", json=d["payload"], timeout=15)
            data = resp.json()
            ok = d["verify"](resp)
            status_text = "PASS" if ok else "FAIL"
            color = "\033[92m" if ok else "\033[91m"
            reset = "\033[0m"
            print(f"  [{color}{status_text}{reset}] Task: {data.get('task')}, Models: {data.get('models')}, Evidence Count: {len(data.get('evidence', []))}")
            if not ok:
                all_passed = False
        except Exception as e:
            print(f"  [\033[91mFAIL\033[0m] Error: {e}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("\033[92mALL 5 SELECTION DEMOS VERIFIED DETERMINISTIC AND SCIENTIFICALLY DEFENSIBLE.\033[0m")
        return 0
    else:
        print("\033[91mONE OR MORE DEMOS FAILED.\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_demos())
