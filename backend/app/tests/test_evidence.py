import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.models.grounding.grounder import RemoteSensingGrounder
from app.models.change_detection.change_detector import ChangeDetector
from app.models.optical_sar.optical_sar_fusion import OpticalSARAnalyzer

def test_grounding_evidence_generation():
    grounder = RemoteSensingGrounder()
    # Image with dark water patch
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    img[20:60, 20:60] = 30  # Dark water feature
    res = grounder.predict([img], query="Highlight the water body.")
    assert "evidence" in res
    assert len(res["evidence"]) >= 2
    types = [e["type"] for e in res["evidence"]]
    assert "overlay" in types
    assert "mask" in types

def test_change_detection_evidence_generation():
    detector = ChangeDetector()
    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2[30:70, 30:70] = 200  # Marked change zone
    res = detector.predict([img1, img2], query="What changed?")
    assert "evidence" in res
    assert res["changed_area_percent"] > 0
    assert len(res["evidence"]) >= 2
    assert res["evidence"][0]["type"] == "change_map"

def test_optical_sar_fusion_evidence_generation():
    analyzer = OpticalSARAnalyzer()
    opt = np.full((100, 100, 3), 120, dtype=np.uint8)
    sar = np.full((100, 100), 80, dtype=np.uint8)
    sar[10:40, 10:40] = 20  # Specular water
    sar[60:90, 60:90] = 240 # Double bounce urban
    res = analyzer.predict([opt, sar], query="Identify built-up and water.")
    assert "evidence" in res
    types = [e["type"] for e in res["evidence"]]
    assert "fused" in types
    assert "mask" in types
