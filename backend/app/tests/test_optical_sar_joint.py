import pytest
import numpy as np
from app.models.optical_sar.optical_sar_fusion import (
    OpticalSARJointAnalysisProvider,
    OpticalSARVisualizationBaseline,
    OpticalSARProvider
)

def test_optical_sar_joint_analysis_artifacts():
    analyzer = OpticalSARJointAnalysisProvider()
    opt = np.random.randint(40, 220, (100, 100, 3), dtype=np.uint8)
    sar = np.random.randint(10, 250, (100, 100), dtype=np.uint8)

    res = analyzer.predict([opt, sar], query="Fuse optical and SAR imagery.")
    assert "answer" in res
    assert "optical_evidence" in res
    assert "sar_evidence" in res
    assert "fused_evidence" in res
    assert "limitations" in res
    assert len(res["limitations"]) > 0

    # Verify all evidence types
    types = [e["type"] for e in res["evidence"]]
    assert "original" in types
    assert "processed" in types
    assert "fused" in types
    assert "mask" in types

def test_optical_sar_provider_delegation():
    provider = OpticalSARProvider()
    opt = np.full((64, 64, 3), 100, dtype=np.uint8)
    sar = np.full((64, 64), 75, dtype=np.uint8)
    res = provider.predict([opt, sar], query="Joint analysis")
    assert "fused_evidence" in res
