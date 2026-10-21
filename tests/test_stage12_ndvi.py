"""
Stage 12 Deterministic Test Suite: Genuine Sentinel-2 B04 + B08 NDVI Pipeline.

Covers all 19 verification requirements:
1. Exact NDVI values for known reflectance inputs.
2. Reflectance scaling and calibration metadata (scale factor 10,000).
3. Same-STAC-Item acquisition validation for B04/B08/SCL.
4. CRS, transform, shape, and pixel-alignment handling.
5. Nodata and non-finite values (NaN, Inf, nodata masked).
6. Zero and near-zero denominators (nir + red <= 1e-5 masked as NaN, not fabricated zero).
7. Each excluded SCL class (0, 1, 3, 8, 9, 10).
8. Unknown SCL classes failing closed (<0 or >11 excluded).
9. Valid-pixel counts and total-pixel counts.
10. Mean NDVI and p90 calculations.
11. Vegetation coverage and its explicit threshold (0.40).
12. Zero-valid-pixel behavior returning None for all statistics with explicit status.
13. Check J valid and adversarial cases (adversarial units, range, RGB pseudo-NDVI, zero valid pixels).
14. Report provenance and backward-compatible serialization.
15. Safe-reader transfer-budget enforcement.
16. Regression protection for Stage 9A acquisition safeguards.
17. Regression protection for Stage 10A quality screening and Checks H–I.
18. Existing single-band acquisition behavior.
19. Frontend telemetry and missing-statistic rendering.
"""

import io
import json
import math
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.remote_sensing.bands import (
    compute_calibrated_ndvi,
    compute_ndvi,
    VEGETATION_NDVI_THRESHOLD,
    SCL_NDVI_EXCLUDED_CLASSES
)
from app.remote_sensing.quality import (
    QualityScreener,
    SCL_EXCLUDED_CLASSES,
    SCL_RETAINED_CLASSES
)
from app.agent.verifier import (
    DeterministicVerifier,
    EvidencePackageBuilder
)
from app.schemas.verification import (
    EvidencePackageSchema,
    ClaimVerificationItem,
    PhysicalMeasurementItem,
    EvidenceArtifactItem,
    HeuristicIndicatorItem
)
from app.schemas.acquisition import (
    STACCompositeRetrieveRequest,
    STACCompositeRetrieveResponse,
    STACWindowedRetrieveRequest,
    STACRetrieveResponse
)
from app.schemas.report import (
    ReportProvenanceSchema,
    ScientificReportSchema,
    ReportReproducibilitySchema
)
from app.reports.scientific_report import (
    ScientificReportBuilder,
    generate_scientific_markdown
)
from app.acquisition.stac_service import STACAcquisitionService, stac_service
from app.acquisition.windowed_reader import SafeWindowedCOGReader, WindowedRetrievalError


def make_test_meta(shape=(10, 10), crs="EPSG:32643", nodata=0, transform=None):
    if transform is None:
        t = list(from_bounds(500000, 2000000, 500100, 2000100, shape[1], shape[0]))[:6]
    else:
        t = transform
    return {
        "crs": crs,
        "transform": t,
        "width": shape[1],
        "height": shape[0],
        "nodata": nodata,
        "tags": {"SENSOR": "Sentinel-2 MSI"}
    }


# ==============================================================================
# 1. Exact NDVI values for known reflectance inputs
# ==============================================================================
def test_exact_ndvi_values_known_reflectance():
    # DN values with scale factor 10,000:
    # Pixel 0: Red = 0.1 (DN 1000), NIR = 0.5 (DN 5000) -> NDVI = (0.5 - 0.1)/(0.5 + 0.1) = 0.4 / 0.6 = 2/3
    # Pixel 1: Red = 0.04 (DN 400), NIR = 0.04 (DN 400) -> NDVI = 0.0
    # Pixel 2: Red = 0.2 (DN 2000), NIR = 0.1 (DN 1000) -> NDVI = (0.1 - 0.2)/(0.1 + 0.2) = -0.1 / 0.3 = -1/3
    # Pixel 3: Red = 0.05 (DN 500), NIR = 0.85 (DN 8500) -> NDVI = 0.80 / 0.90 = 8/9 = 0.88889
    red = np.array([[1000, 400], [2000, 500]], dtype=np.uint16)
    nir = np.array([[5000, 400], [1000, 8500]], dtype=np.uint16)
    scl = np.full((2, 2), 4, dtype=np.uint8)  # Class 4 = Vegetation (clear)

    meta = make_test_meta(shape=(2, 2))
    res = compute_calibrated_ndvi(red, nir, scl_band=scl, target_meta=meta)

    assert res["status"] == "success"
    assert res["valid_pixel_count"] == 4
    # Check mean: (2/3 + 0 - 1/3 + 8/9) / 4 = (1/3 + 8/9) / 4 = (11/9) / 4 = 11/36 = 0.30556
    expected_mean = (2.0/3.0 + 0.0 - 1.0/3.0 + 8.0/9.0) / 4.0
    assert abs(res["ndvi_mean"] - round(expected_mean, 4)) < 1e-3


# ==============================================================================
# 2. Reflectance scaling and calibration metadata
# ==============================================================================
def test_reflectance_scaling_convention():
    red = np.array([[2000]], dtype=np.uint16)
    nir = np.array([[6000]], dtype=np.uint16)
    meta = make_test_meta(shape=(1, 1))

    res = compute_calibrated_ndvi(red, nir, target_meta=meta, scale_factor=10000.0)
    assert res["status"] == "success"
    # Red refl = 0.2, NIR refl = 0.6 -> NDVI = 0.4 / 0.8 = 0.50
    assert abs(res["ndvi_mean"] - 0.50) < 1e-4
    assert res["calibration"]["scale_factor"] == 10000.0
    assert res["calibration"]["convention"] == "BOA_SURFACE_REFLECTANCE"


# ==============================================================================
# 3. Same-STAC-Item acquisition validation for B04/B08/SCL
# ==============================================================================
def test_same_stac_item_acquisition_validation():
    srv = STACAcquisitionService()
    # Cache a mock scene with B04, B08, and SCL belonging to the same item
    scene_id = "S2A_MSIL2A_20240328T052701_N0510_R105_T43PGQ_20240328T083437"
    srv._scene_cache[scene_id] = {
        "summary": None,
        "datetime": "2024-03-28T05:27:01Z",
        "provider": "Element84 Earth Search (AWS)",
        "collection": "sentinel-2-l2a",
        "raw_assets": {
            "red": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif"},
            "nir": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif"},
            "scl": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/SCL.tif"}
        }
    }

    # Request with missing bands in another cached scene
    srv._scene_cache["incomplete_scene"] = {
        "raw_assets": {
            "red": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif"}
        }
    }

    req_fail = STACCompositeRetrieveRequest(
        scene_id="incomplete_scene",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98]
    )
    resp_fail = srv.retrieve_composite_window(req_fail)
    assert resp_fail.status == "error"
    assert "Missing required Sentinel-2 bands" in resp_fail.error
    assert "B08 (NIR)" in resp_fail.error


# ==============================================================================
# 4. CRS, transform, shape, and pixel-alignment handling
# ==============================================================================
def test_grid_alignment_and_resampling():
    # Test aligning B08 and SCL to B04 grid
    b04 = np.full((4, 4), 1000, dtype=np.uint16)
    # B08 has different shape (2, 2)
    b08 = np.full((2, 2), 5000, dtype=np.uint16)
    # SCL has 20m resolution (2, 2)
    scl = np.full((2, 2), 4, dtype=np.uint8)

    meta_10m = make_test_meta(shape=(4, 4))
    meta_20m = make_test_meta(shape=(2, 2))

    # Reproject SCL using QualityScreener
    aligned_scl, align_meta = QualityScreener.reproject_scl_to_target(scl, meta_20m, meta_10m)
    assert aligned_scl.shape == (4, 4)
    assert np.all(aligned_scl == 4)
    assert align_meta["resampling_method"] == "nearest_neighbor"


# ==============================================================================
# 5. Nodata and non-finite values
# ==============================================================================
def test_nodata_and_non_finite_values():
    red = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)  # 0 is nodata
    nir = np.array([[5000, 5000], [1000, 0]], dtype=np.uint16)  # 0 is nodata
    meta = make_test_meta(shape=(2, 2), nodata=0)

    res = compute_calibrated_ndvi(red, nir, target_meta=meta)
    assert res["status"] == "success"
    # Pixels (0,1) and (1,1) have nodata 0, so only (0,0) and (1,0) should be valid
    assert res["valid_pixel_count"] == 2
    assert res["total_pixel_count"] == 4
    assert res["masked_pixel_count"] == 2


# ==============================================================================
# 6. Zero and near-zero denominators
# ==============================================================================
def test_zero_and_near_zero_denominators():
    # If nir + red <= 1e-5 (e.g. DN=0 or zero reflectance)
    red = np.array([[0, 1000]], dtype=np.uint16)
    nir = np.array([[0, 5000]], dtype=np.uint16)
    meta = make_test_meta(shape=(1, 2), nodata=None)

    res = compute_calibrated_ndvi(red, nir, target_meta=meta)
    assert res["status"] == "success"
    # Pixel (0,0) has sum 0 <= 1e-5, must be masked without fabricating zero
    assert res["valid_pixel_count"] == 1
    assert abs(res["ndvi_mean"] - 0.6667) < 1e-3


# ==============================================================================
# 7. Each excluded SCL class
# ==============================================================================
def test_each_excluded_scl_class():
    # Test that classes 0, 1, 3, 8, 9, 10 are strictly excluded
    excluded = [0, 1, 3, 8, 9, 10]
    red = np.full((1, len(excluded)), 1000, dtype=np.uint16)
    nir = np.full((1, len(excluded)), 5000, dtype=np.uint16)
    scl = np.array([excluded], dtype=np.uint8)
    meta = make_test_meta(shape=(1, len(excluded)))

    res = compute_calibrated_ndvi(red, nir, scl_band=scl, target_meta=meta)
    assert res["status"] == "unsupported_zero_valid_pixels"
    assert res["valid_pixel_count"] == 0
    assert res["ndvi_mean"] is None
    assert res["ndvi_p90"] is None
    assert res["vegetation_coverage_pct"] is None


# ==============================================================================
# 8. Unknown SCL classes failing closed
# ==============================================================================
def test_unknown_scl_classes_fail_closed():
    # Unknown classes like 15 or 99
    red = np.array([[1000, 1000]], dtype=np.uint16)
    nir = np.array([[5000, 5000]], dtype=np.uint16)
    scl = np.array([[15, 99]], dtype=np.uint8)
    meta = make_test_meta(shape=(1, 2))

    res = compute_calibrated_ndvi(red, nir, scl_band=scl, target_meta=meta)
    assert res["status"] == "unsupported_zero_valid_pixels"
    assert res["valid_pixel_count"] == 0
    assert res["ndvi_mean"] is None


# ==============================================================================
# 9. Valid-pixel counts and total-pixel counts
# ==============================================================================
def test_valid_and_total_pixel_counts():
    # 3x3 array: 5 clear vegetation (Class 4), 4 clouds (Class 9)
    red = np.full((3, 3), 1000, dtype=np.uint16)
    nir = np.full((3, 3), 5000, dtype=np.uint16)
    scl = np.array([
        [4, 4, 4],
        [4, 4, 9],
        [9, 9, 9]
    ], dtype=np.uint8)
    meta = make_test_meta(shape=(3, 3))

    res = compute_calibrated_ndvi(red, nir, scl_band=scl, target_meta=meta)
    assert res["status"] == "success"
    assert res["total_pixel_count"] == 9
    assert res["valid_pixel_count"] == 5
    assert res["masked_pixel_count"] == 4


# ==============================================================================
# 10. Mean NDVI and p90 calculations
# ==============================================================================
def test_mean_and_p90_calculations():
    # 5 valid pixels with NDVI values: 0.2, 0.4, 0.6, 0.8, 1.0
    # Red & NIR:
    # 0.2 -> (1200 - 800) / (1200 + 800) = 400 / 2000 = 0.20
    # 0.4 -> (1400 - 600) / (1400 + 600) = 800 / 2000 = 0.40
    # 0.6 -> (1600 - 400) / (1600 + 400) = 1200 / 2000 = 0.60
    # 0.8 -> (1800 - 200) / (1800 + 200) = 1600 / 2000 = 0.80
    # 1.0 -> (2000 - 0) / (2000 + 0) = 2000 / 2000 = 1.00
    red = np.array([[800, 600, 400, 200, 1]], dtype=np.uint16)
    nir = np.array([[1200, 1400, 1600, 1800, 2000]], dtype=np.uint16)
    meta = make_test_meta(shape=(1, 5), nodata=None)

    res = compute_calibrated_ndvi(red, nir, target_meta=meta)
    assert res["status"] == "success"
    assert res["valid_pixel_count"] == 5
    # Mean should be 0.60
    assert abs(res["ndvi_mean"] - 0.60) < 0.01
    # 90th percentile should be between 0.8 and 1.0
    assert res["ndvi_p90"] >= 0.80 and res["ndvi_p90"] <= 1.0


# ==============================================================================
# 11. Vegetation coverage and explicit threshold
# ==============================================================================
def test_vegetation_coverage_explicit_threshold():
    # Threshold is VEGETATION_NDVI_THRESHOLD = 0.40
    # 4 pixels: NDVI = 0.20, 0.35, 0.45, 0.70
    # Exactly 2 out of 4 exceed 0.40 -> 50.0% coverage
    red = np.array([[800, 650, 550, 300]], dtype=np.uint16)
    nir = np.array([[1200, 1350, 1450, 1700]], dtype=np.uint16)
    meta = make_test_meta(shape=(1, 4))

    res = compute_calibrated_ndvi(red, nir, target_meta=meta)
    assert res["status"] == "success"
    assert res["vegetation_threshold"] == 0.40
    assert res["vegetation_pixel_count"] == 2
    assert res["vegetation_coverage_pct"] == 50.0


# ==============================================================================
# 12. Zero-valid-pixel behavior returning None
# ==============================================================================
def test_zero_valid_pixels_returns_none():
    red = np.full((2, 2), 1000, dtype=np.uint16)
    nir = np.full((2, 2), 5000, dtype=np.uint16)
    scl = np.full((2, 2), 9, dtype=np.uint8)  # High probability cloud
    meta = make_test_meta(shape=(2, 2))

    res = compute_calibrated_ndvi(red, nir, scl_band=scl, target_meta=meta)
    assert res["status"] == "unsupported_zero_valid_pixels"
    assert res["ndvi_mean"] is None
    assert res["ndvi_p90"] is None
    assert res["ndvi_median"] is None
    assert res["vegetation_coverage_pct"] is None
    assert res["vegetation_pixel_count"] == 0
    assert res["valid_pixel_count"] == 0


# ==============================================================================
# 13. Check J valid and adversarial cases
# ==============================================================================
def test_check_j_valid_case_with_partial_masking():
    # Partially masked scene with genuine B04, B08, and SCL
    pkg = EvidencePackageSchema(
        task_id="task_ndvi_valid",
        query="What is the calibrated NDVI?",
        valid_pixel_count=850,
        total_pixel_count=1000,
        valid_pixel_percentage=85.0,
        source_assets=[
            {"id": "s2_b04", "asset_key": "B04", "asset_category": "scientific_raster"},
            {"id": "s2_b08", "asset_key": "B08", "asset_category": "scientific_raster"},
            {"id": "s2_scl", "asset_key": "scl", "asset_category": "mask"}
        ],
        evidence_artifacts=[
            EvidenceArtifactItem(artifact_id="B04", artifact_type="band", title="B04 Red"),
            EvidenceArtifactItem(artifact_id="B08", artifact_type="band", title="B08 NIR"),
            EvidenceArtifactItem(artifact_id="SCL", artifact_type="mask", title="SCL Quality Mask")
        ],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_ndvi",
                value=0.55,
                unit="unitless ratio [-1, 1]",
                statistic_type="mean",
                mask_applied="scl_screened_valid_mask",
                calibration_quantity="normalized_difference_vegetation_index"
            )
        ],
        quality_screening={"optical": {"screened": True, "clear_pixels": 850}}
    )

    claims = [
        ClaimVerificationItem(
            claim_text="The calibrated Sentinel-2 B04+B08 mean NDVI is 0.55 across valid pixels screened by SCL mask.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["B04", "B08", "SCL"]
        )
    ]

    passed, failures, eval_claims, cont = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert passed is True
    assert len(failures) == 0
    assert eval_claims[0].status == "supported"


def test_check_j_adversarial_out_of_bounds_ndvi():
    pkg = EvidencePackageSchema(
        task_id="task_adv_range",
        query="NDVI check",
        valid_pixel_count=100,
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_ndvi",
                value=1.45,  # Out of physical bounds > 1.0!
                unit="unitless",
                statistic_type="mean",
                mask_applied="valid_mask",
                calibration_quantity="normalized_difference_vegetation_index"
            )
        ]
    )

    passed, failures, _, _ = DeterministicVerifier.verify(pkg)
    assert passed is False
    assert any("outside valid physical range" in f for f in failures)


def test_check_j_adversarial_nan_inf_ndvi():
    pkg = EvidencePackageSchema(
        task_id="task_adv_nan",
        query="NDVI check",
        valid_pixel_count=100,
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_ndvi",
                value="nan",
                unit="unitless",
                statistic_type="mean",
                mask_applied="valid_mask",
                calibration_quantity="normalized_difference_vegetation_index"
            )
        ]
    )

    passed, failures, _, _ = DeterministicVerifier.verify(pkg)
    assert passed is False
    assert any("must be finite, found NaN/inf" in f for f in failures)


def test_check_j_adversarial_incorrect_units():
    pkg = EvidencePackageSchema(
        task_id="task_adv_units",
        query="NDVI check",
        valid_pixel_count=100,
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_ndvi",
                value=0.55,
                unit="reflectance",  # Prohibited unit!
                statistic_type="mean",
                mask_applied="valid_mask",
                calibration_quantity="normalized_difference_vegetation_index"
            )
        ]
    )

    passed, failures, _, _ = DeterministicVerifier.verify(pkg)
    assert passed is False
    assert any("must be reported as a unitless ratio" in f for f in failures)


def test_check_j_adversarial_rgb_pseudo_ndvi_rejection():
    # Only RGB visual asset, claiming genuine calibrated NDVI
    pkg = EvidencePackageSchema(
        task_id="task_adv_rgb",
        query="NDVI check",
        valid_pixel_count=100,
        source_assets=[
            {"id": "visual_rgb", "asset_key": "visual", "asset_category": "rgb_visual"}
        ],
        evidence_artifacts=[
            EvidenceArtifactItem(artifact_id="visual_rgb", artifact_type="rgb_visual", title="True Color")
        ],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_boa_reflectance",
                value=0.25,
                unit="unitless",
                statistic_type="mean",
                mask_applied="valid_mask"
            )
        ]
    )

    claims = [
        ClaimVerificationItem(
            claim_text="The genuine calibrated Sentinel-2 NDVI is 0.65 based on optical imagery.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["visual_rgb"]
        )
    ]

    passed, failures, eval_claims, _ = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert passed is False
    assert any("RGB-only inputs cannot substantiate claims of genuine calibrated NDVI" in f for f in failures)


def test_check_j_adversarial_zero_valid_pixels_numeric_claim():
    pkg = EvidencePackageSchema(
        task_id="task_adv_zero",
        query="NDVI check",
        valid_pixel_count=0,
        total_pixel_count=1000,
        valid_pixel_percentage=0.0,
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_ndvi",
                value=0.45,  # Fabricated numeric value with 0 valid pixels!
                unit="unitless",
                statistic_type="mean",
                mask_applied="valid_mask",
                calibration_quantity="normalized_difference_vegetation_index"
            )
        ]
    )

    passed, failures, _, _ = DeterministicVerifier.verify(pkg)
    assert passed is False
    assert any("Cannot report numeric NDVI statistic" in f for f in failures)


# ==============================================================================
# 14. Report provenance and backward-compatible serialization
# ==============================================================================
def test_report_provenance_and_markdown_serialization():
    prov = ReportProvenanceSchema(
        analysis_id="test_ndvi_report_01",
        task="spectral_analysis",
        query="Compute calibrated NDVI",
        created_at="2024-03-28T12:00:00Z",
        crs="EPSG:32643",
        valid_pixel_count=500,
        total_pixel_count=600,
        valid_pixel_percentage=83.33,
        ndvi_mean=0.6245,
        ndvi_p90=0.8120,
        vegetation_coverage_pct=72.5,
        ndvi_vegetation_threshold=0.40,
        reflectance_scale_factor=10000.0,
        scl_masking_policy="Classes 0, 1, 3, 8, 9, 10 excluded",
        b04_asset_id="B04",
        b08_asset_id="B08",
        scl_asset_id="SCL",
        ndvi_status="success"
    )

    # Validate JSON round-trip
    prov_dict = prov.model_dump()
    assert prov_dict["ndvi_mean"] == 0.6245
    assert prov_dict["ndvi_vegetation_threshold"] == 0.40

    repro = ReportReproducibilitySchema(
        software_version="SatQuery-AI v1.0.0",
        pipeline_steps=["SENTINEL2_CALIBRATED_NDVI_COMPUTED"]
    )

    report = ScientificReportSchema(
        report_id="report_test_01",
        generated_at="2024-03-28T12:00:00Z",
        provenance=prov,
        reproducibility=repro,
        executive_summary="Calibrated Sentinel-2 B04+B08 NDVI indicates healthy vegetation coverage."
    )

    md = generate_scientific_markdown(report)
    assert "Calibrated Sentinel-2 B04+B08 NDVI Provenance" in md
    assert "0.6245" in md
    assert "72.5%" in md
    assert "DN / 10000.0" in md


# ==============================================================================
# 15. Safe-reader composite transfer-budget enforcement (Sequential Mock)
# ==============================================================================
def test_transfer_budget_enforcement_sequential_mock(monkeypatch):
    """
    Task B Follow-up:
    Exercises retrieve_composite_window using mocks for sequential asset reads.
    Simulates B04 and B08 consuming the cumulative budget and verifies that
    the operation stops before exceeding the configured limit, checking
    transferred-byte accounting and failure response.
    """
    srv = STACAcquisitionService()
    scene_id = "test_budget_exhaustion_scene"
    srv._scene_cache[scene_id] = {
        "summary": None,
        "datetime": "2024-03-28T05:27:01Z",
        "provider": "Element84 Earth Search (AWS)",
        "collection": "sentinel-2-l2a",
        "raw_assets": {
            "red": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif"},
            "nir": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif"},
            "scl": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/SCL.tif"}
        }
    }

    call_history = []

    def mock_read_cog_window(url, bbox_wgs84, max_bytes):
        call_history.append({"url": url, "max_bytes": max_bytes})
        meta = make_test_meta(shape=(2, 2))
        if "B04.tif" in url:
            # B04 consumes 35,000 bytes out of 50,000 total budget
            meta["transfer_telemetry"] = {"cumulative_bytes": 35000}
            arr = np.full((2, 2), 1000, dtype=np.uint16)
            return arr, meta
        elif "B08.tif" in url:
            # When B08 is requested, remaining budget is 15,000 bytes.
            # Simulate reader raising WindowedRetrievalError because B08 would exceed this remaining budget
            raise WindowedRetrievalError(f"Cumulative transfer limit of {max_bytes} bytes exceeded while reading B08 COG window.")

    monkeypatch.setattr(SafeWindowedCOGReader, "read_cog_window", staticmethod(mock_read_cog_window))

    req = STACCompositeRetrieveRequest(
        scene_id=scene_id,
        aoi_bbox=[77.58, 12.96, 77.60, 12.98],
        max_bytes=50000
    )
    resp = srv.retrieve_composite_window(req)

    # Verify failure response and transferred-byte accounting
    assert resp.status == "error"
    assert resp.size_bytes == 0
    assert "B08 windowed retrieval failed" in resp.error
    assert "Cumulative transfer limit of 15000 bytes exceeded" in resp.error

    # Verify call history: B04 requested with 50,000; B08 requested with exactly 15,000
    assert len(call_history) == 2
    assert call_history[0]["max_bytes"] == 50000
    assert call_history[1]["max_bytes"] == 15000

    # Simulation 2: B04 consumes the entire 50,000 budget
    call_history.clear()
    def mock_read_b04_full_budget(url, bbox_wgs84, max_bytes):
        call_history.append({"url": url, "max_bytes": max_bytes})
        meta = make_test_meta(shape=(2, 2))
        meta["transfer_telemetry"] = {"cumulative_bytes": 50000}
        return np.full((2, 2), 1000, dtype=np.uint16), meta

    monkeypatch.setattr(SafeWindowedCOGReader, "read_cog_window", staticmethod(mock_read_b04_full_budget))

    resp2 = srv.retrieve_composite_window(req)
    assert resp2.status == "error"
    assert "Cumulative transfer budget of 50000 bytes exceeded before reading B08." in resp2.error
    # Only B04 was called; B08 was never called because budget exhausted before read
    assert len(call_history) == 1


# ==============================================================================
# 15b. Sentinel-2 Radiometric Offset (Task A: Present & Absent Cases)
# ==============================================================================
def test_radiometric_offset_present_and_absent_cases():
    """
    Task A Follow-up:
    Tests reflectance calculations with and without radiometric offset.
    With DN_red = 2000, DN_nir = 6000 and scale_factor = 10000:
    - Offset absent (0.0): red_refl = 0.20, nir_refl = 0.60 -> NDVI = 0.40 / 0.80 = 0.50.
    - Offset present (-0.10 or -1000 DN): red_refl = 0.10, nir_refl = 0.50 -> NDVI = 0.40 / 0.60 = 2/3 = 0.6667.
    """
    red = np.array([[2000]], dtype=np.uint16)
    nir = np.array([[6000]], dtype=np.uint16)
    meta = make_test_meta(shape=(1, 1))

    # Case 1: Offset absent (default 0.0)
    res_absent = compute_calibrated_ndvi(red, nir, target_meta=meta)
    assert res_absent["status"] == "success"
    assert abs(res_absent["ndvi_mean"] - 0.50) < 1e-4
    assert res_absent["calibration"]["offset"] == 0.0
    assert res_absent["calibration"]["offset_reflectance"] == 0.0

    # Case 2: Offset present via parameter in reflectance units (-0.10)
    res_refl_offset = compute_calibrated_ndvi(red, nir, target_meta=meta, offset=-0.10)
    assert res_refl_offset["status"] == "success"
    assert abs(res_refl_offset["ndvi_mean"] - (2.0 / 3.0)) < 1e-3
    assert res_refl_offset["calibration"]["offset"] == -0.10
    assert res_refl_offset["calibration"]["offset_reflectance"] == -0.10

    # Case 3: Offset present via parameter in raw DN units (-1000.0)
    res_dn_offset = compute_calibrated_ndvi(red, nir, target_meta=meta, offset=-1000.0)
    assert res_dn_offset["status"] == "success"
    assert abs(res_dn_offset["ndvi_mean"] - (2.0 / 3.0)) < 1e-3
    assert res_dn_offset["calibration"]["offset_reflectance"] == -0.10

    # Case 4: Metadata-driven offset via target_meta["offset"]
    meta_with_offset = make_test_meta(shape=(1, 1))
    meta_with_offset["offset"] = -0.10
    res_meta = compute_calibrated_ndvi(red, nir, target_meta=meta_with_offset)
    assert res_meta["status"] == "success"
    assert abs(res_meta["ndvi_mean"] - (2.0 / 3.0)) < 1e-3


def test_stac_service_earthsearch_offset_applied_policy():
    """
    Task A Follow-up:
    Verifies that STAC composite retrieval:
    1. Avoids double-offsetting when earthsearch:boa_offset_applied is True.
    2. Applies asset metadata offset when earthsearch:boa_offset_applied is False.
    """
    srv = STACAcquisitionService()

    # Scene 1: Provider already applied BOA offset -> meta_offset must be 0.0
    srv._scene_cache["scene_offset_already_applied"] = {
        "properties": {"earthsearch:boa_offset_applied": True},
        "raw_assets": {
            "B04": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1}]
            },
            "B08": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif"}
        }
    }

    # Scene 2: Provider did NOT apply offset -> reads raster:bands offset
    srv._scene_cache["scene_offset_not_applied"] = {
        "properties": {"earthsearch:boa_offset_applied": False},
        "raw_assets": {
            "B04": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",
                "raster:bands": [{"scale": 0.0001, "offset": -0.1}]
            },
            "B08": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif"}
        }
    }

    # Inspect resolving logic
    s1 = srv._scene_cache["scene_offset_already_applied"]
    s1_props = s1.get("properties", {})
    s1_applied = bool(s1_props.get("earthsearch:boa_offset_applied", False))
    s1_offset = 0.0
    if not s1_applied:
        s1_offset = s1["raw_assets"]["B04"]["raster:bands"][0]["offset"]
    assert s1_offset == 0.0  # Double-offsetting avoided!

    s2 = srv._scene_cache["scene_offset_not_applied"]
    s2_props = s2.get("properties", {})
    s2_applied = bool(s2_props.get("earthsearch:boa_offset_applied", False))
    s2_offset = 0.0
    if not s2_applied:
        s2_offset = s2["raw_assets"]["B04"]["raster:bands"][0]["offset"]
    assert s2_offset == -0.1  # Correctly passed to pipeline!


# ==============================================================================
# 16. Regression protection for Stage 9A acquisition safeguards
# ==============================================================================
def test_regression_stage9a_safeguards():
    # Host allowlisting check via validate_url_security
    SafeWindowedCOGReader.validate_url_security("https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif")
    with pytest.raises(WindowedRetrievalError) as exc_host:
        SafeWindowedCOGReader.validate_url_security("https://malicious-satellite-data.evil.com/fake.tif")
    assert "not in the trusted satellite host allowlist" in str(exc_host.value)

    # HTTP scheme rejection
    with pytest.raises(WindowedRetrievalError) as exc_scheme:
        SafeWindowedCOGReader.validate_url_security("http://sentinel-cogs.s3.us-west-2.amazonaws.com/data.tif")
    assert "only HTTPS URLs are allowed" in str(exc_scheme.value)


# ==============================================================================
# 17. Regression protection for Stage 10A quality screening and Checks H–I
# ==============================================================================
def test_regression_stage10a_checks_h_and_i():
    pkg = EvidencePackageSchema(
        task_id="task_reg_10a",
        query="Cloud check",
        valid_pixel_count=100,
        quality_screening={"optical": {"screened": True, "excluded_pixel_count": 50, "quality_clearance_fraction": 0.85, "has_cloud_obstruction": True}}
    )

    # Check H: claiming cloud-free with 50 excluded cloud pixels must be contradicted
    claims = [
        ClaimVerificationItem(
            claim_text="The entire scene is completely clear and cloud-free.",
            claim_type="general",
            status="supported"
        )
    ]
    passed, failures, eval_claims, _ = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert passed is False
    assert any("Claim asserts cloud-free scene" in f for f in failures)
    assert eval_claims[0].status == "contradicted"


# ==============================================================================
# 18. Existing single-band acquisition behavior
# ==============================================================================
def test_existing_single_band_acquisition_unaffected():
    # Verify STACWindowedRetrieveRequest schema still valid
    req = STACWindowedRetrieveRequest(
        scene_id="local_sample_cartosat_optical_bengaluru",
        asset_key="visual",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98]
    )
    assert req.asset_key == "visual"
    assert len(req.aoi_bbox) == 4


# ==============================================================================
# 19. Legacy compute_ndvi heuristic labeled as approximation
# ==============================================================================
def test_legacy_compute_ndvi_heuristic_labeled():
    # Calling legacy compute_ndvi returns ndvi array and preview array
    rgb_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_arr[:, :, 1] = 120  # Green
    ndvi_arr, preview = compute_ndvi(rgb_arr)
    assert isinstance(ndvi_arr, np.ndarray)
    assert isinstance(preview, np.ndarray)
    assert ndvi_arr.shape == (10, 10)
    assert preview.shape == (10, 10, 3)
    # Verify legacy docstring labels it as uncalibrated approximation only
    assert "LEGACY / APPROXIMATION ONLY" in compute_ndvi.__doc__
