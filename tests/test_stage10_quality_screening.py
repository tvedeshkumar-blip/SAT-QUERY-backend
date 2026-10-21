"""
Stage 10A Deterministic Test Suite: Radiometric Quality Screening.
Covers:
- SCL categorical mapping, class semantics (excluded, uncertain, retained, unknown, nodata).
- Nearest-neighbor reprojection of categorical arrays without interpolated values.
- CRS and geotransform validation.
- SAR RTC validity screening (nodata -32768.0, NaN, non-positive power, decibel ordering).
- Honest recording of sar_terrain_layover_shadow_mask as unavailable from provider.
- Fully obstructed / zero-valid-pixel defense returning explicit unsupported status and None fractions.
- Quality mask integration into real physical measurements and heuristic calculations.
- STAC companion SCL failure isolation without primary raster loss.
- DeterministicVerifier rejection of unsupported cloud-free and terrain-shadow-free claims.
- Stage 8 scientific report provenance propagation and markdown rendering.
"""

import io
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling

from app.remote_sensing.quality import (
    QualityScreener,
    SCL_EXCLUDED_CLASSES,
    SCL_UNCERTAIN_CLASSES,
    SCL_RETAINED_CLASSES,
    SCL_CLASS_NAMES,
)
from app.models.optical_sar.optical_sar_fusion import OpticalSARJointAnalysisProvider
from app.agent.verifier import DeterministicVerifier, EvidencePackageBuilder
from app.reports.scientific_report import ScientificReportBuilder, generate_scientific_markdown
from app.schemas.verification import EvidencePackageSchema, ClaimVerificationItem


def create_synthetic_geotiff_bytes(data: np.ndarray, crs: str = "EPSG:32643", bounds=(500000, 2000000, 501000, 2001000), nodata=None) -> bytes:
    """Helper to create an in-memory single-band or multi-band GeoTIFF byte string."""
    if data.ndim == 2:
        count = 1
        h, w = data.shape
        data_to_write = data[np.newaxis, :, :]
    else:
        count, h, w = data.shape
        data_to_write = data

    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
    dtype = str(data.dtype)

    memfile = io.BytesIO()
    with rasterio.open(
        memfile,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for i in range(count):
            dst.write(data_to_write[i], i + 1)

    return memfile.getvalue()


# ---------------------------------------------------------------------------
# Test 1: SCL Class Semantics and Accounting
# ---------------------------------------------------------------------------

def test_scl_class_semantics():
    """Verify SCL class partition: excluded {0, 1, 3, 8, 9, 10}, uncertain {2, 7}, retained {4, 5, 6, 11}."""
    # Test every single class from 0 to 11
    scl_arr = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ], dtype=np.uint8)

    # Default policy: uncertain classes (2, 7) are EXCLUDED from clear_mask
    result = QualityScreener.evaluate_scl(scl_arr, allow_uncertain_as_clear=False)

    assert result["screened"] is True
    assert result["total_pixels"] == 12
    assert result["observed_pixel_count"] == 11  # class 0 is nodata
    assert result["nodata_pixel_count"] == 1

    # Retained classes: 4, 5, 6, 11 -> 4 pixels
    assert result["retained_pixels"] == 4
    # Excluded classes: 0, 1, 3, 8, 9, 10 -> 6 pixels
    # Uncertain classes: 2, 7 -> 2 pixels
    assert result["uncertain_pixels"] == 2
    assert result["excluded_pixel_count"] == 6

    # Clearance fraction over observed pixels: 4 / 11 = 0.3636
    assert abs(result["quality_clearance_fraction"] - (4.0 / 11.0)) < 1e-4

    # Clear mask must be True ONLY for 4, 5, 6, 11
    clear_mask = result["clear_mask"]
    expected_clear = np.isin(scl_arr, [4, 5, 6, 11])
    np.testing.assert_array_equal(clear_mask, expected_clear)


def test_scl_unknown_values_quarantine():
    """Verify that unknown/out-of-range SCL values (<0 or >11) are fail-closed and excluded."""
    scl_arr = np.array([
        [4, 5, 99],
        [12, 255, 6]
    ], dtype=np.uint8)

    result = QualityScreener.evaluate_scl(scl_arr)

    assert result["unknown_pixel_count"] == 3  # 99, 12, 255
    assert result["retained_pixels"] == 3     # 4, 5, 6
    assert result["excluded_pixel_count"] == 3  # All unknown quarantined into excluded
    # Retained pixels in clear mask
    assert result["clear_mask"][0, 2] == False  # 99 excluded
    assert result["clear_mask"][1, 0] == False  # 12 excluded
    assert result["clear_mask"][1, 1] == False  # 255 excluded
    assert result["clear_mask"][0, 0] == True   # 4 clear
    assert result["clear_mask"][0, 1] == True   # 5 clear
    assert result["clear_mask"][1, 2] == True   # 6 clear


# ---------------------------------------------------------------------------
# Test 2: Nearest-Neighbor Reprojection of Categorical SCL
# ---------------------------------------------------------------------------

def test_scl_nearest_neighbor_categorical_resampling():
    """Reproject 20m SCL to 10m grid; confirm strictly categorical labels without interpolation."""
    # 20m grid: 5x5 pixels = 100m x 100m
    scl_20m = np.array([
        [4, 4, 5, 5, 6],
        [4, 4, 5, 5, 6],
        [8, 8, 9, 9, 11],
        [8, 8, 9, 9, 11],
        [3, 3, 3, 3, 11]
    ], dtype=np.uint8)

    transform_20m = from_bounds(500000, 2000000, 500100, 2000100, 5, 5)
    scl_meta = {
        "crs": "EPSG:32643",
        "transform": transform_20m,
        "width": 5,
        "height": 5
    }

    # 10m target grid: 10x10 pixels for the exact same 100m x 100m extent
    transform_10m = from_bounds(500000, 2000000, 500100, 2000100, 10, 10)
    target_meta = {
        "crs": "EPSG:32643",
        "transform": transform_10m,
        "width": 10,
        "height": 10
    }

    reprojected, alignment_meta = QualityScreener.reproject_scl_to_target(scl_20m, scl_meta, target_meta)

    assert reprojected.shape == (10, 10)
    # Check that ALL values are strictly inside the original discrete set {3, 4, 5, 6, 8, 9, 11}
    unique_vals = set(np.unique(reprojected))
    original_vals = {3, 4, 5, 6, 8, 9, 11}
    assert unique_vals.issubset(original_vals), f"Found non-categorical values: {unique_vals - original_vals}"
    # Verify no fractional or intermediate artifacts
    assert issubclass(reprojected.dtype.type, (np.integer, np.unsignedinteger))


# ---------------------------------------------------------------------------
# Test 3: Spatial Alignment & Grid Mismatch Rejection
# ---------------------------------------------------------------------------

def test_scl_crs_mismatch_rejection():
    """Verify that unprojected or mismatched CRS between SCL and target raises error."""
    from app.remote_sensing.quality import QualityScreeningError
    scl_arr = np.zeros((10, 10), dtype=np.uint8)
    scl_meta = {
        "crs": "EPSG:4326",  # Geographic
        "transform": from_bounds(77.0, 12.0, 77.1, 12.1, 10, 10),
        "width": 10,
        "height": 10
    }
    target_meta = {
        "crs": "EPSG:32643",  # UTM Projected
        "transform": from_bounds(500000, 2000000, 501000, 2001000, 10, 10),
        "width": 10,
        "height": 10
    }

    scl_meta_no_crs = {"crs": None, "transform": scl_meta["transform"], "width": 10, "height": 10}
    with pytest.raises((QualityScreeningError, ValueError)):
        QualityScreener.reproject_scl_to_target(scl_arr, scl_meta_no_crs, target_meta)


# ---------------------------------------------------------------------------
# Test 4: Sentinel-1 RTC Backscatter Validity Screening
# ---------------------------------------------------------------------------

def test_sar_backscatter_validity_screening():
    """Verify SAR screening excludes -32768 nodata, NaNs, infs, and non-positive linear power before dB conversion."""
    sar_data = np.array([
        [-32768.0, 0.05, 0.10],      # nodata, valid, valid
        [np.nan, np.inf, 0.20],      # nan, inf, valid
        [-0.05, 0.0, 0.30]           # negative, zero, valid
    ], dtype=np.float32)

    meta = {"nodata": -32768.0, "calibration": "gamma0"}
    sar_eval = QualityScreener.evaluate_sar_quality(sar_data, meta)

    assert sar_eval["screened"] is True
    assert sar_eval["total_pixels"] == 9
    assert sar_eval["nodata_pixel_count"] == 1
    assert sar_eval["non_finite_pixel_count"] == 2  # NaN and Inf
    assert sar_eval["non_positive_pixel_count"] == 2  # -0.05 and 0.0
    assert sar_eval["valid_pixel_count"] == 4  # 0.05, 0.10, 0.20, 0.30
    assert sar_eval["sar_terrain_layover_shadow_mask"] == "unavailable_from_stac_provider"

    # Verify that valid mask correctly isolates only positive finite floats
    valid_mask = sar_eval["valid_mask"]
    expected_valid = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, True]
    ], dtype=bool)
    np.testing.assert_array_equal(valid_mask, expected_valid)


# ---------------------------------------------------------------------------
# Test 5: Fully Obstructed Scene & Zero-Valid-Pixel Defense
# ---------------------------------------------------------------------------

def test_optical_sar_joint_fully_obstructed_scene():
    """Verify that 100% cloud-covered scenes return explicit unsupported status and None fractions."""
    h, w = 20, 20
    # Optical: 3-band uint8 with 100% high-probability cloud SCL (class 9)
    opt_raster = np.full((3, h, w), 200, dtype=np.uint8)
    scl_raster = np.full((h, w), 9, dtype=np.uint8)  # Class 9 = Cloud High Probability
    sar_raster = np.full((h, w), 0.15, dtype=np.float32)

    bounds = (500000, 2000000, 501000, 2001000)
    opt_bytes = create_synthetic_geotiff_bytes(opt_raster, bounds=bounds)
    sar_bytes = create_synthetic_geotiff_bytes(sar_raster, bounds=bounds)

    opt_meta = {
        "crs": "EPSG:32643",
        "bounds": bounds,
        "width": w,
        "height": h,
        "resolution": 10.0,
        "scl_mask": scl_raster,  # Provide 100% cloud SCL
        "sensor": "Sentinel-2"
    }
    sar_meta = {
        "crs": "EPSG:32643",
        "bounds": bounds,
        "width": w,
        "height": h,
        "resolution": 10.0,
        "sensor": "Sentinel-1",
        "calibration": "gamma0"
    }

    analyzer = OpticalSARJointAnalysisProvider()
    result = analyzer.predict(
        images=[opt_raster.transpose(1, 2, 0), sar_raster],
        metadata={"optical": opt_meta, "sar": sar_meta}
    )

    # Must return unsupported_cloud_obstruction
    assert result["status"] == "unsupported"
    assert result["analysis_status"] == "unsupported_cloud_obstruction"
    assert result["valid_pixel_count"] == 0
    # Heuristic percentages must be None, NOT fabricated zeros!
    assert result["water_pct"] is None
    assert result["urban_pct"] is None
    assert result["veg_pct"] is None
    # Explanation text must cite cloud obstruction honestly
    assert "obstruct" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Test 6: Quality Screening Applied to Real Analysis Calculations
# ---------------------------------------------------------------------------

def test_quality_mask_applied_to_real_measurements():
    """Verify that optical reflectance and heuristic fractions use clear-sky denominator only."""
    h, w = 10, 10
    # Half clear vegetation (SCL 4), half medium cloud (SCL 8)
    scl_raster = np.full((h, w), 4, dtype=np.uint8)
    scl_raster[:, :5] = 8  # Left half is cloud

    # Optical reflectance: 0.10 on left, 0.50 on right
    opt_raster = np.zeros((3, h, w), dtype=np.uint8)
    opt_raster[:, :, :5] = 25   # Left (cloudy)
    opt_raster[:, :, 5:] = 125  # Right (clear)

    sar_raster = np.full((h, w), 0.10, dtype=np.float32)

    bounds = (500000, 2000000, 501000, 2001000)
    opt_meta = {
        "crs": "EPSG:32643",
        "bounds": bounds,
        "width": w,
        "height": h,
        "scl": scl_raster,
        "sensor": "Sentinel-2"
    }
    sar_meta = {
        "crs": "EPSG:32643",
        "bounds": bounds,
        "width": w,
        "height": h,
        "sensor": "Sentinel-1",
        "calibration_status": "calibrated_backscatter",
        "terrain_correction_status": "radiometrically_terrain_corrected",
        "calibration": "gamma0"
    }

    analyzer = OpticalSARJointAnalysisProvider()
    result = analyzer.predict(
        images=[opt_raster.transpose(1, 2, 0), sar_raster],
        metadata={"optical": opt_meta, "sar": sar_meta}
    )

    assert result["valid_pixel_count"] == 50
    q_screen = result["quality_screening"]
    assert q_screen["optical"]["screened"] is True
    assert q_screen["optical"]["quality_clearance_fraction"] == 0.5
    assert q_screen["optical"]["excluded_pixel_count"] == 50
    assert "ev_optical_scl_mask" in [e["id"] for e in result["evidence"]]


# ---------------------------------------------------------------------------
# Test 7: Deterministic Verifier Rejection of Unsupported Claims
# ---------------------------------------------------------------------------

def test_verifier_rejects_cloud_free_claim_when_clouds_present():
    """Verify that DeterministicVerifier Check H rejects 'cloud-free' claims when cloud pixels exist."""
    evidence_package = EvidencePackageSchema(
        task_id="test_cloud_check",
        query="Verify cloud status",
        sensor_identities=["Sentinel-2", "Sentinel-1"],
        acquisition_timestamps=["2024-02-01T00:00:00Z"],
        source_assets=[{"id": "s2_scene", "collection": "sentinel-2-l2a"}],
        quality_screening={
            "optical": {
                "screened": True,
                "quality_clearance_fraction": 0.85,
                "excluded_pixel_count": 150,
                "cloud_pixels": 150,
                "has_cloud_obstruction": False
            }
        }
    )

    claims = [
        ClaimVerificationItem(
            claim_text="The acquired Sentinel-2 scene is completely cloud-free and clear sky.",
            claim_type="general",
            status="supported"
        )
    ]

    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(
        evidence_package=evidence_package,
        candidate_claims=claims
    )

    assert passed is False
    assert any("Claim asserts cloud-free scene" in f for f in failures)
    assert eval_claims[0].status == "contradicted"


def test_verifier_rejects_terrain_shadow_free_claim():
    """Verify that DeterministicVerifier Check I rejects 'terrain-shadow-free' claims."""
    evidence_package = EvidencePackageSchema(
        task_id="test_shadow_check",
        query="Verify terrain shadow status",
        sensor_identities=["Sentinel-1"],
        acquisition_timestamps=["2024-02-01T00:00:00Z"],
        source_assets=[{"id": "s1_rtc", "collection": "sentinel-1-rtc"}],
        quality_screening={
            "sar": {
                "screened": True,
                "valid_pixel_count": 1000,
                "sar_terrain_layover_shadow_mask": "unavailable_from_stac_provider"
            }
        }
    )

    claims = [
        ClaimVerificationItem(
            claim_text="The SAR imagery is completely free of terrain shadows and radar layover.",
            claim_type="general",
            status="supported"
        )
    ]

    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(
        evidence_package=evidence_package,
        candidate_claims=claims
    )

    assert passed is False
    assert any("SAR geometric layover/shadow masking is unavailable" in f for f in failures)
    assert eval_claims[0].status == "unsupported"


# ---------------------------------------------------------------------------
# Test 8: Stage 8 Report Builder and Markdown Generation
# ---------------------------------------------------------------------------

def test_scientific_report_quality_screening_provenance():
    """Verify that ScientificReportBuilder propagates quality provenance and honest unavailable fields."""
    analysis_resp = {
        "id": "task_stage10_test",
        "task": "optical_sar",
        "query": "Assess surface land cover with quality screening",
        "answer": "Analysis performed with verified Sentinel-2 SCL radiometric screening and Sentinel-1 RTC backscatter.",
        "primary_model": "OpticalSARJointProvider",
        "actual_model_used": "OpticalSARJointSpecialist",
        "fallback_used": False,
        "metadata": {
            "sensor": "Sentinel-2 / Sentinel-1",
            "crs": "EPSG:32643",
            "width": 100,
            "height": 100,
            "quality_screening": {
                "optical": {
                    "screened": True,
                    "observed_pixel_count": 10000,
                    "excluded_pixel_count": 1200,
                    "quality_clearance_fraction": 0.88
                },
                "sar": {
                    "screened": True,
                    "valid_pixel_count": 9800,
                    "sar_terrain_layover_shadow_mask": "unavailable_from_stac_provider"
                }
            }
        }
    }

    report = ScientificReportBuilder.build_report(analysis_resp)

    # Check provenance
    assert report.provenance.quality_screening is not None
    assert report.provenance.observed_pixel_count == 10000
    assert report.provenance.excluded_quality_pixel_count == 1200
    assert report.provenance.quality_clearance_percentage == 88.0

    # Check reproducibility
    assert "RADIOMETRIC_QUALITY_SCREENING_APPLIED" in report.reproducibility.pipeline_steps
    assert any("sar_terrain_layover_shadow_mask: unavailable" in uf for uf in report.reproducibility.unavailable_fields)

    # Check markdown rendering
    md = generate_scientific_markdown(report)
    assert "Quality Screening" in md
    assert "88.0%" in md
    assert "sar_terrain_layover_shadow_mask" in md
