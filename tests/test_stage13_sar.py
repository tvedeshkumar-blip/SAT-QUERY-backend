"""
Stage 13 Test Suite: Sentinel-1 Radiometrically Terrain-Corrected (RTC) Dual-Polarization Analysis.

Validates:
1. Exact mathematical formulation of DpRVI (Mandal et al., 2020) and cross-pol ratio.
2. Nodata sentinel (-32768.0) and non-finite float masking.
3. Non-positive linear backscatter power screening prior to logarithms.
4. Input decibel (dB) to linear power conversion.
5. True radiometric average (10*log10(mean linear)) vs mean of pixel-wise dB.
6. Zero-valid-pixel defense: Suppressing statistics (returning None, never fabricating numeric 0).
7. Reprojection and spatial grid alignment between VV and VH channels.
8. Same STAC Item retrieval and cumulative transfer budget (<= 20 MB) enforcement.
9. Deterministic Verifier Check K (Package-Level & Claim-Level adversarial checks).
10. Evidence package extraction and Scientific Report Provenance rendering.
11. Local sample composite SAR window retrieval.
"""

import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.remote_sensing.sar import (
    compute_calibrated_dual_pol_sar,
    DPRVI_FORMULA_REFERENCE,
    SAR_RTC_DEFAULT_NODATA,
)
from app.remote_sensing.quality import QualityScreener
from app.agent.verifier import DeterministicVerifier, EvidencePackageBuilder
from app.schemas.verification import (
    EvidencePackageSchema,
    PhysicalMeasurementItem,
    ClaimVerificationItem,
    EvidenceArtifactItem,
)
from app.schemas.acquisition import (
    STACCompositeRetrieveRequest,
    STACCompositeRetrieveResponse,
)
from app.schemas.report import ReportProvenanceSchema
from app.reports.scientific_report import ScientificReportBuilder, generate_scientific_markdown
from app.acquisition.stac_service import STACAcquisitionService
from app.acquisition.windowed_reader import WindowedRetrievalError


# ---------------------------------------------------------------------------
# 1. Exact Mathematical Calculations & DpRVI Bounds
# ---------------------------------------------------------------------------

def test_01_exact_dprvi_and_cross_pol_ratio():
    """
    Given: VV = 0.20, VH = 0.05
    q = VH / VV = 0.05 / 0.20 = 0.25
    DpRVI = (q * (q + 3)) / ((q + 1) ** 2) = (0.25 * 3.25) / (1.25 ** 2) = 0.8125 / 1.5625 = 0.5200
    q_dB = 10 * log10(0.25) = -6.02 dB
    VV_dB = 10 * log10(0.20) = -6.99 dB
    VH_dB = 10 * log10(0.05) = -13.01 dB
    """
    vv = np.full((10, 10), 0.20, dtype=np.float32)
    vh = np.full((10, 10), 0.05, dtype=np.float32)

    res = compute_calibrated_dual_pol_sar(vv_arr=vv, vh_arr=vh)

    assert res["status"] == "success"
    assert res["is_calibrated"] is True
    assert math.isclose(res["dprvi_mean"], 0.5200, abs_tol=1e-4)
    assert math.isclose(res["dprvi_p90"], 0.5200, abs_tol=1e-4)
    assert math.isclose(res["cross_pol_ratio_linear"], 0.2500, abs_tol=1e-4)
    assert math.isclose(res["cross_pol_ratio_db"], -6.02, abs_tol=1e-2)
    assert math.isclose(res["vv_mean_db"], -6.99, abs_tol=1e-2)
    assert math.isclose(res["vh_mean_db"], -13.01, abs_tol=1e-2)
    assert res["valid_pixel_count"] == 100
    assert res["masked_pixel_count"] == 0


def test_02_dprvi_boundary_extremes():
    """
    Test specular surface (q -> 0.0 => DpRVI -> 0.0)
    and volume scattering saturation (q -> 1.0 => DpRVI -> 1.0).
    """
    # Specular / water: very low VH
    vv_spec = np.full((4, 4), 0.10, dtype=np.float32)
    vh_spec = np.full((4, 4), 0.00001, dtype=np.float32)
    res_spec = compute_calibrated_dual_pol_sar(vv_arr=vv_spec, vh_arr=vh_spec)
    assert res_spec["dprvi_mean"] < 0.01

    # Dense volume scattering: q = 1.0
    vv_dense = np.full((4, 4), 0.08, dtype=np.float32)
    vh_dense = np.full((4, 4), 0.08, dtype=np.float32)
    res_dense = compute_calibrated_dual_pol_sar(vv_arr=vv_dense, vh_arr=vh_dense)
    # (1 * 4) / (2 ^ 2) = 1.0
    assert math.isclose(res_dense["dprvi_mean"], 1.0000, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# 2. Nodata Sentinels & Non-Positive Power Screening
# ---------------------------------------------------------------------------

def test_03_nodata_sentinels_and_non_finite_masking():
    """
    Verifies that -32768.0, NaN, Inf, and non-positive linear power values
    are completely excluded and not treated as physical measurements.
    """
    vv = np.array([
        [0.20, -32768.0],
        [np.nan, 0.0],
        [0.20, np.inf]
    ], dtype=np.float32)

    vh = np.array([
        [0.05, 0.05],
        [0.05, 0.05],
        [-32768.0, 0.05]
    ], dtype=np.float32)

    res = compute_calibrated_dual_pol_sar(vv_arr=vv, vh_arr=vh)

    assert res["total_pixel_count"] == 6
    # Only pixel (0, 0) is valid: vv=0.20, vh=0.05
    assert res["valid_pixel_count"] == 1
    assert res["masked_pixel_count"] == 5
    assert math.isclose(res["dprvi_mean"], 0.5200, abs_tol=1e-4)


def test_04_zero_valid_pixel_defense_returns_none():
    """
    CRITICAL: Never fabricate numeric values when valid pixels are absent.
    All statistics must be None, NOT 0.0.
    """
    vv = np.full((8, 8), -32768.0, dtype=np.float32)
    vh = np.full((8, 8), -32768.0, dtype=np.float32)

    res = compute_calibrated_dual_pol_sar(vv_arr=vv, vh_arr=vh)

    assert res["status"] == "unsupported_zero_valid_pixels"
    assert res["valid_pixel_count"] == 0
    assert res["dprvi_mean"] is None
    assert res["dprvi_p90"] is None
    assert res["dprvi_median"] is None
    assert res["dprvi_min"] is None
    assert res["dprvi_max"] is None
    assert res["cross_pol_ratio_linear"] is None
    assert res["cross_pol_ratio_db"] is None
    assert res["vv_mean_linear"] is None
    assert res["vh_mean_linear"] is None
    assert res["vv_mean_db"] is None
    assert res["vh_mean_db"] is None
    assert res["vegetation_proxy_coverage_pct"] is None


def test_05_input_is_db_conversion():
    """
    Tests input in decibels (dB):
    VV = -10.0 dB (0.1 linear)
    VH = -16.02 dB (0.025 linear)
    q = 0.25 => DpRVI = 0.5200
    """
    vv_db = np.full((4, 4), -10.0, dtype=np.float32)
    vh_db = np.full((4, 4), -16.02, dtype=np.float32)

    res = compute_calibrated_dual_pol_sar(vv_arr=vv_db, vh_arr=vh_db, input_is_db=True)

    assert res["status"] == "success"
    assert math.isclose(res["dprvi_mean"], 0.5200, abs_tol=1e-3)
    assert math.isclose(res["cross_pol_ratio_linear"], 0.2500, abs_tol=1e-3)


def test_06_radiometric_average_vs_pixel_wise_db():
    """
    Verifies that mean linear power in dB (10*log10(mean linear))
    is mathematically distinguished from the mean of pixel-wise dB.
    """
    # Linear powers: 0.1 (-10 dB) and 1.0 (0 dB)
    # Mean linear power = 0.55 => 10*log10(0.55) = -2.60 dB
    # Mean pixel-wise dB = (-10 + 0) / 2 = -5.0 dB
    vv = np.array([[0.1, 1.0]], dtype=np.float32)
    vh = np.array([[0.025, 0.25]], dtype=np.float32)

    res = compute_calibrated_dual_pol_sar(vv_arr=vv, vh_arr=vh)

    assert math.isclose(res["vv_mean_db"], -2.60, abs_tol=1e-2)
    assert math.isclose(res["vv_pixel_db_mean"], -5.00, abs_tol=1e-2)
    assert res["vv_mean_db"] != res["vv_pixel_db_mean"]


def test_07_grid_alignment_reprojection():
    """
    Verifies that mismatched grid dimensions between VV and VH are aligned.
    """
    from rasterio.transform import from_origin

    vv = np.full((20, 20), 0.20, dtype=np.float32)
    vh = np.full((10, 10), 0.05, dtype=np.float32)

    vv_meta = {
        "crs": "EPSG:32643",
        "transform": list(from_origin(500000, 1400000, 10, 10)[:6]),
        "nodata": -32768.0
    }
    vh_meta = {
        "crs": "EPSG:32643",
        "transform": list(from_origin(500000, 1400000, 20, 20)[:6]),
        "nodata": -32768.0
    }

    res = compute_calibrated_dual_pol_sar(
        vv_arr=vv,
        vh_arr=vh,
        vv_meta=vv_meta,
        vh_meta=vh_meta
    )

    assert res["status"] == "success"
    assert res["valid_pixel_count"] == 400


# ---------------------------------------------------------------------------
# 3. Deterministic Verifier Check K (Package-Level & Claim-Level)
# ---------------------------------------------------------------------------

def test_08_verifier_check_k_package_level_valid():
    """
    Valid DpRVI measurement in [0.0, 1.0] with declared unitless ratio passes.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_val",
        query="Analyze SAR vegetation canopy",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv", "polarization": "VV"}, {"id": "s1_vh", "polarization": "VH"}],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_dprvi",
                value=0.5200,
                unit="unitless ratio [0, 1]",
                statistic_type="mean",
                mask_applied="sar_rtc_valid_mask",
                calibration_quantity="dual_polarimetric_radar_vegetation_index"
            )
        ],
        valid_pixel_count=1000,
        total_pixel_count=1000
    )

    passed, failures, _, _ = DeterministicVerifier.verify(evidence)
    assert passed is True
    assert len(failures) == 0


def test_09_verifier_check_k_package_level_out_of_bounds():
    """
    DpRVI > 1.0 or < 0.0 must trigger deterministic verification failure.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_invalid",
        query="Analyze SAR vegetation",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv"}, {"id": "s1_vh"}],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_dprvi",
                value=1.4500,
                unit="unitless ratio [0, 1]",
                statistic_type="mean",
                mask_applied="sar_rtc_valid_mask",
                calibration_quantity="dual_polarimetric_radar_vegetation_index"
            )
        ],
        valid_pixel_count=1000,
        total_pixel_count=1000
    )

    passed, failures, _, _ = DeterministicVerifier.verify(evidence)
    assert passed is False
    assert any("outside valid physical range [0.0, 1.0]" in f for f in failures)


def test_10_verifier_check_k_package_level_invalid_units():
    """
    DpRVI reported in % or dB must trigger deterministic verification failure.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_unit",
        query="Analyze SAR vegetation",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv"}, {"id": "s1_vh"}],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_dprvi",
                value=0.52,
                unit="dB",
                statistic_type="mean",
                mask_applied="sar_rtc_valid_mask",
                calibration_quantity="dual_polarimetric_radar_vegetation_index"
            )
        ],
        valid_pixel_count=1000,
        total_pixel_count=1000
    )

    passed, failures, _, _ = DeterministicVerifier.verify(evidence)
    assert passed is False
    assert any("must be reported as a unitless ratio [0, 1]" in f for f in failures)


def test_11_verifier_check_k_package_level_zero_valid_pixels_defense():
    """
    Cannot report numeric DpRVI when valid_pixel_count is zero.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_zero_px",
        query="Analyze SAR vegetation",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv"}, {"id": "s1_vh"}],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_dprvi",
                value=0.52,
                unit="unitless ratio [0, 1]",
                statistic_type="mean",
                mask_applied="sar_rtc_valid_mask",
                calibration_quantity="dual_polarimetric_radar_vegetation_index"
            )
        ],
        valid_pixel_count=0,
        total_pixel_count=1000
    )

    passed, failures, _, _ = DeterministicVerifier.verify(evidence)
    assert passed is False
    assert any("when valid pixel count is zero" in f for f in failures)


def test_12_verifier_check_k_claim_level_out_of_bounds():
    """
    Candidate claim asserting out-of-bounds DpRVI is contradicted.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_c1",
        query="Analyze SAR vegetation",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv", "polarization": "VV"}, {"id": "s1_vh", "polarization": "VH"}],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_dprvi",
                value=0.52,
                unit="unitless ratio [0, 1]",
                statistic_type="mean",
                mask_applied="sar_rtc_valid_mask",
                calibration_quantity="dual_polarimetric_radar_vegetation_index"
            )
        ],
        valid_pixel_count=100,
        total_pixel_count=100
    )

    claim = ClaimVerificationItem(
        claim_text="Sentinel-1 recorded a DpRVI value of 1.45 across the scene.",
        claim_type="measurement",
        status="supported"
    )

    passed, failures, claims, contradictions = DeterministicVerifier.verify(
        evidence_package=evidence,
        candidate_claims=[claim]
    )

    assert passed is False
    assert claims[0].status == "contradicted"
    assert "outside physical bounds [0.0, 1.0]" in claims[0].reason


def test_13_verifier_check_k_claim_level_single_pol_rejection():
    """
    Claim asserting DpRVI on single-polarization asset (VV only) is rejected as unsupported.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_c2",
        query="Analyze SAR vegetation",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv", "polarization": "VV"}],  # Only VV!
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_linear_power",
                value=0.20,
                unit="linear power (m^2/m^2)",
                statistic_type="mean",
                mask_applied="valid_pixel_mask",
                calibration_quantity="gamma0"
            )
        ],
        valid_pixel_count=100,
        total_pixel_count=100
    )

    claim = ClaimVerificationItem(
        claim_text="Sentinel-1 dual-polarimetric radar vegetation index DpRVI is 0.52.",
        claim_type="measurement",
        status="supported"
    )

    passed, failures, claims, _ = DeterministicVerifier.verify(
        evidence_package=evidence,
        candidate_claims=[claim]
    )

    assert passed is False
    assert claims[0].status == "unsupported"
    assert "requires evidence and provenance for both VV and VH channels" in claims[0].reason


def test_14_verifier_check_k_claim_level_chlorophyll_conflation():
    """
    Claim asserting DpRVI directly measures optical chlorophyll is contradicted.
    """
    evidence = EvidencePackageSchema(
        task_id="task_sar_c3",
        query="Analyze SAR vegetation",
        sensor_identities=["Sentinel-1 RTC"],
        source_assets=[{"id": "s1_vv", "polarization": "VV"}, {"id": "s1_vh", "polarization": "VH"}],
        physical_measurements=[
            PhysicalMeasurementItem(
                name="sar_mean_dprvi",
                value=0.52,
                unit="unitless ratio [0, 1]",
                statistic_type="mean",
                mask_applied="sar_rtc_valid_mask",
                calibration_quantity="dual_polarimetric_radar_vegetation_index"
            )
        ],
        valid_pixel_count=100,
        total_pixel_count=100
    )

    claim = ClaimVerificationItem(
        claim_text="Sentinel-1 DpRVI confirms high photosynthetic chlorophyll concentration.",
        claim_type="interpretation",
        status="supported"
    )

    passed, failures, claims, _ = DeterministicVerifier.verify(
        evidence_package=evidence,
        candidate_claims=[claim]
    )

    assert passed is False
    assert claims[0].status == "contradicted"
    assert "cannot be claimed as a direct measure of optical chlorophyll" in claims[0].reason


# ---------------------------------------------------------------------------
# 4. Composite STAC Retrieval & Transfer Budget Cap Enforcement
# ---------------------------------------------------------------------------

def test_15_stac_composite_sar_retrieval_success():
    """
    Tests STACCompositeRetrieveRequest for Sentinel-1 RTC dual-pol windowed retrieval.
    """
    service = STACAcquisitionService()
    service._scene_cache["S1A_RTC_TEST"] = {
        "raw_assets": {
            "vv": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/s1/vv.tif"},
            "vh": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/s1/vh.tif"}
        },
        "collection": "sentinel-1-rtc",
        "modality": "SAR"
    }

    dummy_meta = {
        "crs": "EPSG:4326",
        "transform": [0.0001, 0, 77.5, 0, -0.0001, 13.0],
        "nodata": -32768.0,
        "transfer_telemetry": {"cumulative_bytes": 1024}
    }

    req = STACCompositeRetrieveRequest(
        scene_id="S1A_RTC_TEST",
        aoi_bbox=[77.50, 12.95, 77.55, 13.00],
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        compute_dprvi=True
    )

    with patch("app.acquisition.windowed_reader.SafeWindowedCOGReader.read_cog_window") as mock_read:
        mock_read.side_effect = [
            (np.full((16, 16), 0.20, dtype=np.float32), dummy_meta),
            (np.full((16, 16), 0.05, dtype=np.float32), dummy_meta)
        ]
        resp = service.retrieve_composite_window(req)

    assert resp.status == "success"
    assert resp.category == "scientific_sar"
    assert resp.dprvi_result is not None
    assert math.isclose(resp.dprvi_result["dprvi_mean"], 0.5200, abs_tol=1e-4)
    assert resp.sar_quality_screening is not None
    assert resp.provenance["polarization"] == "DUAL_VV_VH"


def test_16_stac_composite_sar_missing_asset_rejection():
    """
    Tests that missing VH asset is rejected with explicit descriptive error.
    """
    service = STACAcquisitionService()
    service._scene_cache["S1A_RTC_INCOMPLETE"] = {
        "raw_assets": {
            "vv": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/s1/vv.tif"}
            # Missing VH!
        },
        "collection": "sentinel-1-rtc"
    }

    req = STACCompositeRetrieveRequest(
        scene_id="S1A_RTC_INCOMPLETE",
        aoi_bbox=[77.50, 12.95, 77.55, 13.00],
        collection="sentinel-1-rtc",
        bands=["vv", "vh"]
    )

    resp = service.retrieve_composite_window(req)
    assert resp.status == "error"
    assert "Missing required Sentinel-1 SAR bands" in resp.error
    assert "VH" in resp.error


def test_17_stac_composite_sar_budget_exhaustion():
    """
    Tests that cumulative transfer budget exhaustion before reading VH returns error.
    """
    service = STACAcquisitionService()
    service._scene_cache["S1A_RTC_BUDGET"] = {
        "raw_assets": {
            "vv": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/s1/vv.tif"},
            "vh": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/s1/vh.tif"}
        },
        "collection": "sentinel-1-rtc"
    }

    dummy_meta = {
        "crs": "EPSG:4326",
        "transform": [0.0001, 0, 77.5, 0, -0.0001, 13.0],
        "nodata": -32768.0,
        "transfer_telemetry": {"cumulative_bytes": 5000}
    }

    req = STACCompositeRetrieveRequest(
        scene_id="S1A_RTC_BUDGET",
        aoi_bbox=[77.50, 12.95, 77.55, 13.00],
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        max_bytes=4000  # Budget is 4000 bytes, but VV alone transfers 5000 bytes!
    )

    with patch("app.acquisition.windowed_reader.SafeWindowedCOGReader.read_cog_window") as mock_read:
        mock_read.return_value = (np.full((16, 16), 0.20, dtype=np.float32), dummy_meta)
        resp = service.retrieve_composite_window(req)

    assert resp.status == "error"
    assert "Cumulative transfer budget" in resp.error


# ---------------------------------------------------------------------------
# 5. Evidence Package & Scientific Reporting Integration
# ---------------------------------------------------------------------------

def test_18_evidence_package_builder_sar_dprvi():
    """
    Verifies that EvidencePackageBuilder extracts calibrated DpRVI measurements,
    heuristics, and evidence artifacts from SAR responses.
    """
    dprvi_data = {
        "status": "success",
        "dprvi_mean": 0.5200,
        "dprvi_p90": 0.5800,
        "cross_pol_ratio_linear": 0.2500,
        "cross_pol_ratio_db": -6.02,
        "vv_mean_linear": 0.2000,
        "vh_mean_linear": 0.0500,
        "vv_mean_db": -6.99,
        "vh_mean_db": -13.01,
        "vegetation_proxy_coverage_pct": 65.5,
        "veg_pixel_count": 655,
        "vegetation_threshold": 0.45,
        "preview_base64": "data:image/png;base64,dummy_sar"
    }

    analysis_resp = {
        "id": "analysis_s1_rtc",
        "query": "Assess radar crop canopy",
        "task": "optical_sar",
        "metadata": {
            "dprvi_result": dprvi_data
        }
    }

    pkg = EvidencePackageBuilder.from_analysis_response(analysis_resp)

    # Check physical measurements
    pm_dict = {pm.name: pm for pm in pkg.physical_measurements}
    assert "sar_mean_dprvi" in pm_dict
    assert pm_dict["sar_mean_dprvi"].value == 0.5200
    assert pm_dict["sar_mean_dprvi"].unit == "unitless ratio [0, 1]"
    assert "sar_cross_pol_ratio_db" in pm_dict
    assert pm_dict["sar_cross_pol_ratio_db"].value == -6.02

    # Check heuristic proxy
    heur_dict = {h.name: h for h in pkg.heuristic_indicators}
    assert "radar_vegetation_volume_coverage" in heur_dict
    assert heur_dict["radar_vegetation_volume_coverage"].percentage == 65.5

    # Check evidence artifacts
    art_ids = {a.artifact_id for a in pkg.evidence_artifacts}
    assert "ev_sar_dprvi_map" in art_ids


def test_19_scientific_report_markdown_dprvi_rendering():
    """
    Verifies that ScientificReportGenerator and generate_scientific_markdown
    produce formal provenance sections for Sentinel-1 RTC DpRVI.
    """
    dprvi_data = {
        "status": "success",
        "dprvi_mean": 0.5200,
        "dprvi_p90": 0.5800,
        "cross_pol_ratio_linear": 0.2500,
        "cross_pol_ratio_db": -6.02,
        "vv_mean_db": -6.99,
        "vh_mean_db": -13.01,
        "vegetation_proxy_coverage_pct": 65.5,
        "vegetation_threshold": 0.45,
        "provenance": {"vv_asset_id": "s1_vv_cog", "vh_asset_id": "s1_vh_cog"},
        "validity_policy": {"sar_terrain_layover_shadow_mask": "unavailable_from_stac_provider"}
    }

    analysis_resp = {
        "id": "analysis_report_s1",
        "query": "Evaluate crop canopy with Sentinel-1 RTC",
        "task": "optical_sar",
        "metadata": {
            "dprvi_result": dprvi_data
        }
    }

    report = ScientificReportBuilder.build_report(analysis_resp)

    assert report.provenance.dprvi_mean == 0.5200
    assert report.provenance.cross_pol_ratio_db == -6.02
    assert "SENTINEL1_RTC_DUAL_POL_COMPUTED" in report.reproducibility.pipeline_steps

    md = generate_scientific_markdown(report)

    assert "### Sentinel-1 RTC Dual-Polarization (VV + VH) & DpRVI Provenance" in md
    assert "| **Mean DpRVI** | `0.52` |" in md
    assert "| **Cross-Pol Ratio (dB)** | `-6.02 dB` |" in md
    assert "Mandal et al., 2020" in md
    assert "unavailable_from_stac_provider" in md


def test_20_local_sample_sar_windowed_retrieval():
    """
    Verifies local offline sample SAR retrieval with dual-pol and DpRVI calculation.
    """
    service = STACAcquisitionService()

    req = STACCompositeRetrieveRequest(
        scene_id="local_sample_risat_sar_mumbai",
        aoi_bbox=[72.915, 18.983, 72.922, 18.989],
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        compute_dprvi=True
    )

    resp = service.retrieve_composite_window(req)

    assert resp.status == "success"
    assert resp.category == "scientific_sar"
    assert resp.dprvi_result is not None
    assert resp.dprvi_result["dprvi_mean"] is not None
    assert resp.sar_quality_screening is not None
