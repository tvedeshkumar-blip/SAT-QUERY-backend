import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from app.models.optical_sar.optical_sar_fusion import (
    OpticalSARJointAnalysisProvider,
    OpticalSARVisualizationBaseline,
    OpticalSARProvider
)
from app.remote_sensing.registration import align_image_pair, compute_source_valid_mask


def test_optical_sar_joint_analysis_artifacts():
    analyzer = OpticalSARJointAnalysisProvider()
    opt = np.random.randint(40, 220, (100, 100, 3), dtype=np.uint8)
    sar = np.random.randint(10, 250, (100, 100), dtype=np.uint8)

    bounds = [77.5, 12.9, 77.6, 13.0]
    t = from_bounds(*bounds, 100, 100)
    meta = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "width": 100,
        "height": 100
    }
    res = analyzer.predict([opt, sar], query="Fuse optical and SAR imagery.", metadata={"optical": meta, "sar": meta})
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


# ============================================================================
# Stage 6 Scientific-Integrity Regression Tests (GAP-1, GAP-2, GAP-3)
# ============================================================================

def test_zero_padded_margins_do_not_inflate_water_mask():
    """
    1. Zero-padded, non-overlapping margins do not increase the water mask or any class percentage.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    # 100x100 optical scene covering [10.0, 10.0, 10.1, 10.1]
    bounds_opt = [10.0, 10.0, 10.1, 10.1]
    t_opt = from_bounds(*bounds_opt, 100, 100)
    meta_opt = {
        "crs": "EPSG:4326",
        "bounds": bounds_opt,
        "transform": list(t_opt)[:6],
        "width": 100,
        "height": 100
    }
    opt = np.full((100, 100, 3), 120, dtype=np.uint8)

    # 50x50 SAR scene covering only southern half [10.0, 10.0, 10.1, 10.05]
    bounds_sar = [10.0, 10.0, 10.1, 10.05]
    t_sar = from_bounds(*bounds_sar, 50, 50)
    meta_sar = {
        "crs": "EPSG:4326",
        "bounds": bounds_sar,
        "transform": list(t_sar)[:6],
        "width": 50,
        "height": 50
    }
    sar = np.full((50, 50), 100, dtype=np.uint8)

    res = analyzer.predict([opt, sar], metadata={"optical": meta_opt, "sar": meta_sar})

    # Reprojection leaves the northern half of the aligned SAR raster as 0-padded.
    # The zero-padded northern margin must NOT be counted as specular water!
    assert res.get("status") != "unsupported"
    assert res["valid_pixel_count"] < 10000
    # Inside valid footprint, SAR is 100 (> 45), so water pixel count must be 0
    assert res["water_pct"] == 0.0


def test_genuine_zero_valued_pixels_retained():
    """
    2. Genuine zero-valued pixels are retained when the source validity mask marks them as valid.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    bounds = [10.0, 10.0, 10.1, 10.1]
    t = from_bounds(*bounds, 50, 50)
    meta = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "nodata": None  # Zero is a genuine measurement, not nodata
    }

    opt = np.full((50, 50, 3), 20, dtype=np.uint8)  # Low optical albedo (< 60)
    sar = np.zeros((50, 50), dtype=np.uint8)        # Genuine zero radar backscatter (< 45)

    res = analyzer.predict([opt, sar], metadata={"optical": meta, "sar": meta})
    assert res["valid_pixel_count"] == 2500
    assert res["water_pct"] == 100.0


def test_source_nodata_and_invalids_excluded_from_shared_mask():
    """
    3. Source nodata and invalid pixels from either modality are excluded from shared valid mask.
    """
    bounds = [10.0, 10.0, 10.1, 10.1]
    t = from_bounds(*bounds, 50, 50)

    opt = np.full((50, 50, 3), 100, dtype=np.float32)
    opt[:10, :10] = np.nan  # Invalids in top-left 10x10
    meta_opt = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "nodata": None
    }

    sar = np.full((50, 50), 100, dtype=np.float32)
    sar[40:, 40:] = -9999.0  # Nodata in bottom-right 10x10
    meta_sar = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "nodata": -9999.0
    }

    _, _, reg_info = align_image_pair(opt, sar, meta_opt, meta_sar)
    mask = reg_info["valid_mask"]

    # 2500 total - 100 (NaN in opt) - 100 (nodata in sar) = 2300 valid
    assert reg_info["valid_pixel_count"] == 2300
    assert not mask[:10, :10].any()  # Optical NaNs excluded
    assert not mask[40:, 40:].any()  # SAR nodata excluded


def test_missing_georeferencing_cannot_claim_geographic_alignment():
    """
    4. Missing georeferencing cannot silently fall back to resize-based geographic alignment.
    """
    img_a = np.zeros((60, 60, 3), dtype=np.uint8)
    img_b = np.zeros((100, 80), dtype=np.uint8)

    _, _, reg_info = align_image_pair(img_a, img_b)

    assert reg_info["georeferenced"] is False
    assert reg_info["co_registered"] is False
    assert reg_info["alignment_valid"] is False
    assert reg_info["is_display_only"] is True
    assert reg_info["registration_method"] == "unreferenced_visual_resize_only"


def test_display_only_resizing_cannot_feed_geospatial_classification():
    """
    5. Display-only resizing cannot feed geospatial classification or area calculations.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    opt = np.random.randint(40, 200, (60, 60, 3), dtype=np.uint8)
    sar = np.random.randint(10, 200, (100, 80), dtype=np.uint8)

    res = analyzer.predict([opt, sar])

    assert res["status"] == "unsupported"
    assert res["analysis_status"] == "unsupported_alignment"
    assert res["water_pct"] is None
    assert res["urban_pct"] is None
    assert res["veg_pct"] is None
    # No classification masks emitted
    mask_evidences = [e for e in res["evidence"] if e.get("type") == "mask"]
    assert len(mask_evidences) == 0


def test_existing_correctly_georeferenced_alignment_continues_working():
    """
    6. Existing correctly georeferenced alignment continues to work.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    bounds = [77.5, 12.9, 77.6, 13.0]
    t = from_bounds(*bounds, 64, 64)
    meta = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "width": 64,
        "height": 64
    }

    opt = np.random.randint(40, 200, (64, 64, 3), dtype=np.uint8)
    sar = np.random.randint(10, 200, (64, 64), dtype=np.uint8)

    res = analyzer.predict([opt, sar], metadata={"optical": meta, "sar": meta})

    assert res["actual_model_used"] == "OpticalSARJointAnalysisProvider"
    assert res["fallback_used"] is False
    assert res["registration_info"]["co_registered"] is True
    assert res["valid_pixel_count"] == 64 * 64
    assert res["water_pct"] is not None


def test_empty_valid_data_intersection_produces_explicit_unsupported():
    """
    7. Empty valid-data intersections produce an explicit unsupported result rather than misleading percentages.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    bounds = [10.0, 10.0, 10.1, 10.1]
    t = from_bounds(*bounds, 50, 50)

    # Optical scene where all pixels match nodata = 0
    opt = np.zeros((50, 50, 3), dtype=np.uint8)
    meta_opt = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "nodata": 0
    }

    sar = np.full((50, 50), 100, dtype=np.uint8)
    meta_sar = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6],
        "nodata": None
    }

    res = analyzer.predict([opt, sar], metadata={"optical": meta_opt, "sar": meta_sar})

    assert res["status"] == "unsupported"
    assert res["analysis_status"] == "unsupported_empty_intersection"
    assert res["valid_pixel_count"] == 0
    assert res["water_pct"] is None
    assert res["urban_pct"] is None
    assert res["veg_pct"] is None
    assert "empty" in res["answer"].lower()


def test_heuristic_outputs_retain_scientific_disclaimers():
    """
    8. Heuristic outputs retain their scientific disclaimers and accurate provider/fallback metadata.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    bounds = [10.0, 10.0, 10.1, 10.1]
    t = from_bounds(*bounds, 50, 50)
    meta = {
        "crs": "EPSG:4326",
        "bounds": bounds,
        "transform": list(t)[:6]
    }

    opt = np.random.randint(40, 200, (50, 50, 3), dtype=np.uint8)
    sar = np.random.randint(10, 200, (50, 50), dtype=np.uint8)

    res = analyzer.predict([opt, sar], metadata={"optical": meta, "sar": meta})

    assert res["actual_model_used"] == "OpticalSARJointAnalysisProvider"
    assert res["fallback_used"] is False
    assert "Preliminary Scene-Normalized Heuristic Indicators" in res["answer"]

    lims = " ".join(res["limitations"]).lower()
    assert "preliminary" in lims
    assert "no trained machine learning model" in lims
    assert "non-interchangeable" in lims


# ============================================================================
# Stage 4A P0-2 & P0-3 Regression Tests: Cross-CRS Alignment & Shape Inference
# ============================================================================

def test_p0_2_cross_crs_reprojection_common_grid():
    """
    P0-2 Test 1: Valid optical raster in EPSG:32643 and georeferenced SAR raster in EPSG:4326
    covering the same ground area successfully reprojects onto reference grid.
    Asserts CRS, transform, dimensions, overlap (>90%), alignment_valid, and co_registered.
    """
    # Optical in UTM zone 43N (Bangalore area, approx 13°N)
    # 1000m x 1000m scene with 10m pixels -> 100x100
    opt_bounds = [770000.0, 1430000.0, 771000.0, 1431000.0]
    t_opt = from_bounds(*opt_bounds, 100, 100)
    meta_opt = {
        "crs": "EPSG:32643",
        "bounds": opt_bounds,
        "transform": list(t_opt)[:6],
        "width": 100,
        "height": 100
    }
    opt_img = np.full((100, 100, 3), 120, dtype=np.uint8)

    # SAR in WGS84 geographic (EPSG:4326) covering the same ground area
    sar_bounds_4326 = list(transform_bounds("EPSG:32643", "EPSG:4326", *opt_bounds))
    t_sar = from_bounds(*sar_bounds_4326, 100, 100)
    meta_sar = {
        "crs": "EPSG:4326",
        "bounds": sar_bounds_4326,
        "transform": list(t_sar)[:6],
        "width": 100,
        "height": 100
    }
    sar_img = np.full((100, 100), 90, dtype=np.uint8)

    aligned_opt, aligned_sar, reg_info = align_image_pair(
        opt_img, sar_img,
        meta_a=meta_opt, meta_b=meta_sar
    )

    # Assertions on common grid geometry
    assert reg_info["alignment_valid"] is True
    assert reg_info["co_registered"] is True
    assert reg_info["georeferenced"] is True
    assert reg_info["is_display_only"] is False
    assert reg_info["target_crs"] == "EPSG:32643"
    assert reg_info["target_shape"] == [100, 100]
    assert np.allclose(reg_info["target_transform"], list(t_opt)[:6])
    assert aligned_opt.shape == (100, 100, 3)
    assert aligned_sar.shape == (100, 100)
    assert reg_info["spatial_overlap_percent"] > 90.0
    assert reg_info["valid_pixel_count"] > 0


def test_p0_2_different_crs_no_overlap_rejected():
    """
    P0-2 Test 2: Different-CRS rasters with zero spatial overlap are explicitly rejected.
    """
    # Optical in UTM zone 43N (Bangalore area)
    opt_bounds = [770000.0, 1430000.0, 771000.0, 1431000.0]
    t_opt = from_bounds(*opt_bounds, 100, 100)
    meta_opt = {
        "crs": "EPSG:32643",
        "bounds": opt_bounds,
        "transform": list(t_opt)[:6],
        "width": 100,
        "height": 100
    }
    opt_img = np.full((100, 100, 3), 120, dtype=np.uint8)

    # SAR in WGS84 geographic (Paris area: lon 2.3, lat 48.8)
    sar_bounds_paris = [2.30, 48.80, 2.35, 48.85]
    t_sar = from_bounds(*sar_bounds_paris, 100, 100)
    meta_sar = {
        "crs": "EPSG:4326",
        "bounds": sar_bounds_paris,
        "transform": list(t_sar)[:6],
        "width": 100,
        "height": 100
    }
    sar_img = np.full((100, 100), 90, dtype=np.uint8)

    aligned_opt, aligned_sar, reg_info = align_image_pair(
        opt_img, sar_img,
        meta_a=meta_opt, meta_b=meta_sar
    )

    assert reg_info["alignment_valid"] is False
    assert reg_info["is_display_only"] is True
    assert reg_info["registration_method"] == "no_spatial_overlap"
    assert reg_info["spatial_overlap_percent"] == 0.0


def test_p0_2_missing_crs_or_invalid_transform_fails_safely():
    """
    P0-2 Test 3: Missing CRS or invalid transform fails safely, remaining display-only.
    """
    opt_img = np.full((64, 64, 3), 120, dtype=np.uint8)
    sar_img = np.full((64, 64), 90, dtype=np.uint8)

    # Missing CRS
    meta_missing_crs = {"crs": None, "transform": [10.0, 0, 0, 0, -10.0, 0]}
    meta_valid = {"crs": "EPSG:4326", "transform": [0.0001, 0, 77.0, 0, -0.0001, 13.0]}
    _, _, reg_info = align_image_pair(opt_img, sar_img, meta_missing_crs, meta_valid)
    assert reg_info["alignment_valid"] is False
    assert reg_info["is_display_only"] is True
    assert reg_info["georeferenced"] is False

    # Invalid transform (degenerate matrix with zero pixel resolution)
    meta_invalid_transform = {"crs": "EPSG:4326", "transform": [0, 0, 0, 0, 0, 0]}
    _, _, reg_info2 = align_image_pair(opt_img, sar_img, meta_invalid_transform, meta_valid)
    assert reg_info2["alignment_valid"] is False
    assert reg_info2["is_display_only"] is True
    assert reg_info2["georeferenced"] is False


def test_p0_3_unreferenced_equal_sized_arrays_never_infer_alignment():
    """
    P0-3 Test 4: Two unreferenced 512x512 arrays must NEVER establish geospatial alignment
    solely from having identical dimensions. Alignment must remain invalid, display-only,
    and downstream joint analysis must reject them with status='unsupported'.
    """
    opt_512 = np.random.randint(40, 200, (512, 512, 3), dtype=np.uint8)
    sar_512 = np.random.randint(10, 200, (512, 512), dtype=np.uint8)

    # Direct alignment call with no metadata
    _, _, reg_info = align_image_pair(opt_512, sar_512, meta_a=None, meta_b=None)
    assert reg_info["alignment_valid"] is False
    assert reg_info["co_registered"] is False
    assert reg_info["georeferenced"] is False
    assert reg_info["is_display_only"] is True
    assert reg_info["registration_method"] == "unreferenced_same_dimensions_display_only"

    # Downstream joint analysis provider must treat as unsupported
    analyzer = OpticalSARJointAnalysisProvider()
    res = analyzer.predict([opt_512, sar_512], metadata=None)
    assert res["status"] == "unsupported"
    assert res["analysis_status"] == "unsupported_alignment"
    assert "safe geospatial alignment could not be established" in res["answer"].lower()
    assert "display-only resizing cannot feed geospatial classification" in res["answer"].lower()
    assert res["water_pct"] is None
    assert res["urban_pct"] is None
    assert res["veg_pct"] is None


# ============================================================================
# Stage 4B P1-3 Regression Tests: Multi-band NoData Mask Precision
# ============================================================================

def test_p1_3_compute_source_valid_mask_all_channels_valid():
    """All channels valid produces an all-True 2D boolean mask."""
    arr = np.full((10, 10, 3), 100, dtype=np.uint8)
    mask = compute_source_valid_mask(arr, nodata=0)
    assert mask.shape == (10, 10)
    assert mask.all()


def test_p1_3_compute_source_valid_mask_all_channels_nodata():
    """All channels equal to nodata produces an all-False mask."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = compute_source_valid_mask(arr, nodata=0)
    assert mask.shape == (10, 10)
    assert not mask.any()


def test_p1_3_compute_source_valid_mask_one_channel_nodata_is_invalid():
    """
    CRITICAL P1-3: A pixel must be INVALID if ANY required source channel is NoData.
    Testing pixel (0, 0) having channel 1 as NoData (0), channels 0 and 2 valid (100).
    """
    arr = np.full((10, 10, 3), 100, dtype=np.uint8)
    arr[0, 0, 1] = 0  # One channel is nodata
    mask = compute_source_valid_mask(arr, nodata=0)
    assert mask.shape == (10, 10)
    assert mask[0, 0] == False  # Strictly False: one channel NoData makes pixel invalid
    assert mask[0, 1] == True   # Other pixels remain valid


def test_p1_3_compute_source_valid_mask_nan_and_inf_handling():
    """Floating-point NaN and Inf in any channel invalidate the pixel."""
    arr = np.full((10, 10, 3), 1.0, dtype=np.float32)
    arr[2, 3, 0] = np.nan
    arr[4, 5, 2] = np.inf
    arr[6, 7, 1] = -np.inf
    mask = compute_source_valid_mask(arr, nodata=-9999.0)
    assert mask[2, 3] == False
    assert mask[4, 5] == False
    assert mask[6, 7] == False
    assert mask[0, 0] == True


def test_p1_3_compute_source_valid_mask_single_band():
    """Single-band 2D array (H, W) handles nodata and valid pixels correctly."""
    arr = np.array([
        [10.0, -9999.0, 20.0],
        [np.nan, 30.0, np.inf]
    ], dtype=np.float32)
    mask = compute_source_valid_mask(arr, nodata=-9999.0)
    assert mask.shape == (2, 3)
    assert mask[0, 0] == True
    assert mask[0, 1] == False
    assert mask[0, 2] == True
    assert mask[1, 0] == False
    assert mask[1, 1] == True
    assert mask[1, 2] == False


def test_p1_3_compute_source_valid_mask_channel_first_layout():
    """(C, H, W) layout correctly identifies channel axis 0 and produces (H, W) mask."""
    arr = np.full((3, 20, 30), 100, dtype=np.uint8)
    arr[1, 5, 5] = 0  # Channel 1 at (y=5, x=5) is nodata
    mask = compute_source_valid_mask(arr, nodata=0)
    assert mask.shape == (20, 30)
    assert mask[5, 5] == False
    assert mask[0, 0] == True


# ============================================================================
# Step 4B.2 Regression Tests: Array-Layout Safety, Hints, and Ambiguity Protection
# ============================================================================

def test_step4b2_explicit_layout_hwc_and_chw():
    """Explicit layout parameter accurately directs channel collapse for both layouts."""
    # HWC layout
    arr_hwc = np.full((25, 35, 4), 100.0, dtype=np.float32)
    arr_hwc[10, 15, 2] = -9999.0
    mask_hwc = compute_source_valid_mask(arr_hwc, nodata=-9999.0, layout="HWC")
    assert mask_hwc.shape == (25, 35)
    assert mask_hwc[10, 15] == False
    assert mask_hwc[0, 0] == True

    # CHW layout
    arr_chw = np.full((4, 25, 35), 100.0, dtype=np.float32)
    arr_chw[2, 10, 15] = -9999.0
    mask_chw = compute_source_valid_mask(arr_chw, nodata=-9999.0, layout="CHW")
    assert mask_chw.shape == (25, 35)
    assert mask_chw[10, 15] == False
    assert mask_chw[0, 0] == True


def test_step4b2_channel_first_with_small_spatial_dimension():
    """
    Channel-first array where channel dimension exceeds a spatial dimension (e.g. C=12, H=8, W=256).
    Must not be misclassified as channel-last (H=12, W=8, C=256).
    """
    arr = np.full((12, 8, 256), 50.0, dtype=np.float32)
    # Channel 5 at (y=3, x=10) has nodata
    arr[5, 3, 10] = -9999.0

    # 1. With explicit layout parameter
    mask_layout = compute_source_valid_mask(arr, nodata=-9999.0, layout="CHW")
    assert mask_layout.shape == (8, 256)
    assert mask_layout[3, 10] == False
    assert mask_layout[0, 0] == True

    # 2. With metadata hint
    meta = {"bands": 12, "height": 8, "width": 256}
    mask_meta = compute_source_valid_mask(arr, nodata=-9999.0, metadata=meta)
    assert mask_meta.shape == (8, 256)
    assert mask_meta[3, 10] == False

    # 3. Shape deduction without hint (d0=12 <= 16, d2=256 > 16)
    mask_auto = compute_source_valid_mask(arr, nodata=-9999.0)
    assert mask_auto.shape == (8, 256)
    assert mask_auto[3, 10] == False


def test_step4b2_ambiguous_shape_3_3_3_raises_value_error():
    """
    Genuinely ambiguous 3D array (3, 3, 3) without a layout hint MUST fail explicitly
    rather than guessing and potentially producing wrong masks.
    """
    arr = np.full((3, 3, 3), 10.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Ambiguous 3D array shape"):
        compute_source_valid_mask(arr, nodata=-9999.0)


def test_step4b2_ambiguous_shape_3_3_3_resolves_with_layout_hint():
    """
    Ambiguous (3, 3, 3) resolves safely when explicit layout is specified,
    collapsing the proper channel axis.
    """
    arr = np.full((3, 3, 3), 10.0, dtype=np.float32)
    # Put nodata at coordinate (0, 1, 2)
    arr[0, 1, 2] = -9999.0

    # Under HWC: (H=3, W=3, C=3) -> nodata is at (y=0, x=1, c=2), so pixel (0, 1) is invalid
    mask_hwc = compute_source_valid_mask(arr, nodata=-9999.0, layout="HWC")
    assert mask_hwc.shape == (3, 3)
    assert mask_hwc[0, 1] == False
    assert mask_hwc[1, 2] == True

    # Under CHW: (C=3, H=3, W=3) -> nodata is at (c=0, y=1, x=2), so pixel (1, 2) is invalid
    mask_chw = compute_source_valid_mask(arr, nodata=-9999.0, layout="CHW")
    assert mask_chw.shape == (3, 3)
    assert mask_chw[1, 2] == False
    assert mask_chw[0, 1] == True


def test_step4b2_ambiguous_shape_resolves_with_metadata():
    """Ambiguous shapes resolve accurately using metadata layout hints or channel axis."""
    arr = np.full((3, 3, 3), 20.0, dtype=np.float32)
    arr[0, 1, 2] = -9999.0

    # Metadata layout hint
    mask_hwc = compute_source_valid_mask(arr, nodata=-9999.0, metadata={"layout": "HWC"})
    assert mask_hwc.shape == (3, 3)
    assert mask_hwc[0, 1] == False

    # Metadata channel_axis hint
    mask_chw = compute_source_valid_mask(arr, nodata=-9999.0, metadata={"channel_axis": 0})
    assert mask_chw.shape == (3, 3)
    assert mask_chw[1, 2] == False


def test_step4b2_ambiguous_strip_chip_fails_without_hint_and_succeeds_with_hint():
    """
    Shape (8, 256, 12) has both boundary dimensions <= 16.
    Fails explicitly without a hint, succeeds when layout is provided.
    """
    arr = np.full((8, 256, 12), 10.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Ambiguous 3D array shape"):
        compute_source_valid_mask(arr, nodata=-9999.0)

    # Resolves with layout='HWC' (H=8, W=256, C=12)
    mask_hwc = compute_source_valid_mask(arr, nodata=-9999.0, layout="HWC")
    assert mask_hwc.shape == (8, 256)

    # Resolves with layout='CHW' (C=8, H=256, W=12)
    mask_chw = compute_source_valid_mask(arr, nodata=-9999.0, layout="CHW")
    assert mask_chw.shape == (256, 12)


def test_step4b2_single_band_and_2d_arrays():
    """Handles 2D (H, W), 3D single-band HWC (H, W, 1), and 3D single-band CHW (1, H, W)."""
    # 2D array
    arr_2d = np.array([[1.0, -9999.0], [np.nan, 2.0]], dtype=np.float32)
    mask_2d = compute_source_valid_mask(arr_2d, nodata=-9999.0)
    assert mask_2d.shape == (2, 2)
    assert mask_2d[0, 0] == True
    assert mask_2d[0, 1] == False
    assert mask_2d[1, 0] == False
    assert mask_2d[1, 1] == True

    # 3D single-band (H, W, 1)
    arr_3d_hwc = np.array([[[1.0], [-9999.0]], [[np.nan], [2.0]]], dtype=np.float32)
    mask_3d_hwc = compute_source_valid_mask(arr_3d_hwc, nodata=-9999.0, layout="HWC")
    assert mask_3d_hwc.shape == (2, 2)
    assert mask_3d_hwc[0, 0] == True
    assert mask_3d_hwc[0, 1] == False
    assert mask_3d_hwc[1, 0] == False
    assert mask_3d_hwc[1, 1] == True

    # 3D single-band (1, H, W)
    arr_3d_chw = np.array([[[1.0, -9999.0], [np.nan, 2.0]]], dtype=np.float32)
    mask_3d_chw = compute_source_valid_mask(arr_3d_chw, nodata=-9999.0, layout="CHW")
    assert mask_3d_chw.shape == (2, 2)
    assert mask_3d_chw[0, 0] == True
    assert mask_3d_chw[0, 1] == False
    assert mask_3d_chw[1, 0] == False
    assert mask_3d_chw[1, 1] == True


def test_step4b2_partial_band_nodata_strictly_invalidates():
    """Pixel is invalid if even a single channel out of many contains NoData."""
    arr = np.full((10, 10, 4), 100.0, dtype=np.float32)
    # At pixel (4, 4), channel 3 has nodata, but channels 0, 1, 2 are valid
    arr[4, 4, 3] = -9999.0
    mask = compute_source_valid_mask(arr, nodata=-9999.0, layout="HWC")
    assert mask.shape == (10, 10)
    assert mask[4, 4] == False
    assert mask[0, 0] == True


def test_step4b2_nan_inf_strictness():
    """NaN, +Inf, -Inf in any channel strictly invalidate the pixel."""
    arr = np.full((10, 10, 3), 1.0, dtype=np.float32)
    arr[2, 3, 0] = np.nan
    arr[4, 5, 1] = np.inf
    arr[6, 7, 2] = -np.inf
    mask = compute_source_valid_mask(arr, layout="HWC")
    assert mask[2, 3] == False
    assert mask[4, 5] == False
    assert mask[6, 7] == False
    assert mask[0, 0] == True


def test_step4b2_zero_value_preservation():
    """Legitimate zero values are preserved unless nodata is explicitly 0."""
    arr = np.zeros((10, 10, 3), dtype=np.float32)

    # 1. nodata = -9999.0: zeros are valid observations
    mask_neg = compute_source_valid_mask(arr, nodata=-9999.0, layout="HWC")
    assert mask_neg.all()

    # 2. nodata is None: zeros are valid observations
    mask_none = compute_source_valid_mask(arr, nodata=None, layout="HWC")
    assert mask_none.all()

    # 3. nodata = 0: zeros are declared NoData
    mask_zero = compute_source_valid_mask(arr, nodata=0, layout="HWC")
    assert not mask_zero.any()


def test_step4b2_invalid_array_dimensions_raise_value_error():
    """1D or 4D arrays raise ValueError."""
    arr_1d = np.ones((50,))
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        compute_source_valid_mask(arr_1d)

    arr_4d = np.ones((1, 3, 50, 50))
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        compute_source_valid_mask(arr_4d)


# ============================================================================
# Stage 4B P1-1 Regression Tests: CrossSensorValidator Pre-Fusion Integration
# ============================================================================

def test_p1_1_optical_sar_provider_rejects_incompatible_pair():
    """
    P1-1 Test: OpticalSARJointAnalysisProvider rejects pair rejected by CrossSensorValidator.
    Must return status='unsupported', analysis_status='unsupported_validation',
    water_pct=None, and retain the explicit rejection reason.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    # 10m Optical vs 250m SAR (> 10x resolution disparity rejected by CrossSensorValidator)
    opt = np.random.randint(40, 200, (64, 64, 3), dtype=np.uint8)
    sar = np.random.randint(10, 200, (4, 4), dtype=np.uint8)

    meta_opt = {
        "crs": "EPSG:32643",
        "bounds": [770000.0, 1427000.0, 771000.0, 1428000.0],
        "width": 64,
        "height": 64,
        "modality": "OPTICAL",
        "resolution": [10.0, 10.0]
    }
    meta_sar = {
        "crs": "EPSG:32643",
        "bounds": [770000.0, 1427000.0, 771000.0, 1428000.0],
        "width": 4,
        "height": 4,
        "modality": "SAR",
        "resolution": [250.0, 250.0],
        "tags": {"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    }

    res = analyzer.predict([opt, sar], metadata={"optical": meta_opt, "sar": meta_sar})

    assert res["status"] == "unsupported"
    assert res["analysis_status"] == "unsupported_validation"
    assert "Excessive resolution disparity" in res["answer"]
    assert res["water_pct"] is None
    assert res["urban_pct"] is None
    assert res["veg_pct"] is None
    assert res["cross_sensor_validation"]["is_compatible"] is False


def test_p1_1_temporal_advisory_warning_does_not_hard_reject():
    """
    P1-1 Test: Temporal separation advisory warning (> 60 days) does NOT become a hard rejection.
    Pair proceeds to fusion, with the warning recorded in limitations and warnings.
    """
    analyzer = OpticalSARJointAnalysisProvider()

    opt = np.random.randint(40, 200, (64, 64, 3), dtype=np.uint8)
    sar = np.random.randint(10, 200, (64, 64), dtype=np.uint8)

    meta_opt = {
        "crs": "EPSG:32643",
        "bounds": [770000.0, 1427000.0, 771000.0, 1428000.0],
        "width": 64,
        "height": 64,
        "modality": "OPTICAL",
        "datetime": "2023-01-01T10:00:00Z"
    }
    meta_sar = {
        "crs": "EPSG:32643",
        "bounds": [770000.0, 1427000.0, 771000.0, 1428000.0],
        "width": 64,
        "height": 64,
        "modality": "SAR",
        "datetime": "2023-04-15T10:00:00Z",  # ~104 days apart (> 60 days warning)
        "tags": {"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    }

    res = analyzer.predict([opt, sar], metadata={"optical": meta_opt, "sar": meta_sar})

    # Must NOT be rejected
    assert res["status"] in ("success", None) or res.get("analysis_status") not in ("unsupported_validation",)
    assert res["water_pct"] is not None
    # Temporal advisory warning is preserved in limitations
    assert any("Temporal separation advisory" in lim for lim in res["limitations"])
