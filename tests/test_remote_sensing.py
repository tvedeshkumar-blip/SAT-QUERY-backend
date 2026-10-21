import io
import base64
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from app.remote_sensing.geotiff import parse_geotiff_or_image
from app.remote_sensing.modality import detect_image_modality, detect_modality_structured
from app.remote_sensing.preprocessing import normalize_to_uint8
from app.remote_sensing.registration import align_image_pair, compute_bounds_overlap

def test_parse_standard_png():
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    arr, meta = parse_geotiff_or_image(b64, "scene.png")
    assert arr.shape == (128, 128, 3)
    assert meta["is_geotiff"] is False
    assert meta["crs"] is None
    assert meta["crs_display"] == "CRS unavailable"
    assert meta["width"] == 128
    assert meta["height"] == 128
    assert meta["bands"] == 3

def test_parse_geotiff_with_tags():
    # Valid CRS tag present
    img = Image.new("RGB", (64, 64), color=(30, 90, 150))
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (10.0, 10.0, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, 432100.0, 1421000.0, 0.0)
    ifd[34737] = b"WGS 84 / UTM zone 44N|EPSG:32644|"
    buf = io.BytesIO()
    img.save(buf, format="TIFF", tiffinfo=ifd)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    arr, meta = parse_geotiff_or_image(b64, "cartosat_scene.tif")
    assert meta["is_geotiff"] is True
    assert meta["crs"] is not None
    assert "32644" in meta["crs"] or "UTM" in meta["crs"]
    assert meta["bounds"] is not None
    assert meta["bounds"][0] == 432100.0
    assert meta["bounds"][2] == 432100.0 + (64 * 10.0)

def test_parse_geotiff_missing_crs():
    # GeoTIFF tags present but CRS tag missing -> must be None, NOT EPSG:32644
    img = Image.new("RGB", (64, 64), color=(30, 90, 150))
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (10.0, 10.0, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, 432100.0, 1421000.0, 0.0)
    buf = io.BytesIO()
    img.save(buf, format="TIFF", tiffinfo=ifd)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    arr, meta = parse_geotiff_or_image(b64, "cartosat_scene_nocrs.tif")
    assert meta["is_geotiff"] is True
    assert meta["crs"] is None
    assert meta["crs_display"] == "CRS unavailable"

def test_modality_detection_hints():
    # Optical RGB
    opt_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mod_opt = detect_image_modality(opt_arr, {"filename": "optical.png"}, hint="optical")
    assert mod_opt == "OPTICAL"

    # SAR single-channel / hint
    sar_arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    mod_sar = detect_image_modality(sar_arr, {"filename": "risat_sar.tif"}, hint="sar")
    assert mod_sar == "SAR"

    # Structured provenance
    struct_res = detect_modality_structured(sar_arr, {"filename": "s1_sar.tif"}, hint="sar")
    assert struct_res["source"] in ("user", "filename", "heuristic")
    assert struct_res["confidence"] is None

def test_percentile_normalization():
    u16_arr = (np.random.rand(64, 64) * 4096).astype(np.uint16)
    u8_arr = normalize_to_uint8(u16_arr)
    assert u8_arr.dtype == np.uint8
    assert np.min(u8_arr) >= 0
    assert np.max(u8_arr) <= 255

def test_image_pair_alignment_unreferenced():
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((150, 120, 3), dtype=np.uint8)
    aligned_a, aligned_b, info = align_image_pair(img_a, img_b)
    assert aligned_a.shape == (100, 100, 3)
    assert aligned_b.shape == (100, 100, 3)
    # Unreferenced images must NOT claim true geodetic co-registration
    assert info["co_registered"] is False
    assert info["georeferenced"] is False

def test_spatial_overlap_calculation():
    # Overlapping boxes
    box_a = [0.0, 0.0, 100.0, 100.0]
    box_b = [50.0, 50.0, 150.0, 150.0]
    overlap = compute_bounds_overlap(box_a, box_b)
    assert overlap > 0.0

    # Non-overlapping boxes
    box_c = [200.0, 200.0, 300.0, 300.0]
    overlap_zero = compute_bounds_overlap(box_a, box_c)
    assert overlap_zero == 0.0

def test_crs_mismatch_detection():
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.zeros((100, 100, 3), dtype=np.uint8)
    meta_a = {"crs": "EPSG:32644", "bounds": [0, 0, 100, 100]}
    meta_b = {"crs": "EPSG:32643", "bounds": [0, 0, 100, 100]}
    _, _, info = align_image_pair(img_a, img_b, meta_a, meta_b)
    assert info["crs_matched"] is False
    assert info["co_registered"] is False
