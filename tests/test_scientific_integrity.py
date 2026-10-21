import pytest
import os
import io
import base64
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from fastapi.testclient import TestClient

from app.main import app
from app.remote_sensing.geotiff import parse_geotiff_or_image
from app.agent.registry import model_registry
from app.evaluation.synthetic_validator import SyntheticPipelineValidator

client = TestClient(app)

def create_raw_tif_b64(with_crs_tag: bool = False):
    img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (2.5, 2.5, 0.0) # ModelPixelScaleTag
    ifd[33922] = (0.0, 0.0, 0.0, 432000.0, 1420000.0, 0.0) # ModelTiepointTag
    if with_crs_tag:
        ifd[34737] = b"WGS 84 / UTM zone 44N|EPSG:32644|"
    buf = io.BytesIO()
    img.save(buf, format="TIFF", tiffinfo=ifd)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def test_1_no_hardcoded_benchmark_scores():
    """Rule 1: GET /api/v1/evaluation must not return fabricated scores."""
    res = client.get("/api/v1/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("not_evaluated", "completed")
    
    # If not evaluated, results must be empty or contain completed status only
    forbidden_fabricated_scores = ["84.6%", "62.4%", "79.2%", "88.1%"]
    for item in data.get("results", []):
        assert item.get("evaluation_status") != "not_evaluated" or item.get("score") is None
        assert item.get("score") not in forbidden_fabricated_scores

def test_2_no_fake_confidence_values():
    """Rule 2: Models without calibrated confidence must report confidence = None."""
    # Test VQA
    vqa_model = model_registry.get_model("vqa")
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = vqa_model.predict([dummy_img], query="Test query")
    assert res["confidence"] is None
    assert "not" in res.get("confidence_label", "").lower()

    # Test Captioning
    cap_model = model_registry.get_model("captioning")
    res_cap = cap_model.predict([dummy_img])
    assert res_cap["confidence"] is None

def test_3_no_guessed_crs():
    """Rule 3: Missing CRS must return None and 'CRS unavailable', never guessing EPSG:32644."""
    # A. Raster with pixel scale & tiepoint but missing CRS tag
    b64_no_crs = create_raw_tif_b64(with_crs_tag=False)
    arr, meta = parse_geotiff_or_image(b64_no_crs, "test_nocrs.tif")
    assert meta["crs"] is None
    assert meta["crs_display"] == "CRS unavailable"

    # B. Plain PNG without any spatial tags
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    arr_png, meta_png = parse_geotiff_or_image(png_b64, "scene.png")
    assert meta_png["crs"] is None
    assert meta_png["crs_display"] == "CRS unavailable"

def test_4_synthetic_evaluation_is_labelled_synthetic():
    """Rule 4: Synthetic validation must explicitly declare itself as synthetic."""
    res = client.post("/api/v1/evaluation/synthetic-validation", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["is_official_isro_evaluation"] is False
    assert "SYNTHETIC" in data["disclaimer"].upper()
    assert data["validation_mode"] in ("procedural_synthetic", "user_provided_pair")

def test_5_unloaded_models_not_reported_as_loaded():
    """Rule 5: Models without initialized tensor weights must NOT report status = 'loaded'."""
    detailed_models = model_registry.list_models_detailed()
    for m in detailed_models:
        if m["implementation_status"] == "baseline":
            assert m["status"] == "baseline"
        elif m["implementation_status"] == "unavailable":
            assert m["status"] == "unavailable"
        elif m["implementation_status"] == "demo":
            assert m["status"] in ("demo", "baseline")

def test_6_missing_weights_do_not_produce_trained_claims():
    """Rule 6: Untrained heuristic models must not claim is_trained=True or is_remote_sensing_adapted=True."""
    detailed_models = model_registry.list_models_detailed()
    for m in detailed_models:
        if m["implementation_status"] in ("baseline", "demo"):
            assert m["is_trained"] is False
            assert m["is_remote_sensing_adapted"] is False
        elif m["implementation_status"] == "unavailable":
            assert m["is_trained"] is False


def test_7_bitemporal_coregistration_integrity_and_no_contradiction():
    """Rule 7: Real co-registered GeoTIFF scenes must resolve co_registered=True without contradiction."""
    t1_path = "data/retrieved/S2B_43PGQ_20230528_0_L2A_visual_window_1792262748.tif"
    t2_path = "data/retrieved/S2A_43PGQ_20231119_0_L2A_visual_window_1792262818.tif"
    if os.path.exists(t1_path) and os.path.exists(t2_path):
        with open(t1_path, "rb") as f:
            b64_1 = base64.b64encode(f.read()).decode("utf-8")
        with open(t2_path, "rb") as f:
            b64_2 = base64.b64encode(f.read()).decode("utf-8")
        fn1, fn2 = os.path.basename(t1_path), os.path.basename(t2_path)
    else:
        b64_1 = create_raw_tif_b64(with_crs_tag=True)
        b64_2 = create_raw_tif_b64(with_crs_tag=True)
        fn1, fn2 = "scene_t1.tif", "scene_t2.tif"

    payload = {
        "query": "What changed between May 2023 and November 2023 in this scene?",
        "mode": "bitemporal",
        "images": [
            {
                "data": b64_1,
                "filename": fn1,
                "role": "primary",
                "modality": "OPTICAL",
                "asset_category": "scientific_multispectral",
                "provenance": {
                    "collection": "sentinel-2-l2a",
                    "scene_id": fn1,
                    "datetime": "2023-05-28T05:46:49Z",
                    "bands": ["B04", "B03", "B02"],
                    "asset_key": "visual"
                }
            },
            {
                "data": b64_2,
                "filename": fn2,
                "role": "secondary",
                "modality": "OPTICAL",
                "asset_category": "scientific_multispectral",
                "provenance": {
                    "collection": "sentinel-2-l2a",
                    "scene_id": fn2,
                    "datetime": "2023-11-19T05:51:21Z",
                    "bands": ["B04", "B03", "B02"],
                    "asset_key": "visual"
                }
            }
        ]
    }

    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    # Deterministic verifier must pass with zero contradictions
    verification = data.get("verification", {})
    assert verification.get("deterministic_passed") is True
    assert len(verification.get("contradictions", [])) == 0

    # Registration status must establish co_registered=True
    report = data.get("reproducible_report", {})
    prov = report.get("provenance", {})
    reg_status = prov.get("registration_status", {})
    assert reg_status.get("co_registered") is True
    assert reg_status.get("georeferenced") is True

    # Provenance source assets must match actual scenes
    assert fn1 in prov.get("source_asset_ids", [])
    assert fn2 in prov.get("source_asset_ids", [])

    # Quantitative measurements must be grounded in ev_change_map
    measurements = report.get("physical_measurements", [])
    assert len(measurements) >= 3
    pm_names = [pm["name"] for pm in measurements]
    assert "bitemporal_changed_pixel_count" in pm_names
    assert "bitemporal_total_scene_pixels" in pm_names
    assert "bitemporal_difference_threshold" in pm_names

    for pm in measurements:
        assert len(pm["source_artifact_ids"]) > 0
        assert "ev_change_map" in pm["source_artifact_ids"]


def test_8_unreferenced_bitemporal_never_claims_coregistration():
    """Rule 8: Unreferenced or CRS-absent rasters must never claim co-registration or physical area."""
    buf1 = io.BytesIO()
    Image.new("RGB", (64, 64), color=(60, 60, 60)).save(buf1, format="PNG")
    b64_1 = base64.b64encode(buf1.getvalue()).decode("utf-8")

    buf2 = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(buf2, format="PNG")
    b64_2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    payload = {
        "query": "Identify changes between these dates.",
        "mode": "bitemporal",
        "images": [
            {"data": b64_1, "filename": "unref_t1.png", "role": "primary"},
            {"data": b64_2, "filename": "unref_t2.png", "role": "secondary"}
        ]
    }

    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    # Must NOT claim co-registration
    report = data.get("reproducible_report", {})
    reg_status = report.get("provenance", {}).get("registration_status", {})
    assert reg_status.get("co_registered") is False
    assert reg_status.get("georeferenced") is False

    # Verifier must pass without contradiction
    v = data.get("verification", {})
    assert v.get("deterministic_passed") is True
    assert len(v.get("contradictions", [])) == 0

    # Physical area in hectares must NOT be reported without uniform metric CRS
    pm_names = [pm["name"] for pm in report.get("physical_measurements", [])]
    assert "bitemporal_changed_area_hectares" not in pm_names


def test_9_bitemporal_baseline_discloses_fallback_and_no_neural_claim():
    """Rule 9: PixelDifferenceChangeBaseline must disclose fallback and not claim neural model."""
    t1_path = "data/retrieved/S2B_43PGQ_20230528_0_L2A_visual_window_1792262748.tif"
    t2_path = "data/retrieved/S2A_43PGQ_20231119_0_L2A_visual_window_1792262818.tif"
    if os.path.exists(t1_path) and os.path.exists(t2_path):
        with open(t1_path, "rb") as f:
            b64_1 = base64.b64encode(f.read()).decode("utf-8")
        with open(t2_path, "rb") as f:
            b64_2 = base64.b64encode(f.read()).decode("utf-8")
    else:
        b64_1 = create_raw_tif_b64(with_crs_tag=True)
        b64_2 = create_raw_tif_b64(with_crs_tag=True)

    payload = {
        "query": "Detect change map",
        "mode": "bitemporal",
        "images": [
            {"data": b64_1, "filename": "t1.tif", "role": "primary"},
            {"data": b64_2, "filename": "t2.tif", "role": "secondary"}
        ]
    }

    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data.get("fallback_used") is True
    assert data.get("actual_model_used") == "PixelDifferenceChangeBaseline"
    assert data.get("implementation_status") == "baseline"

    # Verifier confirms no neural model claim was accepted
    v = data.get("verification", {})
    assert v.get("deterministic_passed") is True
    assert len(v.get("contradictions", [])) == 0


def test_10_missing_bands_keep_ndvi_sar_unavailable():
    """Rule 10: RGB visual inputs must not fabricate calibrated NDVI or SAR measurements."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 100)).save(buf, format="PNG")
    b64_rgb = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "query": "Calculate NDVI vegetation index and radar backscatter.",
        "mode": "single",
        "images": [
            {"data": b64_rgb, "filename": "rgb_visual.png", "role": "primary"}
        ]
    }

    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    # Calibrated NDVI and SAR measurements must remain absent/unavailable from RGB-only inputs
    rep = data.get("reproducible_report", {})
    pms = rep.get("physical_measurements", [])
    assert not any("ndvi" in pm["name"].lower() or "dprvi" in pm["name"].lower() or "gamma" in pm["name"].lower() for pm in pms)
    prov = rep.get("provenance", {})
    assert prov.get("ndvi_status") is None or prov.get("ndvi_status") != "available"
    assert prov.get("dprvi_status") is None or prov.get("dprvi_status") != "available"

    # Verifier must reject genuine calibrated NDVI claims on RGB-only inputs
    from app.agent.verifier import DeterministicVerifier, EvidencePackageBuilder
    from app.schemas.verification import ClaimVerificationItem

    evidence_pkg = EvidencePackageBuilder.from_analysis_response({
        "metadata": {"asset_category": "visual_preview"},
        "evidence": [{"id": "visual_rgb", "type": "preview"}]
    })

    claim = ClaimVerificationItem(
        claim_text="Genuine calibrated NDVI for this vegetation scene is 0.65.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["visual_rgb"]
    )

    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        evidence_package=evidence_pkg,
        candidate_claims=[claim]
    )
    assert passed is False
    assert any("RGB-only" in f or "B04" in f or "zero physical" in f for f in failures)


def test_11_demo3_area_and_registration_validation():
    """Rule 11: Demo 3 changed area must derive from authentic 10m x 10m pixel dimensions and verified co-registration."""
    import rasterio

    t1_path = "data/retrieved/S2B_43PGQ_20230528_0_L2A_visual_window_1792262748.tif"
    t2_path = "data/retrieved/S2A_43PGQ_20231119_0_L2A_visual_window_1792262818.tif"
    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        return

    with rasterio.open(t1_path) as ds1:
        res1 = ds1.res
        crs1 = str(ds1.crs)
        transform1 = list(ds1.transform[:6])
        bounds1 = list(ds1.bounds)
        w1, h1 = ds1.width, ds1.height
        masks1 = ds1.read_masks()

    with rasterio.open(t2_path) as ds2:
        res2 = ds2.res
        crs2 = str(ds2.crs)
        transform2 = list(ds2.transform[:6])
        bounds2 = list(ds2.bounds)
        w2, h2 = ds2.width, ds2.height
        masks2 = ds2.read_masks()

    # 1. Pixel dimensions and units
    assert res1 == (10.0, 10.0)
    assert res2 == (10.0, 10.0)
    assert crs1 == "EPSG:32643"
    assert crs2 == "EPSG:32643"
    assert w1 == 1024 and h1 == 1024
    assert w2 == 1024 and h2 == 1024

    # 2. Valid data extent (zero nodata)
    assert np.all(masks1 == 255)
    assert np.all(masks2 == 255)

    # 3. Identical grid verification
    assert crs1 == crs2
    assert transform1 == transform2
    assert bounds1 == bounds2

    # 4. End-to-end pipeline verification
    with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2:
        b64_1 = base64.b64encode(f1.read()).decode("utf-8")
        b64_2 = base64.b64encode(f2.read()).decode("utf-8")

    payload = {
        "query": "What changed between these two dates?",
        "mode": "bitemporal",
        "images": [
            {"data": b64_1, "filename": os.path.basename(t1_path), "role": "primary"},
            {"data": b64_2, "filename": os.path.basename(t2_path), "role": "secondary"}
        ]
    }

    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    ev_map = next(e for e in data["evidence"] if e["id"] == "ev_change_map")
    stats = ev_map["statistics"]

    # Verify pixel count and area
    assert stats["total_scene_pixels"] == 1048576
    assert stats["changed_pixel_count"] == 853675
    assert stats["changed_area_percent"] == "81.41%"
    assert stats["changed_area_hectares"] == 8536.75  # 853675 * (10m * 10m) / 10000 = 8536.75 ha
    assert stats["registration_method"] == "identical_grid_co_registered"
    assert stats["co_registered"] is True
    assert stats["georeferenced"] is True
    assert "Unconfirmed" in stats["confirmed_real_world_status"]

    # Verify reproducible report provenance has T1 and T2 timestamps
    rep = data["reproducible_report"]
    prov = rep["provenance"]
    assert prov.get("temporal_t1_datetime") is not None
    assert prov.get("temporal_t2_datetime") is not None

