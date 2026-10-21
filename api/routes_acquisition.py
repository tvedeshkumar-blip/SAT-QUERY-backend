import os
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.schemas.acquisition import (
    STACSearchRequest, 
    STACSearchResponse, 
    STACRetrieveRequest, 
    STACRetrieveResponse,
    STACWindowedRetrieveRequest,
    STACCompositeRetrieveRequest,
    STACCompositeRetrieveResponse,
    STACValidateAssetRequest,
    STACValidateAssetResponse,
    STACTemporalPairValidateRequest,
    STACTemporalPairValidateResponse,
    STACTemporalSearchRequest,
    STACTemporalSearchResponse,
    STACOpticalSARPairValidateRequest,
    STACOpticalSARPairValidateResponse,
    STACOpticalSARSearchRequest,
    STACOpticalSARSearchResponse,
    CompositeInspectorRequest,
    CompositeInspectorResponse
)
from app.acquisition.stac_service import stac_service
from app.acquisition.asset_validator import AssetValidator
from app.acquisition.temporal_validator import TemporalPairValidator
from app.acquisition.cross_sensor_validator import CrossSensorValidator

logger = logging.getLogger("satquery.api.acquisition")
router = APIRouter()

@router.post("/acquisition/search", response_model=STACSearchResponse)
def search_stac_scenes(request: STACSearchRequest):
    """
    Search satellite acquisitions across public STAC catalogs (Element84 Earth Search AWS)
    by AOI bounds, temporal window, cloud cover threshold, and sensor collection.
    """
    try:
        return stac_service.search(request)
    except Exception as e:
        logger.error(f"Error executing STAC catalog search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/acquisition/retrieve", response_model=STACRetrieveResponse)
def retrieve_stac_asset(request: STACRetrieveRequest):
    """
    Safely download and inspect a validated satellite asset (visual preview or GeoTIFF)
    from a previously discovered scene. Enforces download size limits, trusted hosts,
    and returns base64 data for immediate workspace ingestion.
    """
    try:
        resp = stac_service.retrieve(request)
        if resp.status == "error":
            raise HTTPException(status_code=400, detail=resp.error or "Failed to retrieve asset.")
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving STAC asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/acquisition/retrieve-window", response_model=STACRetrieveResponse)
def retrieve_stac_windowed_asset(request: STACWindowedRetrieveRequest):
    """
    Stage 9: Safely retrieves a bounded spatial window (AOI crop) from a remote COG
    satellite raster using HTTP 206 Range requests.
    Enforces host allowlists, byte budgets, valid georeferencing, and returns
    calibrated, georeferenced in-memory GeoTIFF data ready for scientific workspace ingestion.
    """
    try:
        resp = stac_service.retrieve_window(request)
        if resp.status == "error":
            raise HTTPException(status_code=400, detail=resp.error or "Failed to retrieve windowed asset.")
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving windowed STAC asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquisition/retrieve-composite", response_model=STACCompositeRetrieveResponse)
def retrieve_stac_composite_asset(request: STACCompositeRetrieveRequest):
    """
    Stage 12: Safely retrieves multi-asset Sentinel-2 windowed rasters (B04 Red, B08 NIR, and SCL)
    belonging to the exact same STAC Item and acquisition time under a cumulative <= 20 MB budget cap.
    Aligns grids, computes calibrated surface reflectance NDVI with SCL atmospheric screening,
    and returns a composite GeoTIFF with full physical provenance.
    """
    try:
        resp = stac_service.retrieve_composite_window(request)
        if resp.status == "error":
            raise HTTPException(status_code=400, detail=resp.error or "Failed to retrieve composite window.")
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving composite STAC asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquisition/validate", response_model=STACValidateAssetResponse)
def validate_asset_endpoint(request: STACValidateAssetRequest):
    """
    Physically validates an imported or retrieved asset binary, checking file readability,
    band counts, dimensions, dtype, nodata, CRS, transform, scientific status,
    and analysis task compatibility.
    """
    try:
        data_or_path = request.data_base64
        filename = request.filename or "asset"

        # If data_base64 not provided, search local retrieval storage for the scene/asset
        if not data_or_path and request.scene_id:
            safe_prefix = f"{request.scene_id}_{request.asset_key or 'thumbnail'}"
            if os.path.exists(stac_service.retrieval_dir):
                for fname in os.listdir(stac_service.retrieval_dir):
                    if fname.startswith(safe_prefix):
                        data_or_path = os.path.join(stac_service.retrieval_dir, fname)
                        filename = fname
                        break

        # Also check local samples if local sample ID
        if not data_or_path and request.scene_id and request.scene_id.startswith("local_sample_"):
            name_part = request.scene_id.replace("local_sample_", "") + ".tif"
            sample_p = os.path.join(stac_service.samples_dir, name_part)
            if os.path.exists(sample_p):
                data_or_path = sample_p
                filename = name_part

        if not data_or_path:
            raise HTTPException(
                status_code=400, 
                detail="No asset payload or local file found to validate. Please provide data_base64 or valid scene_id."
            )

        report = AssetValidator.validate_asset(
            data_or_path, 
            filename=filename, 
            provenance=request.provenance
        )
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/acquisition/validate-temporal-pair", response_model=STACTemporalPairValidateResponse)
def validate_temporal_pair_endpoint(request: STACTemporalPairValidateRequest):
    """
    Validates physical compatibility of two temporal candidate satellite assets
    prior to bi-temporal change detection inference.
    Inspects readability, CRS, dimensions, resolution, spatial overlap within AOI,
    modality, acquisition chronology, and cloud confounders.
    Strictly rejects browse-only preview assets (thumbnails) for change detection.
    """
    try:
        report = TemporalPairValidator.validate_pair(
            t1_input=request.t1_data_base64,
            t2_input=request.t2_data_base64,
            t1_filename=request.t1_filename,
            t2_filename=request.t2_filename,
            t1_provenance=request.t1_provenance or {"scene_id": request.t1_scene_id, "asset_key": request.t1_asset_key},
            t2_provenance=request.t2_provenance or {"scene_id": request.t2_scene_id, "asset_key": request.t2_asset_key},
            t1_metadata=request.t1_metadata,
            t2_metadata=request.t2_metadata,
            aoi_bbox=request.aoi_bbox,
            retrieval_dir=stac_service.retrieval_dir,
            samples_dir=stac_service.samples_dir
        )
        return report
    except Exception as e:
        logger.error(f"Error validating temporal pair: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/acquisition/search-temporal-pair", response_model=STACTemporalSearchResponse)
def search_temporal_pair_endpoint(request: STACTemporalSearchRequest):
    """
    Simultaneously queries satellite acquisitions across two distinct temporal windows
    (Pre-Event T1 and Post-Event T2) for the same AOI.
    """
    try:
        # Search T1
        req_1 = STACSearchRequest(
            bbox=request.bbox,
            start_date=request.t1_start_date,
            end_date=request.t1_end_date,
            collection=request.collection,
            max_cloud_cover=request.max_cloud_cover,
            limit=request.limit
        )
        t1_resp = stac_service.search(req_1)

        # Search T2
        req_2 = STACSearchRequest(
            bbox=request.bbox,
            start_date=request.t2_start_date,
            end_date=request.t2_end_date,
            collection=request.collection,
            max_cloud_cover=request.max_cloud_cover,
            limit=request.limit
        )
        t2_resp = stac_service.search(req_2)

        # Determine overall status
        if t1_resp.total_results > 0 and t2_resp.total_results > 0:
            status = "success"
            msg = f"Discovered {t1_resp.total_results} pre-event (T1) and {t2_resp.total_results} post-event (T2) candidate acquisitions."
        elif t1_resp.total_results > 0 or t2_resp.total_results > 0:
            status = "partial"
            msg = f"Partial match: T1 yielded {t1_resp.total_results} scene(s), T2 yielded {t2_resp.total_results} scene(s)."
        else:
            status = "zero_results"
            msg = "No satellite scenes found intersecting the active AOI in either temporal window."

        # Estimate delta days from median timestamps if available
        delta_days_est = None
        try:
            if t1_resp.scenes and t2_resp.scenes:
                d1 = datetime.fromisoformat(t1_resp.scenes[0].datetime.replace("Z", "+00:00"))
                d2 = datetime.fromisoformat(t2_resp.scenes[0].datetime.replace("Z", "+00:00"))
                delta_days_est = round(abs((d2 - d1).total_seconds()) / 86400.0, 1)
        except Exception:
            pass

        return STACTemporalSearchResponse(
            status=status,
            provider=t1_resp.provider or "Element84 Earth Search (AWS)",
            search_bbox=request.bbox,
            t1_results=t1_resp,
            t2_results=t2_resp,
            temporal_delta_days_estimate=delta_days_est,
            message=msg
        )
    except Exception as e:
        logger.error(f"Error executing temporal pair search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquisition/validate-optical-sar-pair", response_model=STACOpticalSARPairValidateResponse)
def validate_optical_sar_pair_endpoint(request: STACOpticalSARPairValidateRequest):
    """
    Validates physical compatibility of an Optical reflectance scene and a SAR microwave backscatter scene
    prior to cross-sensor joint analysis.
    Inspects:
    1. Readability and raster data suitability (rejects browse previews / unprojected thumbnails).
    2. Modality verification (ensures exactly 1 Optical and 1 SAR; rejects optical+optical or sar+sar).
    3. Coordinate Reference System (CRS) matching and georeferencing.
    4. Mutual spatial footprint overlap (rejects 0% or <10% overlap).
    5. Resolution compatibility and temporal baseline separation.
    6. SAR product processing metadata (polarization, calibration status, terrain correction status).
    """
    try:
        report = CrossSensorValidator.validate_pair(
            optical_input=request.optical_data_base64,
            sar_input=request.sar_data_base64,
            optical_filename=request.optical_filename,
            sar_filename=request.sar_filename,
            optical_provenance=request.optical_provenance or {"scene_id": request.optical_scene_id, "asset_key": request.optical_asset_key},
            sar_provenance=request.sar_provenance or {"scene_id": request.sar_scene_id, "asset_key": request.sar_asset_key},
            optical_metadata=request.optical_metadata,
            sar_metadata=request.sar_metadata,
            aoi_bbox=request.aoi_bbox,
            retrieval_dir=stac_service.retrieval_dir,
            samples_dir=stac_service.samples_dir
        )
        return report
    except Exception as e:
        logger.error(f"Error validating optical+SAR cross-sensor pair: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquisition/search-optical-sar-pair", response_model=STACOpticalSARSearchResponse)
def search_optical_sar_pair_endpoint(request: STACOpticalSARSearchRequest):
    """
    Simultaneously queries satellite acquisitions across complementary sensor modalities:
    Optical (Sentinel-2 L2A BOA reflectance) and SAR (Sentinel-1 GRD microwave radar) for the same AOI.
    """
    try:
        # Search Optical (Sentinel-2 L2A)
        opt_col = getattr(request, "optical_collection", None) or "sentinel-2-l2a"
        req_opt = STACSearchRequest(
            bbox=request.bbox,
            start_date=request.optical_start_date,
            end_date=request.optical_end_date,
            collection=opt_col,
            max_cloud_cover=request.max_cloud_cover,
            limit=request.limit
        )
        opt_resp = stac_service.search(req_opt)

        # Search SAR (Sentinel-1 RTC by default for projected COG windowing)
        sar_col = getattr(request, "sar_collection", None) or "sentinel-1-rtc"
        req_sar = STACSearchRequest(
            bbox=request.bbox,
            start_date=request.sar_start_date,
            end_date=request.sar_end_date,
            collection=sar_col,
            max_cloud_cover=None,
            limit=request.limit
        )
        sar_resp = stac_service.search(req_sar)

        # Determine overall status and combined provider display
        sar_label = "Sentinel-1 RTC" if "rtc" in sar_col.lower() else "Sentinel-1"
        provider_display = opt_resp.provider
        if sar_resp.provider and sar_resp.provider != opt_resp.provider:
            provider_display = f"{opt_resp.provider} + {sar_resp.provider}"

        if opt_resp.total_results > 0 and sar_resp.total_results > 0:
            status = "success"
            msg = f"Discovered {opt_resp.total_results} Optical (Sentinel-2) and {sar_resp.total_results} SAR ({sar_label}) candidate acquisitions for the active AOI."
        elif opt_resp.total_results > 0 or sar_resp.total_results > 0:
            status = "partial"
            msg = f"Partial cross-sensor match: Optical yielded {opt_resp.total_results} scene(s), SAR yielded {sar_resp.total_results} scene(s)."
        else:
            status = "zero_results"
            msg = "No satellite scenes found intersecting the active AOI in either sensor modality."

        # Estimate delta days between best candidates
        delta_days_est = None
        try:
            if opt_resp.scenes and sar_resp.scenes:
                d_opt = datetime.fromisoformat(opt_resp.scenes[0].datetime.replace("Z", "+00:00"))
                d_sar = datetime.fromisoformat(sar_resp.scenes[0].datetime.replace("Z", "+00:00"))
                delta_days_est = round(abs((d_sar - d_opt).total_seconds()) / 86400.0, 1)
        except Exception:
            pass

        return STACOpticalSARSearchResponse(
            status=status,
            provider=provider_display,
            search_bbox=request.bbox,
            optical_results=opt_resp,
            sar_results=sar_resp,
            temporal_delta_days_estimate=delta_days_est,
            message=msg
        )
    except Exception as e:
        logger.error(f"Error executing optical+SAR cross-sensor search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acquisition/composite-inspector", response_model=CompositeInspectorResponse)
def get_composite_mask_inspector(request: CompositeInspectorRequest):
    """
    Returns visual evidence layers for multi-scene Sentinel-2 temporal composites:
    1. Red surface reflectance (B04)
    2. NIR surface reflectance (B08)
    3. Valid observation depth (0, 1, 2+)
    4. Categorical SCL cloud and shadow validity mask
    """
    try:
        from app.remote_sensing.temporal import (
            compute_temporal_optical_change, 
            generate_composite_inspector_layers
        )
        from app.remote_sensing.geotiff import parse_geotiff_or_image

        t1_scenes = list(request.t1_scenes or [])
        t2_scenes = list(request.t2_scenes or [])

        # Parse base64 raster if provided
        if request.t1_data_base64 and not t1_scenes:
            arr1, meta1 = parse_geotiff_or_image(request.t1_data_base64, "t1.tif")
            if arr1 is not None and arr1.ndim >= 2:
                b04_1 = arr1[:, :, 0] if arr1.ndim == 3 else arr1
                b08_1 = arr1[:, :, 1] if arr1.ndim == 3 and arr1.shape[2] > 1 else arr1
                scl_1 = arr1[:, :, 2] if arr1.ndim == 3 and arr1.shape[2] > 2 else None
                t1_scenes.append({
                    "scene_id": "t1_input",
                    "b04": b04_1,
                    "b08": b08_1,
                    "scl": scl_1,
                    "meta": meta1,
                    "datetime": meta1.get("acquisition_date") or meta1.get("datetime") or "2023-01-15T10:00:00Z"
                })

        if request.t2_data_base64 and not t2_scenes:
            arr2, meta2 = parse_geotiff_or_image(request.t2_data_base64, "t2.tif")
            if arr2 is not None and arr2.ndim >= 2:
                b04_2 = arr2[:, :, 0] if arr2.ndim == 3 else arr2
                b08_2 = arr2[:, :, 1] if arr2.ndim == 3 and arr2.shape[2] > 1 else arr2
                scl_2 = arr2[:, :, 2] if arr2.ndim == 3 and arr2.shape[2] > 2 else None
                t2_scenes.append({
                    "scene_id": "t2_input",
                    "b04": b04_2,
                    "b08": b08_2,
                    "scl": scl_2,
                    "meta": meta2,
                    "datetime": meta2.get("acquisition_date") or meta2.get("datetime") or "2023-07-15T10:00:00Z"
                })

        if not t1_scenes and not t2_scenes:
            return CompositeInspectorResponse(
                status="unavailable",
                message="No candidate scenes or GeoTIFF data provided for composite inspection.",
                t1={"status": "unavailable", "layers": {}, "stats": {"total_pixels": 0, "valid_pixels": 0, "clear_coverage_pct": 0.0}},
                t2={"status": "unavailable", "layers": {}, "stats": {"total_pixels": 0, "valid_pixels": 0, "clear_coverage_pct": 0.0}},
                bounds=request.aoi_bbox
            )

        res = compute_temporal_optical_change(
            t1_scenes=t1_scenes,
            t2_scenes=t2_scenes,
            composite_method=request.composite_method or "median",
            allow_uncertain_as_clear=request.allow_uncertain_as_clear or False,
            aoi_bounds=request.aoi_bbox
        )

        inspector = res.get("visual_inspector") or {}
        return CompositeInspectorResponse(
            status=inspector.get("status", res.get("status", "success")),
            message=res.get("summary"),
            t1=inspector.get("t1", {}),
            t2=inspector.get("t2", {}),
            bounds=inspector.get("bounds") or request.aoi_bbox,
            crs=inspector.get("crs", "EPSG:4326"),
            composite_method=inspector.get("composite_method", f"valid_pixel_temporal_{request.composite_method or 'median'}"),
            scl_mask_policy=inspector.get("scl_mask_policy"),
            scientific_limitations=res.get("scientific_limitations", [])
        )
    except Exception as e:
        logger.error(f"Error computing composite inspector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


_CACHED_DEMO_BITEMPORAL_PAIR = None

@router.get("/acquisition/demo-bitemporal-pair")
def get_demo_bitemporal_pair():
    """
    Serves the authentic Sentinel-2 T1 (Pre-Monsoon) and T2 (Post-Monsoon)
    STAC-acquired GeoTIFFs from data/retrieved/ for Demo 3 (Bi-Temporal Change Detection).
    Preserves authentic GeoTIFF bytes, CRS (EPSG:32643), spatial bounds, and STAC provenance.
    """
    global _CACHED_DEMO_BITEMPORAL_PAIR
    if _CACHED_DEMO_BITEMPORAL_PAIR is not None:
        return _CACHED_DEMO_BITEMPORAL_PAIR

    import base64
    t1_filename = "S2B_43PGQ_20230528_0_L2A_visual_window_1792262748.tif"
    t2_filename = "S2A_43PGQ_20231119_0_L2A_visual_window_1792262818.tif"

    t1_path = os.path.join(stac_service.retrieval_dir, t1_filename)
    t2_path = os.path.join(stac_service.retrieval_dir, t2_filename)

    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        raise HTTPException(
            status_code=404,
            detail="Real Sentinel-2 bi-temporal GeoTIFF scenes not found in data/retrieved/."
        )

    with open(t1_path, "rb") as f1:
        t1_bytes = f1.read()
    with open(t2_path, "rb") as f2:
        t2_bytes = f2.read()

    t1_b64 = "data:image/tiff;base64," + base64.b64encode(t1_bytes).decode("utf-8")
    t2_b64 = "data:image/tiff;base64," + base64.b64encode(t2_bytes).decode("utf-8")

    t1_provenance = {
        "provider": "Element84 Earth Search (AWS)",
        "catalog_url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "scene_id": "S2B_43PGQ_20230528_0_L2A",
        "datetime": "2023-05-28T05:26:49Z",
        "sensor": "Sentinel-2B MSI Level-2A",
        "modality": "OPTICAL",
        "asset_key": "visual",
        "tile": "43PGQ",
        "cloud_cover": 0.0
    }

    t2_provenance = {
        "provider": "Element84 Earth Search (AWS)",
        "catalog_url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "scene_id": "S2A_43PGQ_20231119_0_L2A",
        "datetime": "2023-11-19T05:27:01Z",
        "sensor": "Sentinel-2A MSI Level-2A",
        "modality": "OPTICAL",
        "asset_key": "visual",
        "tile": "43PGQ",
        "cloud_cover": 0.0
    }

    t1_preview_path = os.path.join(stac_service.retrieval_dir, "S2B_43PGQ_20230528_preview.jpg")
    t2_preview_path = os.path.join(stac_service.retrieval_dir, "S2A_43PGQ_20231119_preview.jpg")
    if os.path.exists(t1_preview_path) and os.path.exists(t2_preview_path):
        with open(t1_preview_path, "rb") as p1:
            t1_preview_b64 = "data:image/jpeg;base64," + base64.b64encode(p1.read()).decode("utf-8")
        with open(t2_preview_path, "rb") as p2:
            t2_preview_b64 = "data:image/jpeg;base64," + base64.b64encode(p2.read()).decode("utf-8")
    else:
        import rasterio
        from app.remote_sensing.preprocessing import convert_array_to_base64_png
        with rasterio.open(t1_path) as ds1:
            t1_preview_b64 = convert_array_to_base64_png(ds1.read())
        with rasterio.open(t2_path) as ds2:
            t2_preview_b64 = convert_array_to_base64_png(ds2.read())

    t1_val = AssetValidator.validate_asset(t1_path, filename=t1_filename, provenance=t1_provenance)
    t2_val = AssetValidator.validate_asset(t2_path, filename=t2_filename, provenance=t2_provenance)

    # Scientific limitation: Explicitly clarify that visual True-Color (RGB) rasters do not contain NIR Band 8
    scientific_limitations = [
        "True-color (RGB) visual Sentinel-2 imagery contains visible reflectance bands (B04, B03, B02).",
        "Calibrated NDVI requires near-infrared Band 8 (B08), which is not present in RGB-only visual assets."
    ]

    payload = {
        "status": "success",
        "mode": "bitemporal",
        "query": "What changed between these two dates?",
        "aoi_bbox": [77.4491, 13.0597, 77.5444, 13.1531],
        "files": [
            {
                "dataUrl": t1_b64,
                "previewDataUrl": t1_preview_b64,
                "previewUrl": "/api/v1/acquisition/demo-bitemporal-preview/t1",
                "filename": t1_filename,
                "role": "primary",
                "isGeoTIFF": True,
                "size": len(t1_bytes),
                "assetCategory": "scientific_raster",
                "stacProvenance": t1_provenance,
                "validation": t1_val,
                "scientificLimitations": scientific_limitations
            },
            {
                "dataUrl": t2_b64,
                "previewDataUrl": t2_preview_b64,
                "previewUrl": "/api/v1/acquisition/demo-bitemporal-preview/t2",
                "filename": t2_filename,
                "role": "secondary",
                "isGeoTIFF": True,
                "size": len(t2_bytes),
                "assetCategory": "scientific_raster",
                "stacProvenance": t2_provenance,
                "validation": t2_val,
                "scientificLimitations": scientific_limitations
            }
        ]
    }
    _CACHED_DEMO_BITEMPORAL_PAIR = payload
    return payload

@router.get("/acquisition/demo-bitemporal-preview/{scene}")
def get_demo_bitemporal_preview(scene: str):
    """
    Serves authentic browser-displayable JPEG preview bytes for Demo 3 scenes ('t1' or 't2').
    """
    from fastapi.responses import FileResponse
    filename = "S2B_43PGQ_20230528_preview.jpg" if scene == "t1" else ("S2A_43PGQ_20231119_preview.jpg" if scene == "t2" else None)
    if not filename:
        raise HTTPException(status_code=404, detail="Scene must be 't1' or 't2'")
    preview_path = os.path.join(stac_service.retrieval_dir, filename)
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail=f"Preview file not found: {filename}")
    return FileResponse(preview_path, media_type="image/jpeg")
