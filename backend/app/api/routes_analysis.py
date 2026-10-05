import numpy as np
from fastapi import APIRouter, HTTPException
from app.schemas.analysis import AnalysisRequest, AnalysisResponseSchema
from app.agent.controller import agent_controller

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponseSchema)
def analyze_imagery(request: AnalysisRequest):
    try:
        return agent_controller.process_request(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/query", response_model=AnalysisResponseSchema)
def query_alias(request: AnalysisRequest):
    return analyze_imagery(request)

@router.post("/analyze/vqa", response_model=AnalysisResponseSchema)
def analyze_vqa(request: AnalysisRequest):
    request.mode = "single"
    return analyze_imagery(request)

@router.post("/analyze/caption", response_model=AnalysisResponseSchema)
def analyze_caption(request: AnalysisRequest):
    request.mode = "single"
    return analyze_imagery(request)

@router.post("/analyze/grounding", response_model=AnalysisResponseSchema)
def analyze_grounding(request: AnalysisRequest):
    request.mode = "single"
    return analyze_imagery(request)

@router.post("/analyze/change", response_model=AnalysisResponseSchema)
def analyze_change(request: AnalysisRequest):
    request.mode = "bitemporal"
    return analyze_imagery(request)

@router.post("/analyze/change-vqa", response_model=AnalysisResponseSchema)
def analyze_change_vqa(request: AnalysisRequest):
    request.mode = "bitemporal"
    return analyze_imagery(request)

@router.post("/analyze/optical-sar", response_model=AnalysisResponseSchema)
def analyze_optical_sar(request: AnalysisRequest):
    request.mode = "optical_sar"
    return analyze_imagery(request)

@router.post("/analyze/spectral-indices")
def analyze_spectral_indices(request: AnalysisRequest):
    try:
        from app.remote_sensing.geotiff import parse_geotiff_or_image
        from app.remote_sensing.bands import compute_ndvi, compute_ndwi, compute_false_color_cir, compute_sar_decibel_and_filter
        from app.remote_sensing.preprocessing import convert_array_to_base64_png

        if not request.images or len(request.images) == 0:
            raise HTTPException(status_code=400, detail="No image provided.")

        img_in = request.images[0]
        arr, meta = parse_geotiff_or_image(img_in.data, img_in.filename)

        ndvi_f, ndvi_col = compute_ndvi(arr)
        ndwi_f, ndwi_col = compute_ndwi(arr)
        cir_rgb = compute_false_color_cir(arr)
        sar_db, sar_stats = compute_sar_decibel_and_filter(arr)

        return {
            "status": "success",
            "metadata": meta,
            "indices": {
                "ndvi": {
                    "data_base64": convert_array_to_base64_png(ndvi_col),
                    "mean_ndvi": round(float(np.mean(ndvi_f)), 3),
                    "max_ndvi": round(float(np.max(ndvi_f)), 3)
                },
                "ndwi": {
                    "data_base64": convert_array_to_base64_png(ndwi_col),
                    "mean_ndwi": round(float(np.mean(ndwi_f)), 3)
                },
                "cir": {
                    "data_base64": convert_array_to_base64_png(cir_rgb)
                },
                "sar_db": {
                    "data_base64": convert_array_to_base64_png(sar_db),
                    "stats": sar_stats
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
