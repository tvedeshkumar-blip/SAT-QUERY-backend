import numpy as np
from fastapi import APIRouter, HTTPException
from app.schemas.analysis import AnalysisRequest, AnalysisResponseSchema
from app.schemas.verification import VerificationRequestSchema, VerificationResponseSchema
from app.agent.verifier import scientific_verifier, EvidencePackageBuilder
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
    if not request.images or len(request.images) == 0:
        raise HTTPException(status_code=400, detail="No image provided. Single-Image VQA requires an input satellite image.")
    request.mode = "vqa"
    return analyze_imagery(request)

@router.post("/analyze/caption", response_model=AnalysisResponseSchema)
def analyze_caption(request: AnalysisRequest):
    if not request.images or len(request.images) == 0:
        raise HTTPException(status_code=400, detail="No image provided. Captioning requires an input satellite image.")
    request.mode = "captioning"
    return analyze_imagery(request)

@router.post("/analyze/grounding", response_model=AnalysisResponseSchema)
def analyze_grounding(request: AnalysisRequest):
    if not request.images or len(request.images) == 0:
        raise HTTPException(status_code=400, detail="No image provided. Visual grounding requires an input satellite image.")
    request.mode = "grounding"
    return analyze_imagery(request)

@router.post("/analyze/classify", response_model=AnalysisResponseSchema)
def analyze_classify(request: AnalysisRequest):
    request.mode = "classification"
    return analyze_imagery(request)

@router.post("/analyze/classification", response_model=AnalysisResponseSchema)
def analyze_classification_alias(request: AnalysisRequest):
    request.mode = "classification"
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

@router.post("/analyze/orchestrate", response_model=AnalysisResponseSchema)
def analyze_orchestrate(request: AnalysisRequest):
    request.mode = "orchestration"
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

        # Enforce scientific integrity: reject uncalibrated visual preview browse assets
        is_preview = (
            getattr(img_in, "asset_category", None) == "visual_preview"
            or (img_in.filename or "").lower().endswith((".jpg", ".jpeg"))
        )
        if is_preview:
            raise HTTPException(
                status_code=400,
                detail="Scientific integrity validation failed: Quantitative spectral index calculation is strictly forbidden on visual preview browse imagery. Calibrated NDVI/NDWI requires verified multispectral surface reflectance bands (Red Band 4 and NIR Band 8)."
            )


        # Enforce band requirement: single-band raster cannot compute NDVI
        if meta.get("bands", 1) == 1:
            raise HTTPException(
                status_code=400,
                detail=f"Scientific integrity validation failed: Asset '{img_in.filename}' contains only 1 spectral channel. NDVI calculation requires distinct multispectral Red and NIR bands."
            )


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

@router.post("/analyze/verify", response_model=VerificationResponseSchema)
def verify_analysis_endpoint(request: VerificationRequestSchema):
    """
    Dedicated Stage 7 Scientific Verification endpoint.
    Performs deterministic integrity checks and LLM-assisted verification
    over a structured evidence package or prior analysis response.
    """
    try:
        if request.evidence_package:
            pkg = request.evidence_package
        elif request.analysis_response:
            pkg = EvidencePackageBuilder.from_analysis_response(request.analysis_response)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'evidence_package' or 'analysis_response' must be provided."
            )

        return scientific_verifier.verify(
            evidence_package=pkg,
            candidate_claims=request.custom_claims
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze/report")
def generate_report_endpoint(payload: dict):
    """
    Stage 8: Generates or returns a validated ScientificReportSchema
    from an analysis response dictionary.
    """
    try:
        from app.reports.scientific_report import ScientificReportBuilder
        from app.schemas.report import ScientificReportSchema
        if "reproducible_report" in payload and payload["reproducible_report"]:
            rep = payload["reproducible_report"]
            if isinstance(rep, dict):
                return ScientificReportSchema(**rep)
            if isinstance(rep, ScientificReportSchema):
                return rep
        return ScientificReportBuilder.build_report(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate scientific report: {str(e)}")
