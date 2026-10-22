"""
Stage 14: Agent Tool Registry for Dynamic Multi-Step Remote Sensing Orchestration.

Provides explicit, observable tools with typed inputs, outputs, error handling,
and provenance tracking for:
- STAC catalog search and windowed COG retrieval
- Sentinel-2 calibrated NDVI and SCL quality screening
- Sentinel-1 RTC dual-polarization and DpRVI calculation
- Temporal optical and SAR change detection
- Cross-sensor optical-SAR fusion
- Computer vision specialists (BIT-CD, OWL-ViT, VQA)
- Deterministic verification (Checks A-L)
- Reproducible scientific reporting
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

from app.acquisition.stac_service import stac_service
from app.remote_sensing.bands import compute_calibrated_sentinel2_ndvi
from app.remote_sensing.sar import compute_calibrated_dual_pol_sar
from app.remote_sensing.temporal import compute_temporal_optical_change, compute_temporal_sar_change
from app.agent.registry import model_registry

logger = logging.getLogger("satquery.tools")


class AgentTool(ABC):
    """Abstract base class for all traceable agent tools."""
    def __init__(self, name: str, description: str, category: str):
        self.name = name
        self.description = description
        self.category = category

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Executes the tool and returns a structured output payload with status and provenance."""
        pass


class STACSearchTool(AgentTool):
    """Tool for querying public STAC catalogs for satellite acquisitions."""
    def __init__(self):
        super().__init__(
            name="stac_search",
            description="Search satellite acquisitions by geographic bounding box, temporal window, and sensor collection.",
            category="acquisition"
        )

    def run(self, bbox: List[float], start_date: Optional[str] = None, end_date: Optional[str] = None,
            collection: Optional[str] = None, max_cloud_cover: float = 20.0, limit: int = 5, **kwargs) -> Dict[str, Any]:
        from app.schemas.acquisition import STACSearchRequest
        req = STACSearchRequest(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collection=collection or "sentinel-2-l2a",
            max_cloud_cover=max_cloud_cover,
            limit=limit
        )
        resp = stac_service.search(req)
        scenes_list = getattr(resp, "scenes", getattr(resp, "results", []))
        return {
            "status": resp.status,
            "scenes_found": len(scenes_list),
            "scenes": [
                {
                    "scene_id": getattr(s, "id", getattr(s, "scene_id", None)),
                    "datetime": getattr(s, "datetime", None),
                    "cloud_cover": getattr(s, "cloud_cover", None),
                    "collection": getattr(s, "collection", None),
                    "sensor": getattr(s, "sensor", None),
                    "bbox": getattr(s, "bbox", None)
                }
                for s in scenes_list
            ],
            "provider": getattr(resp, "provider", "STAC Catalog"),
            "catalog_url": getattr(resp, "catalog_url", None)
        }


class STACCompositeRetrieveTool(AgentTool):
    """Tool for windowed acquisition of multi-band composite rasters under a cumulative transfer budget."""
    def __init__(self):
        super().__init__(
            name="stac_composite_retrieve",
            description="Retrieve windowed multi-band satellite rasters (e.g. S2 B04+B08+SCL or S1 VV+VH) with budget cap.",
            category="acquisition"
        )

    def run(self, scene_id: str, aoi_bbox: List[float], collection: Optional[str] = None,
            bands: Optional[List[str]] = None, include_scl: bool = True,
            compute_ndvi: bool = False, compute_dprvi: bool = False,
            max_bytes: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        from app.schemas.acquisition import STACCompositeRetrieveRequest
        req = STACCompositeRetrieveRequest(
            scene_id=scene_id,
            aoi_bbox=aoi_bbox,
            collection=collection or "sentinel-2-l2a",
            bands=bands or (["vv", "vh"] if compute_dprvi else ["b04", "b08"]),
            include_scl=include_scl,
            compute_ndvi=compute_ndvi,
            compute_dprvi=compute_dprvi,
            max_bytes=max_bytes
        )
        resp = stac_service.retrieve_composite_window(req)
        parsed_arr = None
        parsed_meta = {}
        if resp.data_base64:
            try:
                from app.remote_sensing.geotiff import parse_geotiff_or_image
                parsed_arr, parsed_meta = parse_geotiff_or_image(resp.data_base64, resp.filename)
            except Exception:
                pass
        return {
            "status": resp.status,
            "scene_id": resp.scene_id,
            "collection": resp.collection,
            "temporal_role": kwargs.get("temporal_role") or kwargs.get("target_role"),
            "category": resp.category,
            "bands_retrieved": resp.bands_retrieved,
            "size_bytes": resp.size_bytes,
            "ndvi_result": resp.ndvi_result,
            "dprvi_result": resp.dprvi_result,
            "quality_screening": resp.quality_screening or resp.sar_quality_screening,
            "provenance": resp.provenance,
            "data_base64": resp.data_base64,
            "array": parsed_arr,
            "raster_metadata": parsed_meta,
            "scientific_limitations": resp.scientific_limitations,
            "error": resp.error
        }


class Sentinel2NDVIPipelineTool(AgentTool):
    """Tool for calibrated Sentinel-2 surface reflectance NDVI and SCL screening."""
    def __init__(self):
        super().__init__(
            name="sentinel2_ndvi_pipeline",
            description="Compute calibrated Sentinel-2 B04+B08 NDVI with SCL atmospheric quality screening.",
            category="processing"
        )

    def run(self, b04_arr: Any, b08_arr: Any, scl_arr: Optional[Any] = None,
            target_meta: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        res = compute_calibrated_sentinel2_ndvi(
            b04_arr=b04_arr,
            b08_arr=b08_arr,
            scl_arr=scl_arr,
            target_meta=target_meta
        )
        return res


class Sentinel1RTCDpRVIPipelineTool(AgentTool):
    """Tool for calibrated Sentinel-1 RTC dual-polarization SAR backscatter and DpRVI calculation."""
    def __init__(self):
        super().__init__(
            name="sentinel1_rtc_dprvi_pipeline",
            description="Compute calibrated Sentinel-1 RTC dual-pol backscatter power and Mandal et al. (2020) DpRVI.",
            category="processing"
        )

    def run(self, vv_arr: Any, vh_arr: Any, vv_meta: Optional[Dict[str, Any]] = None,
            vh_meta: Optional[Dict[str, Any]] = None, input_is_db: bool = False, **kwargs) -> Dict[str, Any]:
        res = compute_calibrated_dual_pol_sar(
            vv_arr=vv_arr,
            vh_arr=vh_arr,
            vv_meta=vv_meta,
            vh_meta=vh_meta,
            input_is_db=input_is_db
        )
        return res


class TemporalOpticalChangeTool(AgentTool):
    """Tool for bi-temporal Sentinel-2 surface reflectance NDVI change detection with multi-scene SCL screening."""
    def __init__(self):
        super().__init__(
            name="temporal_optical_change",
            description="Compute calibrated bi-temporal Sentinel-2 Delta-NDVI and physical vegetation change with multi-scene SCL composites.",
            category="temporal"
        )

    def run(self, t1_b04: Any = None, t1_b08: Any = None, t2_b04: Any = None, t2_b08: Any = None,
            t1_scl: Optional[Any] = None, t2_scl: Optional[Any] = None,
            t1_meta: Optional[Dict[str, Any]] = None, t2_meta: Optional[Dict[str, Any]] = None,
            t1_datetime: Optional[str] = None, t2_datetime: Optional[str] = None,
            t1_scenes: Optional[List[Dict[str, Any]]] = None, t2_scenes: Optional[List[Dict[str, Any]]] = None,
            composite_method: str = "median",
            loss_threshold: float = -0.15, gain_threshold: float = 0.15,
            cloud_threshold: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        return compute_temporal_optical_change(
            t1_b04=t1_b04,
            t1_b08=t1_b08,
            t2_b04=t2_b04,
            t2_b08=t2_b08,
            t1_scl=t1_scl,
            t2_scl=t2_scl,
            t1_meta=t1_meta,
            t2_meta=t2_meta,
            t1_datetime=t1_datetime,
            t2_datetime=t2_datetime,
            t1_scenes=t1_scenes,
            t2_scenes=t2_scenes,
            composite_method=composite_method,
            loss_threshold=loss_threshold,
            gain_threshold=gain_threshold,
            cloud_threshold=cloud_threshold,
            **kwargs
        )


class TemporalSARChangeTool(AgentTool):
    """Tool for bi-temporal Sentinel-1 RTC dual-polarization change detection (Delta-DpRVI & Delta-dB)."""
    def __init__(self):
        super().__init__(
            name="temporal_sar_change",
            description="Compute calibrated bi-temporal Sentinel-1 RTC Delta-DpRVI and backscatter change.",
            category="temporal"
        )

    def run(self, t1_vv: Any, t1_vh: Any, t2_vv: Any, t2_vh: Any,
            t1_meta: Optional[Dict[str, Any]] = None, t2_meta: Optional[Dict[str, Any]] = None,
            t1_datetime: Optional[str] = None, t2_datetime: Optional[str] = None,
            canopy_loss_threshold: float = -0.15, canopy_gain_threshold: float = 0.15,
            input_is_db: bool = False, **kwargs) -> Dict[str, Any]:
        return compute_temporal_sar_change(
            t1_vv=t1_vv,
            t1_vh=t1_vh,
            t2_vv=t2_vv,
            t2_vh=t2_vh,
            t1_meta=t1_meta,
            t2_meta=t2_meta,
            t1_datetime=t1_datetime,
            t2_datetime=t2_datetime,
            canopy_loss_threshold=canopy_loss_threshold,
            canopy_gain_threshold=canopy_gain_threshold,
            input_is_db=input_is_db
        )


class OpticalSARJointFusionTool(AgentTool):
    """Tool for multimodal optical and SAR cross-sensor fusion analysis."""
    def __init__(self):
        super().__init__(
            name="optical_sar_fusion",
            description="Execute calibrated optical and SAR cross-sensor alignment and joint heuristic segmentation.",
            category="fusion"
        )

    def run(self, images: List[Any], query: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        model = model_registry.get("optical_sar")
        return model.predict(images=images, query=query, metadata=metadata)


class VisualGroundingTool(AgentTool):
    """Tool for open-vocabulary geospatial visual grounding (OWL-ViT)."""
    def __init__(self):
        super().__init__(
            name="visual_grounding",
            description="Ground target geospatial objects and features using open-vocabulary visual localization.",
            category="vision"
        )

    def run(self, images: List[Any], query: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        model = model_registry.get("grounding")
        return model.predict(images=images, query=query, metadata=metadata)


class NeuralChangeDetectionTool(AgentTool):
    """Tool for deep learning bi-temporal change detection (BIT-CD)."""
    def __init__(self):
        super().__init__(
            name="neural_change_detection",
            description="Detect surface land-cover changes using Bitemporal Image Transformer (BIT-CD).",
            category="vision"
        )

    def run(self, images: List[Any], query: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        model = model_registry.get("change_detection")
        return model.predict(images=images, query=query, metadata=metadata)


class VQACaptioningTool(AgentTool):
    """Tool for vision-language questioning and spectral scene captioning."""
    def __init__(self):
        super().__init__(
            name="vqa_captioning",
            description="Answer visual questions or produce spectral captioning summaries for satellite scenes.",
            category="vision"
        )

    def run(self, images: List[Any], query: str, mode: str = "vqa", metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        task_key = "captioning" if mode == "captioning" else "vqa"
        model = model_registry.get(task_key)
        return model.predict(images=images, query=query, metadata=metadata)


class DeterministicVerifierTool(AgentTool):
    """Tool for deterministic evidence package verification (Checks A-L)."""
    def __init__(self):
        super().__init__(
            name="deterministic_verifier",
            description="Verify physical integrity, boundary conditions, units, and consistency across Checks A through L.",
            category="verification"
        )

    def run(self, evidence_package: Any, candidate_claims: Optional[List[Any]] = None,
            answer_text: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        from app.agent.verifier import DeterministicVerifier
        passed, failures, claims, contradictions = DeterministicVerifier.verify(
            evidence_package=evidence_package,
            candidate_claims=candidate_claims,
            answer_text=answer_text
        )
        status = "supported" if passed and not contradictions else ("partially_supported" if passed else "unsupported")
        return {
            "passed": passed,
            "deterministic_passed": passed,
            "verification_status": status,
            "failures": failures,
            "deterministic_failures": failures,
            "claims": [c.dict() if hasattr(c, "dict") else c.model_dump() for c in claims],
            "contradictions": contradictions,
            "unsupported_claims": [],
            "summary_explanation": f"Deterministic verification {'PASSED' if passed else 'FAILED with ' + str(len(failures)) + ' violations'}.",
            "missing_evidence": [],
            "scientific_limitations": getattr(evidence_package, "known_limitations", []) or [],
            "recommended_checks": []
        }


class LandCoverClassificationTool(AgentTool):
    """Tool for multi-label remote sensing land-cover classification (BigEarthNet ResNet-50)."""
    def __init__(self):
        super().__init__(
            name="land_cover_classification",
            description="Classify Sentinel-2 10-band multispectral rasters into 19 CORINE land-cover classes.",
            category="vision"
        )

    def run(self, images: List[Any], query: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        if not images or len(images) == 0:
            return {
                "status": "failed",
                "error": "No input satellite raster provided for land-cover classification. Sentinel-2 10-band multispectral data is required."
            }
        model = model_registry.get("classification")
        return model.predict(images=images, query=query, metadata=metadata, **kwargs)


class AgentToolRegistry:
    """Central registry and lookup dispatcher for all remote sensing agent tools."""
    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        tools = [
            STACSearchTool(),
            STACCompositeRetrieveTool(),
            Sentinel2NDVIPipelineTool(),
            Sentinel1RTCDpRVIPipelineTool(),
            TemporalOpticalChangeTool(),
            TemporalSARChangeTool(),
            OpticalSARJointFusionTool(),
            VisualGroundingTool(),
            NeuralChangeDetectionTool(),
            VQACaptioningTool(),
            LandCoverClassificationTool(),
            DeterministicVerifierTool(),
        ]
        for t in tools:
            self._tools[t.name] = t

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description, "category": t.category}
            for t in self._tools.values()
        ]


# Singleton instance
tool_registry = AgentToolRegistry()
