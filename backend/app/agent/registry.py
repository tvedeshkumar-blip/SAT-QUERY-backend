import logging
from typing import Dict, Any, Optional, List
from app.models.base import BaseModel
from app.models.vqa.vqa_model import GenericVLMOrchestrator
from app.models.captioning.captioner import ClassicalSpectralCaptioner
from app.models.grounding.grounder import GroundingProvider, ClassicalBaselineGrounder
from app.models.change_detection.change_detector import ChangeDetectionProvider
from app.models.change_vqa.change_vqa import ChangeVQA
from app.models.optical_sar.optical_sar_fusion import OpticalSARJointAnalysisProvider

logger = logging.getLogger("satquery.registry")

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._init_registry()

    def _init_registry(self):
        """
        Registers remote sensing models, providers, and explicit baseline adapters.
        Honest declarations: un-trained heuristics are designated as 'baseline'.
        """
        # 1. VQA
        self.register(
            "vqa", 
            GenericVLMOrchestrator(),
            metadata={
                "implementation": "GenericVLMOrchestrator (Florence-2-base / Gemini API / Spectral Baseline)",
                "model_source": "Multi-Provider (microsoft/Florence-2-base / gemini-2.5-flash / local heuristics)",
                "modality": "OPTICAL / MULTISPECTRAL",
                "input_requirements": "Single satellite raster (1+ bands)",
                "device_requirements": "CPU / Network API / GPU (optional)",
                "license": "MIT / Apache-2.0"
            }
        )

        # 2. Captioning
        self.register(
            "captioning", 
            ClassicalSpectralCaptioner(),
            metadata={
                "implementation": "ClassicalSpectralCaptioner (Band statistics and albedo dispersion)",
                "model_source": "Local procedural spectral heuristics",
                "modality": "OPTICAL / MULTISPECTRAL",
                "input_requirements": "Single satellite raster (1+ bands)",
                "device_requirements": "CPU",
                "license": "Apache-2.0"
            }
        )

        # 3. Grounding
        self.register(
            "grounding", 
            GroundingProvider(),
            metadata={
                "implementation": "GroundingProvider (OWL-ViT Open-Vocabulary Grounder / Classical Baseline fallback)",
                "model_source": "google/owlvit-base-patch32 + OpenCV/SciPy contouring baseline",
                "modality": "OPTICAL / MULTISPECTRAL",
                "input_requirements": "Single satellite raster + target query string",
                "device_requirements": "CPU / GPU",
                "license": "Apache-2.0"
            }
        )

        # 4. Change Detection
        self.register(
            "change_detection", 
            ChangeDetectionProvider(),
            metadata={
                "implementation": "ChangeDetectionProvider (BIT-CD Transformer / PixelDifferenceChangeBaseline fallback)",
                "model_source": "Bitemporal Image Transformer (BIT-CD) + Rasterio differencing baseline",
                "modality": "BI-TEMPORAL OPTICAL / SAR",
                "input_requirements": "2 co-registered or rescaled rasters (T1 & T2)",
                "device_requirements": "CPU / GPU",
                "license": "MIT / Apache-2.0"
            }
        )

        # 5. Change VQA
        self.register(
            "change_vqa", 
            ChangeVQA(),
            metadata={
                "implementation": "EvidenceGroundedChangeVQA (Consumes difference evidence)",
                "model_source": "Structured evidence synthesis pipeline",
                "modality": "BI-TEMPORAL OPTICAL / SAR",
                "input_requirements": "2 rasters (T1 & T2) + change query string",
                "device_requirements": "CPU",
                "license": "Apache-2.0"
            }
        )

        # 6. Optical + SAR
        self.register(
            "optical_sar", 
            OpticalSARJointAnalysisProvider(),
            metadata={
                "implementation": "OpticalSARJointAnalysisProvider (Calibrated dB Scaling & Cross-Modal Composite)",
                "model_source": "Radiometric cross-modal joint analysis",
                "modality": "OPTICAL + SAR DUAL MODALITY",
                "input_requirements": "2 rasters (Optical reflectance + SAR backscatter)",
                "device_requirements": "CPU",
                "license": "Apache-2.0"
            }
        )

    def register(self, task_type: str, model: BaseModel, metadata: Optional[Dict[str, Any]] = None):
        self._models[task_type] = model
        self._metadata[task_type] = metadata or {}
        logger.info(f"Registered adapter '{model.model_name}' for task '{task_type}' [status: {model.status}]")

    def get_model(self, task_type: str) -> BaseModel:
        if task_type not in self._models:
            raise KeyError(f"No specialist model or baseline registered for task type '{task_type}'")
        
        model = self._models[task_type]
        if not model.is_loaded and model.implementation_status not in ("baseline", "unavailable"):
            logger.info(f"Loading weights for task '{task_type}' ({model.model_name})...")
            model.load()
        return model

    def list_models(self) -> Dict[str, str]:
        return {task: model.model_name for task, model in self._models.items()}

    def list_models_detailed(self) -> List[Dict[str, Any]]:
        entries = []
        for task, model in self._models.items():
            meta = self._metadata.get(task, {})
            entries.append({
                "name": model.model_name,
                "task": task,
                "implementation": meta.get("implementation", model.model_name),
                "version": model.version,
                "model_source": meta.get("model_source", "local"),
                "license": model.license,
                "modality": meta.get("modality", "OPTICAL"),
                "input_requirements": meta.get("input_requirements", "Satellite raster"),
                "device_requirements": meta.get("device_requirements", "CPU"),
                "status": model.status,
                "implementation_status": model.implementation_status,
                "is_trained": model.is_trained,
                "is_remote_sensing_adapted": model.is_remote_sensing_adapted
            })
        return entries

# Global Registry Instance
model_registry = ModelRegistry()
