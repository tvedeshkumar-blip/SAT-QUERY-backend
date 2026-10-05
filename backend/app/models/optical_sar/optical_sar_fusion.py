import logging
import os
import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from app.models.base import BaseModel
from app.remote_sensing.registration import align_image_pair
from app.remote_sensing.preprocessing import (
    convert_array_to_base64_png, 
    normalize_to_uint8,
    robust_remote_sensing_preprocess
)

logger = logging.getLogger("satquery.optical_sar")

class BaseOpticalSARFusion(BaseModel):
    """Abstract base adapter for Optical + SAR multimodal fusion."""
    def __init__(self, model_name: str, implementation_status: str = "baseline"):
        super().__init__(
            model_name=model_name,
            task_type="optical_sar",
            implementation_status=implementation_status,
            is_trained=False,
            is_remote_sensing_adapted=False
        )

    @abstractmethod
    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        pass

class OpticalSARVisualizationBaseline(BaseOpticalSARFusion):
    """
    Classical Optical + SAR linear composite baseline.
    Preserved as a fast, honest reference baseline.
    """
    def __init__(
        self,
        water_intensity_threshold: float = 40.0,
        urban_intensity_threshold: float = 200.0
    ):
        super().__init__(
            model_name="OpticalSARVisualizationBaseline",
            implementation_status="baseline"
        )
        self.water_threshold = water_intensity_threshold
        self.urban_threshold = urban_intensity_threshold

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if len(images) < 2:
            raise ValueError("Optical+SAR analysis requires 2 images (Optical & SAR).")

        meta = metadata or {}
        meta_opt = meta.get("optical") or meta.get("primary", {})
        meta_sar = meta.get("sar") or meta.get("secondary", {})

        img_opt, img_sar = images[0], images[1]
        img_opt, img_sar_aligned, reg_info = align_image_pair(img_opt, img_sar, meta_opt, meta_sar)

        h, w = img_opt.shape[:2]
        opt_u8, _ = robust_remote_sensing_preprocess(img_opt, meta_opt, target_modality="OPTICAL")
        sar_u8, _ = robust_remote_sensing_preprocess(img_sar_aligned, meta_sar, target_modality="SAR")

        if opt_u8.ndim == 3 and opt_u8.shape[2] >= 3:
            opt_rgb = opt_u8[:, :, :3]
        else:
            gray_opt = opt_u8 if opt_u8.ndim == 2 else opt_u8[:, :, 0]
            opt_rgb = np.stack([gray_opt]*3, axis=-1)

        sar_gray = sar_u8 if sar_u8.ndim == 2 else sar_u8[:, :, 0]
        fused_rgb = (0.5 * opt_rgb.astype(float) + 0.5 * np.stack([sar_gray]*3, axis=-1).astype(float)).astype(np.uint8)

        fused_b64 = convert_array_to_base64_png(fused_rgb)

        answer = (
            f"Dual-modality Optical + SAR linear blend baseline generated across {w}x{h} scene. "
            f"Co-registration status: {reg_info.get('co_registered')}."
        )

        return {
            "answer": answer,
            "model_type": "visualization_baseline",
            "implementation_status": "baseline",
            "confidence": None,
            "confidence_label": "Not available (Visualization Baseline)",
            "models": [self.model_name],
            "registration_info": reg_info,
            "evidence": [
                {
                    "id": "ev_fused_scene",
                    "type": "fused",
                    "title": "50/50 Dual-Modality Linear Composite",
                    "description": "50/50 linear blend of optical RGB and SAR backscatter.",
                    "data_base64": fused_b64,
                    "statistics": {
                        "fusion_technique": "Linear Dual-Modality Visual Blend (Baseline)",
                        "scene_width": w,
                        "scene_height": h,
                        "co_registered": reg_info.get("co_registered")
                    }
                }
            ]
        }

class OpticalSARJointAnalysisProvider(BaseOpticalSARFusion):
    """
    Credible Optical + SAR Multimodal Joint Analysis Provider.
    
    Scientific Workflow:
    1. Separate normalization for Optical (surface reflectance) and SAR (microwave backscatter).
    2. Decibel (dB) scaling for SAR microwave intensity.
    3. Produces 3 distinct evidence products:
       - Optical RGB Reflectance
       - Calibrated SAR Backscatter (dB)
       - Cross-Modal Joint False Color Composite (R=SAR, G=Green/NIR, B=Blue)
    4. Generates physically grounded joint interpretation combining dielectric roughness & spectral albedo.
    """
    def __init__(self):
        super().__init__(
            model_name="OpticalSARJointAnalysisProvider",
            implementation_status="baseline"
        )

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if len(images) < 2:
            raise ValueError("Optical+SAR joint analysis requires 2 images (Optical & SAR).")

        meta = metadata or {}
        meta_opt = meta.get("optical") or meta.get("primary", {})
        meta_sar = meta.get("sar") or meta.get("secondary", {})

        img_opt, img_sar = images[0], images[1]
        img_opt, img_sar_aligned, reg_info = align_image_pair(img_opt, img_sar, meta_opt, meta_sar)

        h, w = img_opt.shape[:2]

        # Step 1: Normalize modalities separately
        opt_u8, meta_opt_proc = robust_remote_sensing_preprocess(img_opt, meta_opt, target_modality="OPTICAL")
        sar_u8, meta_sar_proc = robust_remote_sensing_preprocess(img_sar_aligned, meta_sar, target_modality="SAR")

        # Optical RGB extraction
        if opt_u8.ndim == 3 and opt_u8.shape[2] >= 3:
            opt_rgb = opt_u8[:, :, :3]
        else:
            gray_opt = opt_u8 if opt_u8.ndim == 2 else opt_u8[:, :, 0]
            opt_rgb = np.stack([gray_opt]*3, axis=-1)

        # SAR extraction
        sar_gray = sar_u8 if sar_u8.ndim == 2 else sar_u8[:, :, 0]

        # Step 2: Cross-modal false-color composite
        # Channel 0 (Red) = SAR Microwave backscatter (surface roughness & structural double bounce)
        # Channel 1 (Green) = Optical Green/NIR reflectance (vegetation vitality)
        # Channel 2 (Blue) = Optical Blue reflectance (water absorption & cloud contrast)
        joint_composite = np.zeros((h, w, 3), dtype=np.uint8)
        joint_composite[:, :, 0] = sar_gray
        joint_composite[:, :, 1] = opt_rgb[:, :, 1]
        joint_composite[:, :, 2] = opt_rgb[:, :, 2]

        # Step 3: Compute joint physical statistics
        mean_opt = float(np.mean(opt_rgb))
        mean_sar = float(np.mean(sar_gray))
        
        # Specular water candidates: Low Optical Reflectance (< 60) AND Low SAR Backscatter (< 45)
        specular_water = np.logical_and(np.mean(opt_rgb, axis=-1) < 60, sar_gray < 45)
        water_px = int(np.count_nonzero(specular_water))

        # Urban double-bounce candidates: Moderate-to-High Optical AND High SAR Backscatter (> 190)
        double_bounce_urban = np.logical_and(np.mean(opt_rgb, axis=-1) > 80, sar_gray > 190)
        urban_px = int(np.count_nonzero(double_bounce_urban))

        # Vegetated canopy: High Optical Green (> 90) AND Moderate SAR Backscatter (60 to 160)
        canopy_veg = np.logical_and(opt_rgb[:, :, 1] > 90, np.logical_and(sar_gray >= 60, sar_gray <= 160))
        veg_px = int(np.count_nonzero(canopy_veg))

        total_pixels = h * w
        water_pct = (water_px / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        urban_pct = (urban_px / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        veg_pct = (veg_px / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        # Step 4: Encode evidence artifacts
        opt_b64 = convert_array_to_base64_png(opt_rgb)
        sar_b64 = convert_array_to_base64_png(sar_gray)
        joint_b64 = convert_array_to_base64_png(joint_composite)

        polarization = meta_sar.get("polarization") or meta_sar.get("tags", {}).get("POLARIZATION", "VV/VH")
        sensor_type = meta_sar.get("sensor") or meta_sar.get("tags", {}).get("SENSOR", "Microwave SAR")

        answer = (
            f"Optical + SAR Multimodal Joint Analysis completed across {w}x{h} scene. "
            f"Cross-modal integration combines optical surface reflectance (mean albedo: {mean_opt:.1f}) "
            f"with {sensor_type} microwave backscatter ({polarization}, mean intensity: {mean_sar:.1f}). "
            f"Surface interpretations: "
            f"(1) Specular Water / Smooth Surfaces: {water_pct:.2f}% of scene ({water_px:,} px with joint low optical albedo & low radar return); "
            f"(2) Structural / Urban Double Bounce: {urban_pct:.2f}% of scene ({urban_px:,} px with high radar backscatter); "
            f"(3) Vegetated Canopy / Diffuse Scattering: {veg_pct:.2f}% of scene ({veg_px:,} px). "
            f"Spatial co-registration status: {reg_info.get('co_registered')} ({reg_info.get('registration_method')})."
        )

        # Empirical low-backscatter mask (< 45)
        sar_water_mask = np.where(sar_gray < 45, 255, 0).astype(np.uint8)
        sar_water_b64 = convert_array_to_base64_png(sar_water_mask)

        evidence = [
            {
                "id": "ev_optical_reflectance",
                "type": "original",
                "title": "Optical Scene (Surface Reflectance)",
                "description": f"Preprocessed optical raster ({w}x{h} px, CRS: {meta_opt.get('crs_display', 'CRS unavailable')}).",
                "data_base64": opt_b64,
                "statistics": {
                    "modality": "OPTICAL",
                    "mean_albedo": round(mean_opt, 1),
                    "bands": opt_rgb.shape[2]
                }
            },
            {
                "id": "ev_sar_backscatter",
                "type": "processed",
                "title": f"SAR Scene ({sensor_type} Microwave Backscatter)",
                "description": f"Decibel-scaled microwave intensity map ({polarization} polarization, surface roughness/dielectric response).",
                "data_base64": sar_b64,
                "statistics": {
                    "modality": "SAR",
                    "polarization": polarization,
                    "mean_backscatter_intensity": round(mean_sar, 1)
                }
            },
            {
                "id": "ev_joint_fused",
                "type": "fused",
                "title": "Cross-Modal False Color Composite (R=SAR, G=NIR/Green, B=Blue)",
                "description": "Multimodal fusion combining radar surface roughness (Red channel) with optical spectral bands (Green & Blue channels).",
                "data_base64": joint_b64,
                "statistics": {
                    "fusion_type": "Radiometric Cross-Modal Composite",
                    "specular_water_area": f"{water_pct:.2f}%",
                    "structural_urban_area": f"{urban_pct:.2f}%",
                    "vegetated_canopy_area": f"{veg_pct:.2f}%",
                    "co_registered": reg_info.get("co_registered"),
                    "registration_method": reg_info.get("registration_method")
                }
            },
            {
                "id": "ev_sar_specular_mask",
                "type": "mask",
                "title": "SAR Specular Water / Smooth Surface Mask",
                "description": "Empirical low-intensity radar backscatter mask (< 45 DN) indicating specular reflection away from sensor.",
                "data_base64": sar_water_b64,
                "statistics": {
                    "specular_pixels": water_px,
                    "specular_area_percent": f"{water_pct:.2f}%",
                    "polarization": polarization
                }
            }
        ]

        limitations = [
            "Radiometric calibration coefficients (sigma0/gamma0) and incidence angle look-up tables are required for absolute quantitative backscatter modeling.",
            "Sub-pixel geometric co-registration between steep terrain SAR range-Doppler projections and optical orthorectified imagery requires high-precision DEM data."
        ]

        return {
            "answer": answer,
            "model_type": "joint_analysis_baseline",
            "model_name": self.model_name,
            "model_status": "baseline",
            "implementation_status": "baseline",
            "confidence": None,
            "confidence_label": "Not available (Physically Grounded Radiometric Baseline)",
            "models": [self.model_name],
            "registration_info": reg_info,
            "evidence": evidence,
            "optical_evidence": opt_b64,
            "sar_evidence": sar_b64,
            "fused_evidence": joint_b64,
            "limitations": limitations
        }

class OpticalSARProvider(BaseModel):
    """
    Optical + SAR Multimodal Provider Orchestrator.
    Manages selection between OpticalSARJointAnalysisProvider and OpticalSARVisualizationBaseline.
    """
    def __init__(self):
        super().__init__(
            model_name="OpticalSARProvider",
            task_type="optical_sar",
            implementation_status="baseline",
            is_trained=False,
            is_remote_sensing_adapted=True
        )
        self.joint_analyzer = OpticalSARJointAnalysisProvider()
        self.vis_baseline = OpticalSARVisualizationBaseline()

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        res = self.joint_analyzer.predict(images, query, metadata)
        res["models"] = [self.model_name, self.joint_analyzer.model_name]
        return res

# Backwards compatibility alias
OpticalSARAnalyzer = OpticalSARJointAnalysisProvider
