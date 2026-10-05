from typing import Dict, Any, List

class AgentPlanner:
    """
    Formulates structured specialist execution plan based on task classification,
    modalities, image count, and model availability.
    """
    @staticmethod
    def create_plan(task: str, modalities: List[str], metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        image_count = len(metadata_list)
        
        # Determine required inputs
        if task in ("change_detection", "change_vqa"):
            required_inputs = ["temporal_pair"]
            selected_models = ["ChangeDetectionProvider", "PixelDifferenceChangeBaseline"]
            fallback_models = ["PixelDifferenceChangeBaseline"]
            evidence_required = ["ev_t1_before", "ev_t2_after", "ev_change_map", "ev_change_overlay"]
        elif task == "optical_sar":
            required_inputs = ["optical_scene", "sar_scene"]
            selected_models = ["OpticalSARJointAnalysisProvider"]
            fallback_models = ["OpticalSARVisualizationBaseline"]
            evidence_required = ["ev_optical_reflectance", "ev_sar_backscatter", "ev_joint_fused"]
        elif task == "captioning":
            required_inputs = ["single_scene"]
            selected_models = ["ClassicalSpectralCaptionerBaseline"]
            fallback_models = ["GenericVLMOrchestrator"]
            evidence_required = ["ev_spectral_profile"]
        elif task == "grounding":
            required_inputs = ["single_scene", "target_query"]
            selected_models = ["GroundingProvider", "google/owlvit-base-patch32"]
            fallback_models = ["ClassicalBaselineGrounder"]
            evidence_required = ["ev_grounding_overlay", "ev_grounding_mask"]
        else: # vqa
            required_inputs = ["single_scene", "natural_language_question"]
            selected_models = ["GenericVLMOrchestrator"]
            fallback_models = ["SpectralStatisticsBaseline"]
            evidence_required = ["ev_vqa_preprocessed"]

        preprocessing_steps = [
            "geotiff_parse",
            "nodata_nan_cleanup",
            "percentile_2_98_normalization"
        ]

        if image_count > 1:
            preprocessing_steps.append("spatial_extent_and_crs_alignment")

        if any("SAR" in m.upper() for m in modalities):
            preprocessing_steps.append("sar_decibel_db_transformation")

        return {
            "task": task,
            "required_inputs": required_inputs,
            "selected_models": selected_models,
            "evidence_required": evidence_required,
            "fallback_models": fallback_models,
            "preprocessing_steps": preprocessing_steps,
            "modalities_detected": modalities,
            "image_count": image_count,
            "max_reanalysis_attempts": 1
        }
