import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("satquery.router")

class RoutingDecision(BaseModel):
    task: str
    reason: str
    required_modalities: List[str]
    required_image_count: int
    selected_models: List[str]
    classifier_type: str = "rule_based"

class IntentClassifier:
    """
    Modular Intent Classifier supporting rule-based heuristics,
    optional LLM classification, and optional ML classification.
    """

    @classmethod
    def rule_based_classifier(
        cls, 
        query: str, 
        image_count: int, 
        modalities: List[str], 
        mode_hint: str = "auto"
    ) -> RoutingDecision:
        q = (query or "").lower().strip()

        # Explicit user UI mode override
        if mode_hint == "optical_sar" and image_count >= 2:
            return RoutingDecision(
                task="optical_sar",
                reason="User explicitly selected Optical+SAR dual modality mode.",
                required_modalities=["OPTICAL", "SAR"],
                required_image_count=2,
                selected_models=["OpticalSARVisualizationBaseline"],
                classifier_type="explicit_user_override"
            )

        if mode_hint == "bitemporal" and image_count >= 2:
            is_detection = any(w in q for w in ["map", "detect", "difference", "mask"])
            task = "change_detection" if is_detection else "change_vqa"
            model = "PixelDifferenceChangeBaseline" if is_detection else "EvidenceGroundedChangeVQA"
            return RoutingDecision(
                task=task,
                reason="User selected Bi-Temporal mode; routed based on query intent.",
                required_modalities=["OPTICAL"],
                required_image_count=2,
                selected_models=[model],
                classifier_type="explicit_user_override"
            )

        # Multi-image auto routing
        if image_count >= 2:
            has_sar = "SAR" in [m.upper() for m in modalities] or "sar" in q or "microwave" in q
            if has_sar or any(w in q for w in ["fusion", "optical and sar", "multimodal"]):
                return RoutingDecision(
                    task="optical_sar",
                    reason="Two input scenes detected; one or more identified with SAR microwave modality.",
                    required_modalities=["OPTICAL", "SAR"],
                    required_image_count=2,
                    selected_models=["OpticalSARVisualizationBaseline"],
                    classifier_type="rule_based"
                )
            if any(w in q for w in ["changed", "change", "increase", "decrease", "temporal", "expansion", "evolution"]):
                return RoutingDecision(
                    task="change_vqa",
                    reason="Bi-temporal pair provided with natural language change query.",
                    required_modalities=["OPTICAL"],
                    required_image_count=2,
                    selected_models=["EvidenceGroundedChangeVQA", "PixelDifferenceChangeBaseline"],
                    classifier_type="rule_based"
                )
            return RoutingDecision(
                task="change_detection",
                reason="Two scenes provided; default to spectral change detection mapping.",
                required_modalities=["OPTICAL"],
                required_image_count=2,
                selected_models=["PixelDifferenceChangeBaseline"],
                classifier_type="rule_based"
            )

        # Single image tasks
        if any(w in q for w in ["highlight", "detect", "ground", "find", "locate", "outline", "mask", "box", "segment", "where is", "where are", "show me", "show", "pinpoint", "bounding box"]):
            return RoutingDecision(
                task="grounding",
                reason="Single scene provided with spatial feature localization/grounding prompt.",
                required_modalities=["OPTICAL"],
                required_image_count=1,
                selected_models=["GroundingProvider", "google/owlvit-base-patch32", "ClassicalBaselineGrounder"],
                classifier_type="rule_based"
            )
        if any(w in q for w in ["what", "which", "is there", "are there", "how many", "?"]):
            return RoutingDecision(
                task="vqa",
                reason="Single scene provided with interrogative visual question prompt.",
                required_modalities=["OPTICAL"],
                required_image_count=1,
                selected_models=["GenericVLMOrchestrator"],
                classifier_type="rule_based"
            )
        if any(w in q for w in ["describe", "caption", "summary", "overview", "report"]):
            return RoutingDecision(
                task="captioning",
                reason="Single scene provided with scene description/captioning prompt.",
                required_modalities=["OPTICAL"],
                required_image_count=1,
                selected_models=["ClassicalSpectralCaptionerBaseline"],
                classifier_type="rule_based"
            )

        # Default single image task: VQA
        return RoutingDecision(
            task="vqa",
            reason="Single scene provided; routed to remote sensing visual question answering.",
            required_modalities=["OPTICAL"],
            required_image_count=1,
            selected_models=["GenericVLMOrchestrator"],
            classifier_type="rule_based"
        )

    @classmethod
    def optional_llm_classifier(cls, query: str, modalities: List[str]) -> Optional[RoutingDecision]:
        """Optional LLM-based intent classifier hook when external LLM is configured."""
        return None

    @classmethod
    def optional_ml_classifier(cls, query: str) -> Optional[RoutingDecision]:
        """Optional lightweight ML (TF-IDF / scikit-learn) intent classifier hook."""
        return None

    @classmethod
    def classify(
        cls, 
        query: str, 
        image_count: int, 
        modalities: List[str], 
        mode_hint: str = "auto"
    ) -> RoutingDecision:
        # Check ML classifier first if configured
        ml_decision = cls.optional_ml_classifier(query)
        if ml_decision:
            return ml_decision

        # Default to robust rule-based classifier
        return cls.rule_based_classifier(query, image_count, modalities, mode_hint)


class TaskRouter:
    """
    Backwards-compatible wrapper around IntentClassifier.
    """
    @classmethod
    def route(
        cls, 
        query: str, 
        image_count: int, 
        modalities: List[str], 
        mode_hint: str = "auto"
    ) -> RoutingDecision:
        return IntentClassifier.classify(query, image_count, modalities, mode_hint)

    @classmethod
    def classify_task(
        cls, 
        query: str, 
        image_count: int, 
        modalities: List[str], 
        mode_hint: str = "auto"
    ) -> str:
        decision = cls.route(query, image_count, modalities, mode_hint)
        return decision.task
