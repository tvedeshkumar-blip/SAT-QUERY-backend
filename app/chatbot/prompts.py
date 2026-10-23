"""
Dedicated System Prompts and Evidence Formatting for SatQuery AI Chatbot.
Establishes domain capabilities, evidence ingestion rules, and strict scientific integrity.
"""

import json
from typing import Any, Dict, List, Optional, Union

SATQUERY_SYSTEM_PROMPT = """You are SatQuery AI, a satellite imagery analysis assistant.

CAPABILITIES:
You assist Earth Observation scientists, remote sensing practitioners, and geospatial analysts. You work with:
- Satellite imagery across multi-spectral, optical, and Synthetic Aperture Radar (SAR) modalities.
- Spectral indices derived from band mathematics (e.g., NDVI for vegetation vitality, NDWI for surface water, false-color CIR).
- Microwave SAR backscatter intensity (dB scaling, polarization VV/VH, specular vs double-bounce scattering).
- Bi-temporal change detection utilizing neural architectures like BIT-CD (Bitemporal Image Transformer).
- Open-vocabulary visual grounding and object localization utilizing neural models like google/owlvit-base-patch32.
- Geospatial metadata including Coordinate Reference Systems (CRS/EPSG), spatial resolution, UTM projections, and sensor characteristics.
- Structured visual evidence artifacts including localized bounding boxes, binary change maps, and cross-modal composites.

SCIENTIFIC INTEGRITY & OBJECTIVITY RULES (MANDATORY):
1. Never invent satellite observations: Ground every analytical statement exclusively in the empirical imagery or structured evidence provided.
2. Never fabricate measurements: Do not invent change percentages, bounding-box coordinates, pixel counts, or reflectance values.
3. Never claim a model ran when it did not: If a task used a classical baseline or if model weights were offline, accurately preserve that provenance.
4. Distinguish model observations from interpretation: Explicitly separate empirical sensor detections (what the model outputs) from thematic interpretation (what the observations might physically mean).
5. Preserve model provenance: Report which specific model or algorithm produced each finding (e.g., BIT-CD vs PixelDifferenceChangeBaseline, OWL-ViT vs ClassicalBaselineGrounder).
6. Treat neural confidence scores as uncalibrated unless explicitly calibrated: Acknowledge raw model logits or detection scores as heuristic ranking indicators, not frequentist probabilities.
7. If evidence is unavailable, say so: Clearly state when data, spectral bands, or evidence layers are missing rather than guessing.
8. Do not pretend to have access to imagery that was not supplied: If the user refers to an image or scene that is absent from the session context, inform them directly.
9. Do not fabricate coordinates, CRS, sensor information, dates, or detected objects: Use only the metadata provided in the query context.
10. Scientific precision: Maintain an authoritative, concise, objective, and professional tone suitable for Earth Observation workflows.
"""

def format_evidence_context(
    evidence: Optional[Union[Dict[str, Any], List[Any]]] = None,
    workspace_context: Optional[Union[Dict[str, Any], List[Any]]] = None
) -> str:
    """
    Formats structured SatQuery specialist evidence and workspace scene metadata into
    an unambiguous grounding block appended to the system prompt.
    """
    if not evidence and not workspace_context:
        return (
            "\n[EVIDENCE STATUS: NO ACTIVE RASTER EVIDENCE SUPPLIED]\n"
            "No satellite imagery analysis, specialist model outputs, or raster evidence were provided for this message. "
            "If the user asks about specific image content, clearly state that no raster evidence is currently attached.\n"
        )

    context_parts: List[str] = []

    # 1. Active Workspace Scene Metadata (if scenes are imported into the workspace)
    if workspace_context:
        try:
            if isinstance(workspace_context, (dict, list)):
                ws_str = json.dumps(workspace_context, indent=2, default=str)
            else:
                ws_str = str(workspace_context)
        except Exception:
            ws_str = str(workspace_context)

        if not evidence:
            context_parts.append(
                "\n--- ACTIVE WORKSPACE SCENE METADATA (PIPELINE NOT YET EXECUTED) ---\n"
                "[WORKSPACE STATUS: SCENES IMPORTED, AWAITING PIPELINE EXECUTION]\n"
                "The user has acquired, validated, and imported the following satellite scene metadata into their active workspace session:\n"
                f"{ws_str}\n\n"
                "MANDATORY INSTRUCTIONS FOR IMPORTED WORKSPACE SCENES:\n"
                "1. You MAY use this metadata to answer questions about which scenes are loaded, their catalog/scene IDs, acquisition dates/timestamps, satellite platforms, sensors, modalities, CRS, spatial extents, and raster dimensions.\n"
                "2. CRITICAL CONSTRAINT: Deep perception and change analysis models (e.g., BIT-CD change detection, OWL-ViT visual grounding, spectral index masks, BigEarthNet land-cover classification) have NOT yet been executed on these rasters.\n"
                "3. You DO NOT possess direct pixel-level inspection, visual perception, or verified change detection for these scenes.\n"
                "4. NEVER fabricate, hallucinate, or guess pixel changes, change percentages, or visual scene contents.\n"
                "5. If the user asks whether features have changed between T1 and T2, asks for change percentages, asks for object detection bounding boxes, or asks what is visually visible on the ground, you MUST explicitly explain that the scenes are imported and direct the user to click 'Execute Pipeline' to run the perception models.\n"
                "--- END ACTIVE WORKSPACE SCENE METADATA ---\n"
            )
        else:
            context_parts.append(
                "\n--- ACTIVE WORKSPACE SCENE METADATA ---\n"
                "The following satellite scene metadata is associated with this analysis session:\n"
                f"{ws_str}\n"
                "--- END WORKSPACE SCENE METADATA ---\n"
            )

    # 2. Verified Specialist Perception Evidence (if perception pipeline has run)
    if evidence:
        try:
            if isinstance(evidence, (dict, list)):
                evidence_str = json.dumps(evidence, indent=2, default=str)
            else:
                evidence_str = str(evidence)
        except Exception:
            evidence_str = str(evidence)

        context_parts.append(
            "\n--- VERIFIED SATQUERY SPECIALIST EVIDENCE ---\n"
            "[EVIDENCE INJECTION]: The following data represents verified empirical telemetry "
            "and inference artifacts produced by SatQuery specialist perception models (e.g. BIT-CD, OWL-ViT, spectral indices).\n"
            "RULES FOR EVIDENCE USE:\n"
            "1. This is ground-truth evidence supplied by the SatQuery perception engine.\n"
            "2. You must strictly base all observational statements and metrics on this evidence.\n"
            "3. DO NOT extrapolate, fabricate additional detections, or hallucinate metrics beyond what is stated below.\n"
            "4. Quote specific metrics (e.g., change area %, detection counts, box coordinates, CRS) faithfully.\n\n"
            f"{evidence_str}\n"
            "--- END VERIFIED SATQUERY EVIDENCE ---\n"
        )

    return "".join(context_parts)


def assemble_qwen_prompt(
    message: str,
    history: Optional[List[Any]] = None,
    evidence: Optional[Union[Dict[str, Any], List[Any]]] = None,
    workspace_context: Optional[Union[Dict[str, Any], List[Any]]] = None
) -> str:
    """
    Assembles a complete, grounded, single-turn prompt for Qwen/Qwen2.5-VL-3B-Instruct.
    Incorporates:
    - Domain system prompt and scientific integrity rules
    - Active workspace scene metadata (if provided)
    - Verified specialist perception evidence (if provided)
    - Multi-turn conversation history (if provided)
    - The user's current query
    """
    cleaned_query = (message or "").strip()

    prompt_sections: List[str] = []

    # 1. System domain rules and capabilities
    prompt_sections.append(SATQUERY_SYSTEM_PROMPT.strip())

    # 2. Evidence and workspace context
    context_str = format_evidence_context(
        evidence=evidence,
        workspace_context=workspace_context
    ).strip()
    if context_str:
        prompt_sections.append(context_str)

    # 3. Conversation history
    if history:
        history_lines = ["\n--- RECENT CONVERSATION HISTORY ---"]
        for turn in history:
            role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else None) or "user"
            content = getattr(turn, "content", None) or (turn.get("content") if isinstance(turn, dict) else None) or ""
            role_label = "User" if str(role).lower() == "user" else ("Assistant" if str(role).lower() == "assistant" else "System")
            history_lines.append(f"{role_label}: {content}")
        history_lines.append("--- END CONVERSATION HISTORY ---")
        prompt_sections.append("\n".join(history_lines))

    # 4. Current user query
    prompt_sections.append(f"[CURRENT USER QUERY]:\n{cleaned_query}\n\n[ASSISTANT RESPONSE]:")

    return "\n\n".join(prompt_sections)

