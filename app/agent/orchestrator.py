"""
Stage 14: Dynamic Multi-Step Agent Orchestrator for Multimodal Remote Sensing.

Implements a deterministic, controlled execution engine that:
1. Parses natural-language queries for spatial areas (gazetteer & bounding boxes),
   dates, date ranges, and multimodal analysis intents.
2. Formulates explicit clarification requests when required location or time ranges
   are ambiguous, preventing hallucinated coordinates or dates.
3. Plans and executes a bounded, observable multi-step tool sequence (DAG)
   using discrete tools from the AgentToolRegistry.
4. Enforces strict safety caps:
   - Maximum steps: <= 10
   - Maximum retries: <= 2
   - Cumulative transfer budget: <= 20 MB
5. Executes Sentinel-2 NDVI, Sentinel-1 RTC DpRVI, and temporal multimodal change detection.
6. Packages evidence into EvidencePackageSchema and executes DeterministicVerifier (Checks A-L).
7. Generates reproducible scientific reports and maintains complete execution traces.
"""

import re
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from app.agent.tools import tool_registry
from app.agent.trace import ExecutionTraceTracker
from app.remote_sensing.temporal import compute_temporal_optical_change, compute_temporal_sar_change
from app.agent.verifier import DeterministicVerifier, EvidencePackageBuilder
from app.agent.conflict_detector import ConflictDetector, ConflictCheckResult
from app.schemas.analysis import ConflictInfoSchema
from app.schemas.verification import EvidencePackageSchema, ClaimVerificationItem
from app.reports.scientific_report import ScientificReportBuilder
from app.acquisition.windowed_reader import SafeWindowedCOGReader

logger = logging.getLogger("satquery.orchestrator")

# Safety Caps
MAX_ORCHESTRATION_STEPS = 10
MAX_TOOL_RETRIES = 2
MAX_CUMULATIVE_TRANSFER_BYTES = 20 * 1024 * 1024  # 20 MB

# Gazetteer of demonstration geographic regions
GAZETTEER: Dict[str, List[float]] = {
    "mumbai": [72.75, 18.88, 73.05, 19.30],
    "delhi": [76.85, 28.40, 77.35, 28.88],
    "bengaluru": [77.45, 12.85, 77.75, 13.15],
    "bangalore": [77.45, 12.85, 77.75, 13.15],
    "chennai": [80.15, 12.90, 80.35, 13.25],
    "hyderabad": [78.30, 17.25, 78.60, 17.55],
    "kolkata": [88.25, 22.45, 88.45, 22.65],
    "kerala": [75.80, 9.50, 76.80, 10.50],
    "pune": [73.75, 18.45, 73.98, 18.62],
    "ahmedabad": [72.50, 22.95, 72.70, 23.15],
    "california": [-122.55, 37.60, -122.25, 37.90],
    "san francisco": [-122.55, 37.60, -122.25, 37.90],
    "amazon": [-63.20, -10.20, -62.80, -9.80],
    "rondonia": [-63.20, -10.20, -62.80, -9.80]
}

MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12"
}


class QueryParser:
    """Natural-language remote sensing query parser with ambiguity detection."""

    @staticmethod
    def parse(
        query: str, 
        has_images: bool = False,
        aoi_bbox: Optional[List[float]] = None,
        start_date_in: Optional[str] = None,
        end_date_in: Optional[str] = None,
        t1_start_date_in: Optional[str] = None,
        t1_end_date_in: Optional[str] = None,
        t2_start_date_in: Optional[str] = None,
        t2_end_date_in: Optional[str] = None,
        max_cloud_cover_in: Optional[float] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        q_lower = query.lower()

        # 1. Spatial Resolution
        bbox = None
        location_name = None

        # Check explicitly provided AOI bounding box
        if aoi_bbox and len(aoi_bbox) == 4:
            try:
                coords = [float(c) for c in aoi_bbox]
                if coords[0] < coords[2] and coords[1] < coords[3] and -180 <= coords[0] <= 180 and -90 <= coords[1] <= 90:
                    bbox = coords
                    location_name = "User-Selected AOI"
            except (ValueError, TypeError):
                pass

        # Check explicit bounding box coordinates in query string e.g. [72.8, 18.9, 72.9, 19.0]
        if not bbox:
            coord_match = re.search(r'\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]', query)
            if coord_match:
                try:
                    coords = [float(c) for c in coord_match.groups()]
                    # Validate min < max
                    if coords[0] < coords[2] and coords[1] < coords[3] and -180 <= coords[0] <= 180 and -90 <= coords[1] <= 90:
                        bbox = coords
                        location_name = "User-Specified Bounding Box"
                except ValueError:
                    pass

        # Check gazetteer names to resolve location_name (and bbox if not already explicitly specified)
        for name, box in GAZETTEER.items():
            if name in q_lower:
                if not bbox:
                    bbox = box
                if not location_name:
                    location_name = name.title()
                break

        # 2. Temporal Resolution
        t1_start = t1_start_date_in
        t1_end = t1_end_date_in
        t2_start = t2_start_date_in
        t2_end = t2_end_date_in

        # Check bi-temporal date window ordering if provided
        is_invalid_date_order = False
        date_order_error = None
        if t1_start and t1_end and t2_start and t2_end:
            try:
                d1s = datetime.fromisoformat(t1_start.replace("Z", "+00:00").split("T")[0])
                d1e = datetime.fromisoformat(t1_end.replace("Z", "+00:00").split("T")[0])
                d2s = datetime.fromisoformat(t2_start.replace("Z", "+00:00").split("T")[0])
                d2e = datetime.fromisoformat(t2_end.replace("Z", "+00:00").split("T")[0])
                if d1s > d1e:
                    is_invalid_date_order = True
                    date_order_error = f"Pre-event start date ({t1_start}) must precede or equal pre-event end date ({t1_end})."
                elif d2s > d2e:
                    is_invalid_date_order = True
                    date_order_error = f"Post-event start date ({t2_start}) must precede or equal post-event end date ({t2_end})."
                elif d1e >= d2s:
                    is_invalid_date_order = True
                    date_order_error = f"Post-event window start ({t2_start}) must be strictly later than pre-event window end ({t1_end})."
            except Exception as e:
                is_invalid_date_order = True
                date_order_error = f"Invalid date format in bi-temporal windows: {e}"

        start_date = start_date_in or t1_start
        end_date = end_date_in or t2_end
        t1_date = t1_start or start_date
        t2_date = t2_end or end_date

        # Check ISO dates: YYYY-MM-DD
        if not start_date or not end_date:
            iso_dates = re.findall(r'\b(\d{4}-\d{2}-\d{2})\b', query)
            if len(iso_dates) >= 2:
                s_d, e_d = sorted(iso_dates[:2])
                if not start_date:
                    start_date = s_d
                if not end_date:
                    end_date = e_d
                t1_date, t2_date = start_date, end_date
            elif len(iso_dates) == 1 and not start_date:
                start_date = iso_dates[0]

        # Check Month-Year ranges e.g. "January 2024 to March 2024"
        if not start_date:
            m_range = re.findall(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{4})\b', q_lower)
            if len(m_range) >= 2:
                m1_name, y1 = m_range[0]
                m2_name, y2 = m_range[1]
                m1_num = MONTH_MAP.get(m1_name, "01")
                m2_num = MONTH_MAP.get(m2_name, "12")
                start_date = f"{y1}-{m1_num}-01"
                end_date = f"{y2}-{m2_num}-28"
                t1_date, t2_date = start_date, end_date

        # Check Year comparisons e.g. "2023 to 2024" or "between 2023 and 2024"
        if not start_date:
            y_range = re.findall(r'\b(?:between|from|in)?\s*(\d{4})\s*(?:and|to|vs|versus|-)\s*(\d{4})\b', q_lower)
            if y_range:
                y1, y2 = y_range[0]
                start_date = f"{y1}-01-01"
                end_date = f"{y2}-12-31"
                t1_date, t2_date = f"{y1}-06-01", f"{y2}-06-01"

        # Cloud Cover Threshold parsing
        cloud_thresh = 20.0
        if max_cloud_cover_in is not None:
            try:
                cloud_thresh = float(max_cloud_cover_in)
                cloud_thresh = max(0.0, min(100.0, cloud_thresh))
            except (ValueError, TypeError):
                cloud_thresh = 20.0
        else:
            c_match = re.search(r'(?:cloud(?:s|\s+cover)?\s*(?:<|<=|under|below|less\s+than)\s*(\d+(?:\.\d+)?)\s*%?|(\d+(?:\.\d+)?)\s*%\s*(?:max(?:imum)?\s+)?cloud)', q_lower)
            if c_match:
                val = c_match.group(1) or c_match.group(2)
                try:
                    cloud_thresh = float(val)
                except ValueError:
                    pass

        # 3. Intent Detection
        is_temporal_change = any(t in q_lower for t in ["change", "temporal", "difference", "trend", "loss", "gain", "increase", "decrease", "compare", "vs", "versus"])
        is_sar = any(t in q_lower for t in ["sar", "radar", "backscatter", "dprvi", "sentinel-1", "s1", "rtc", "vv", "vh", "polarization"])
        is_optical = any(t in q_lower for t in ["optical", "ndvi", "vegetation", "canopy", "sentinel-2", "s2", "red", "nir", "b04", "b08", "greenery"])
        is_fusion = (is_sar and is_optical) or "fusion" in q_lower or "joint" in q_lower or "cross-sensor" in q_lower
        is_classification = any(t in q_lower for t in ["classify", "classification", "land cover", "land-cover", "corine", "bigearthnet", "classes", "lulc"])
        is_grounding = any(t in q_lower for t in ["ground", "locate", "detect", "bounding box", "find", "where is", "where are"])
        is_vqa = any(t in q_lower for t in ["what", "how many", "is there", "describe", "explain", "caption"])

        # Override or specialize task using caller-specified mode if provided
        if mode == "optical_sar" or is_fusion:
            is_fusion = True
            task = "optical_sar"
        elif mode in ("classification", "land_cover_classification") or (is_classification and not is_temporal_change and not is_fusion):
            task = "land_cover_classification"
        elif mode in ("bitemporal", "temporal") or is_temporal_change:
            is_temporal_change = True
            task = "temporal_sar_change" if is_sar else "temporal_optical_change"
        elif mode == "single":
            task = "sentinel1_rtc_dprvi" if is_sar else "sentinel2_ndvi"
        elif is_sar:
            task = "sentinel1_rtc_dprvi"
        elif is_optical:
            task = "sentinel2_ndvi"
        elif is_grounding:
            task = "grounding"
        else:
            task = "vqa"

        # 4. Ambiguity Evaluation
        # If images are already provided in the workspace, spatial location and dates are implicitly present
        missing_params = []
        is_offline_demo = mode == "offline" or "offline" in q_lower or "demo" in q_lower
        if not has_images and not is_offline_demo:
            if not bbox:
                missing_params.append("spatial_location")
            if is_temporal_change and not (start_date and end_date) and not (t1_start and t2_end):
                missing_params.append("temporal_date_range")

        is_ambiguous = len(missing_params) > 0

        clarification_prompt = None
        if is_ambiguous:
            parts = []
            if "spatial_location" in missing_params:
                parts.append("the geographic area of interest (e.g. 'Mumbai', 'Delhi', or a bounding box [min_lon, min_lat, max_lon, max_lat])")
            if "temporal_date_range" in missing_params:
                parts.append("the temporal comparison dates or date range (e.g. 'between 2023-01-01 and 2024-01-01')")
            clarification_prompt = (
                f"To execute calibrated remote-sensing analysis without fabricating data, please clarify {' and '.join(parts)}."
            )

        return {
            "task": task,
            "bbox": bbox,
            "location_name": location_name,
            "start_date": start_date,
            "end_date": end_date,
            "t1_date": t1_date,
            "t2_date": t2_date,
            "t1_start_date": t1_start,
            "t1_end_date": t1_end,
            "t2_start_date": t2_start,
            "t2_end_date": t2_end,
            "max_cloud_cover": cloud_thresh,
            "is_invalid_date_order": is_invalid_date_order,
            "date_order_error": date_order_error,
            "is_temporal_change": is_temporal_change,
            "is_sar": is_sar,
            "is_optical": is_optical,
            "is_fusion": is_fusion,
            "is_classification": is_classification,
            "is_ambiguous": is_ambiguous,
            "missing_parameters": missing_params,
            "clarification_prompt": clarification_prompt,
            "raw_query": query
        }


class MultiStepOrchestrator:
    """Executes a controlled, observable multi-step remote sensing workflow."""

    def __init__(self):
        self.tool_registry = tool_registry

    @staticmethod
    def _resolve_temporal_search_windows(parsed: Dict[str, Any]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Resolves distinct, non-overlapping search windows for T1 (pre-event) and T2 (post-event)."""
        if parsed.get("t1_start_date") and parsed.get("t1_end_date") and parsed.get("t2_start_date") and parsed.get("t2_end_date"):
            return (parsed["t1_start_date"], parsed["t1_end_date"]), (parsed["t2_start_date"], parsed["t2_end_date"])

        s_date = parsed.get("start_date") or "2023-01-01"
        e_date = parsed.get("end_date") or "2024-01-01"
        t1_date = parsed.get("t1_date")
        t2_date = parsed.get("t2_date")

        try:
            d_start = datetime.fromisoformat(s_date.replace("Z", "+00:00").split("T")[0])
            d_end = datetime.fromisoformat(e_date.replace("Z", "+00:00").split("T")[0])
            if d_end <= d_start:
                d_end = datetime(d_start.year + 1, d_start.month, d_start.day)
        except Exception:
            return ("2023-01-01", "2023-06-30"), ("2023-07-01", "2024-01-01")

        if t1_date and t2_date and t1_date != t2_date:
            try:
                d_t1 = datetime.fromisoformat(t1_date.replace("Z", "+00:00").split("T")[0])
                d_t2 = datetime.fromisoformat(t2_date.replace("Z", "+00:00").split("T")[0])
                if d_t1 < d_t2:
                    w1_start = max(d_start, d_t1 - timedelta(days=30)).strftime("%Y-%m-%d")
                    w1_end = min(d_t2 - timedelta(days=1), d_t1 + timedelta(days=30)).strftime("%Y-%m-%d")
                    w2_start = max(d_t1 + timedelta(days=1), d_t2 - timedelta(days=30)).strftime("%Y-%m-%d")
                    w2_end = min(d_end, d_t2 + timedelta(days=30)).strftime("%Y-%m-%d")
                    return (w1_start, w1_end), (w2_start, w2_end)
            except Exception:
                pass

        mid_ts = d_start.timestamp() + (d_end.timestamp() - d_start.timestamp()) / 2.0
        d_mid = datetime.fromtimestamp(mid_ts)
        w1_start = d_start.strftime("%Y-%m-%d")
        w1_end = d_mid.strftime("%Y-%m-%d")
        w2_start = (d_mid + timedelta(days=1)).strftime("%Y-%m-%d")
        w2_end = d_end.strftime("%Y-%m-%d")
        return (w1_start, w1_end), (w2_start, w2_end)

    def _synthesize_claims(
        self,
        task: str,
        analysis_metadata: Dict[str, Any],
        execution_steps: Optional[List[Dict[str, Any]]] = None,
        max_bytes: int = MAX_CUMULATIVE_TRANSFER_BYTES,
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[float], str, str, List[ClaimVerificationItem]]:
        """Synthesize answer text, confidence metadata, and candidate verification claims."""
        candidate_claims: List[ClaimVerificationItem] = []
        execution_steps = execution_steps or []
        is_budget_exhausted = analysis_metadata.get("budget_exhausted", False)
        overall_status = "success"
        confidence: Optional[float] = 0.95
        confidence_label = "Deterministic Verified"

        if is_budget_exhausted:
            overall_status = "budget_exhausted"
            answer = f"Analysis halted: Cumulative STAC transfer budget of {max_bytes} bytes was exhausted before all required satellite assets could be retrieved."
            confidence = None
            confidence_label = "Budget Exhausted"

        elif task == "temporal_optical_change":
            chg = analysis_metadata.get("temporal_optical_change", {})
            chg_status = chg.get("status")
            if chg_status in ("blocked", "failed", "error") or not chg:
                overall_status = "failed"
                err_detail = chg.get("error", "Upstream Sentinel-2 optical scenes could not be retrieved.")
                answer = f"Temporal Sentinel-2 optical change analysis failed: {err_detail}"
                confidence = None
                confidence_label = "Acquisition Failed"
            elif chg_status == "unsupported_zero_valid_pixels":
                overall_status = "unsupported"
                answer = chg.get("summary")
                confidence = None
                confidence_label = "Zero Valid Pixels"
            else:
                d_mean = chg.get("delta_ndvi_mean", 0.0)
                v_loss = chg.get("vegetation_loss_pct", 0.0)
                v_gain = chg.get("vegetation_gain_pct", 0.0)
                px_count = chg.get("joint_valid_pixel_count", 0)
                answer = (
                    f"Temporal Sentinel-2 optical change analysis computed across {px_count:,} joint clear pixels. "
                    f"Mean Delta-NDVI is {d_mean:+.4f}. Observed vegetation loss: {v_loss}%, vegetation gain: {v_gain}%."
                )
                candidate_claims.append(ClaimVerificationItem(
                    claim_text=f"Mean Delta-NDVI is {d_mean:.4f} across joint clear pixels.",
                    claim_type="measurement",
                    status="supported"
                ))

        elif task == "temporal_sar_change":
            chg = analysis_metadata.get("temporal_sar_change", {})
            chg_status = chg.get("status")
            if chg_status in ("blocked", "failed", "error") or not chg:
                overall_status = "failed"
                err_detail = chg.get("error", "Upstream Sentinel-1 RTC SAR scenes could not be retrieved.")
                answer = f"Temporal Sentinel-1 RTC SAR change analysis failed: {err_detail}"
                confidence = None
                confidence_label = "Acquisition Failed"
            elif chg_status == "unsupported_zero_valid_pixels":
                overall_status = "unsupported"
                answer = chg.get("summary")
                confidence = None
                confidence_label = "Zero Valid Pixels"
            else:
                d_mean = chg.get("delta_dprvi_mean", 0.0)
                c_loss = chg.get("radar_canopy_loss_pct", 0.0)
                c_gain = chg.get("radar_canopy_gain_pct", 0.0)
                px_count = chg.get("joint_valid_pixel_count", 0)
                answer = (
                    f"Temporal Sentinel-1 RTC dual-polarization change analysis computed across {px_count:,} joint valid pixels. "
                    f"Mean Delta-DpRVI is {d_mean:+.4f}. Radar canopy loss: {c_loss}%, radar canopy gain: {c_gain}%."
                )
                candidate_claims.append(ClaimVerificationItem(
                    claim_text=f"Mean Delta-DpRVI is {d_mean:.4f} across valid radar pixels.",
                    claim_type="measurement",
                    status="supported"
                ))

        elif task == "optical_sar":
            fusion = analysis_metadata.get("optical_sar_fusion", {})
            f_status = fusion.get("status")
            if f_status in ("blocked", "failed", "error") or not fusion:
                overall_status = "failed"
                err_detail = fusion.get("error", "Upstream optical or SAR scene could not be retrieved.")
                answer = f"Optical-SAR cross-sensor fusion analysis failed: {err_detail}"
                confidence = None
                confidence_label = "Acquisition Failed"
            elif f_status == "unsupported" or (isinstance(f_status, str) and f_status.startswith("unsupported")):
                overall_status = "unsupported"
                answer = fusion.get("answer") or "Optical-SAR cross-sensor fusion analysis unsupported."
                confidence = None
                ana_status = fusion.get("analysis_status") or f_status
                confidence_label = f"Unsupported ({ana_status.replace('_', ' ').title()})"
            else:
                ans_text = fusion.get("answer") or fusion.get("summary") or "Optical and SAR multimodal cross-sensor analysis completed successfully."
                answer = ans_text
                candidate_claims.append(ClaimVerificationItem(
                    claim_text="Multimodal optical and SAR cross-sensor alignment and fusion analysis completed successfully.",
                    claim_type="general",
                    status="supported"
                ))

        elif task == "sentinel1_rtc_dprvi":
            dprvi = analysis_metadata.get("dprvi_result", {})
            d_val = dprvi.get("dprvi_mean")
            q_db = dprvi.get("cross_pol_ratio_db")
            if d_val is not None:
                answer = (
                    f"Calibrated Sentinel-1 RTC dual-polarization backscatter computed across {dprvi.get('valid_pixel_count', 0):,} valid pixels. "
                    f"Mean DpRVI is {d_val:.4f} (cross-polarization ratio: {q_db:.1f} dB)."
                )
                candidate_claims.append(ClaimVerificationItem(
                    claim_text=f"Sentinel-1 DpRVI is {d_val:.4f}.",
                    claim_type="measurement",
                    status="supported"
                ))
            else:
                d_status = dprvi.get("status") if dprvi else None
                if not d_status:
                    for step in reversed(execution_steps):
                        s_status = step.get("status")
                        s_err = (step.get("error") or "").lower()
                        if s_status == "unsupported" or (s_err and "unsupported" in s_err and "acquisition" not in s_err):
                            d_status = "unsupported"
                            break
                if d_status == "unsupported_zero_valid_pixels":
                    overall_status = "unsupported"
                    answer = dprvi.get("summary") or "No valid radar pixels found. DpRVI calculation unsupported."
                    confidence = None
                    confidence_label = "Zero Valid Pixels"
                elif d_status and d_status.startswith("unsupported"):
                    overall_status = "unsupported"
                    answer = dprvi.get("summary") or dprvi.get("error") or "Sentinel-1 RTC dual-polarization analysis unsupported."
                    confidence = None
                    confidence_label = "Unsupported Analysis"
                else:
                    overall_status = "failed"
                    confidence = None
                    confidence_label = "Acquisition Failed"
                    err_detail = dprvi.get("error")
                    if not err_detail and execution_steps:
                        for s in reversed(execution_steps):
                            if s.get("status") in ("failed", "blocked") and s.get("error"):
                                err_detail = s["error"]
                                break
                    err_msg = err_detail or "Radar imagery unavailable or STAC retrieval failed."
                    answer = f"Sentinel-1 RTC dual-polarization analysis failed: {err_msg}"

        elif task == "sentinel2_ndvi":
            ndvi = analysis_metadata.get("ndvi_result", {})
            n_val = ndvi.get("ndvi_mean")
            if n_val is not None:
                answer = f"Calibrated Sentinel-2 surface reflectance NDVI is {n_val:.4f} across valid SCL-screened pixels."
                candidate_claims.append(ClaimVerificationItem(
                    claim_text=f"Mean Sentinel-2 NDVI is {n_val:.4f}.",
                    claim_type="measurement",
                    status="supported"
                ))
            else:
                n_status = ndvi.get("status") if ndvi else None
                if not n_status:
                    for step in reversed(execution_steps):
                        s_status = step.get("status")
                        s_err = (step.get("error") or "").lower()
                        if s_status == "unsupported" or (s_err and "unsupported" in s_err and "acquisition" not in s_err):
                            n_status = "unsupported"
                            break
                if n_status == "unsupported_zero_valid_pixels":
                    overall_status = "unsupported"
                    answer = ndvi.get("summary") or "No valid clear-sky pixels found after applying SCL quality mask. NDVI calculation unsupported."
                    confidence = None
                    confidence_label = "Zero Valid Pixels"
                elif n_status and n_status.startswith("unsupported"):
                    overall_status = "unsupported"
                    answer = ndvi.get("summary") or ndvi.get("error") or "Sentinel-2 NDVI analysis unsupported."
                    confidence = None
                    confidence_label = "Unsupported Analysis"
                else:
                    overall_status = "failed"
                    confidence = None
                    confidence_label = "Acquisition Failed"
                    err_detail = ndvi.get("error")
                    if not err_detail and execution_steps:
                        for s in reversed(execution_steps):
                            if s.get("status") in ("failed", "blocked") and s.get("error"):
                                err_detail = s["error"]
                                break
                    err_msg = err_detail or "Optical imagery unavailable or STAC retrieval failed."
                    answer = f"Sentinel-2 NDVI analysis failed: {err_msg}"

        elif task == "land_cover_classification":
            clf = analysis_metadata.get("land_cover_classification", {})
            if clf.get("status") == "success" or "top_predictions" in clf:
                preds = clf.get("top_predictions", [])
                top_str = ", ".join([f"{p['label']} ({p['probability']:.1%})" for p in preds[:3]])
                answer = f"BigEarthNet ResNet-50 land-cover classification identified: {top_str}."
                for p in preds[:3]:
                    candidate_claims.append(ClaimVerificationItem(
                        claim_text=f"Detected land cover class '{p['label']}' with probability {p['probability']:.2f}.",
                        claim_type="classification",
                        status="supported"
                    ))
            else:
                err_detail = clf.get("error", "Classification failed")
                answer = f"Land-cover classification failed: {err_detail}"
                overall_status = "failed"
                confidence = None
                confidence_label = "Classification Failed"
        else:
            answer = (analysis_result or {}).get("answer", f"Multi-step analysis for '{task}' completed successfully.")

        return overall_status, confidence, confidence_label, answer, candidate_claims

    def orchestrate(
        self,
        query: str,
        images: Optional[List[Any]] = None,
        max_bytes: int = MAX_CUMULATIVE_TRANSFER_BYTES,
        aoi_bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        t1_start_date: Optional[str] = None,
        t1_end_date: Optional[str] = None,
        t2_start_date: Optional[str] = None,
        t2_end_date: Optional[str] = None,
        max_cloud_cover: Optional[float] = None,
        mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        trace_id = f"orch_{uuid.uuid4().hex[:10]}"
        images = images or []
        has_images = len(images) > 0

        meta = metadata or {}
        use_offline = (
            meta.get("use_offline_fixtures") is True
            or meta.get("offline_mode") is True
            or meta.get("demo_fixtures") is True
            or mode == "offline"
            or "local_sample" in query
            or "offline" in query.lower()
            or "demo" in query.lower()
            or (not (aoi_bbox or meta.get("aoi_bbox")) and not has_images)
        )

        # Step 1: Natural-Language Query Parsing
        trace = ExecutionTraceTracker(task="orchestration")
        trace.add_step("QUERY_RECEIVED", f"Received query: '{query}' ({len(images)} workspace image(s) attached)")

        parsed = QueryParser.parse(
            query=query, 
            has_images=has_images,
            aoi_bbox=aoi_bbox or meta.get("aoi_bbox"),
            start_date_in=start_date or meta.get("start_date"),
            end_date_in=end_date or meta.get("end_date"),
            t1_start_date_in=t1_start_date or meta.get("t1_start_date"),
            t1_end_date_in=t1_end_date or meta.get("t1_end_date"),
            t2_start_date_in=t2_start_date or meta.get("t2_start_date"),
            t2_end_date_in=t2_end_date or meta.get("t2_end_date"),
            max_cloud_cover_in=max_cloud_cover if max_cloud_cover is not None else meta.get("max_cloud_cover"),
            mode=mode
        )
        parsed["use_offline_fixtures"] = use_offline
        task = parsed["task"]
        trace.task = task

        # Check Bi-Temporal Chronological Order
        if parsed.get("is_invalid_date_order"):
            trace.add_step("DATE_ORDER_VALIDATION_FAILED", parsed["date_order_error"], status="failed")
            elapsed = round((time.time() - start_time) * 1000, 2)
            return {
                "id": trace_id,
                "status": "failed",
                "task": task,
                "mode": mode or "orchestration",
                "answer": f"Bi-temporal date validation failed: {parsed['date_order_error']}",
                "confidence": None,
                "confidence_label": "Invalid Date Windows",
                "models": [],
                "evidence": [],
                "trace": trace.to_dict(),
                "metadata": {
                    "error": parsed["date_order_error"],
                    "parsed_query": parsed,
                    "date_validation": {
                        "is_valid": False,
                        "error": parsed["date_order_error"]
                    }
                },
                "execution_time_ms": elapsed,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }

        # Check Ambiguity Gate: If required inputs cannot be determined, ask for clarification
        if parsed["is_ambiguous"]:
            trace.add_step("CLARIFICATION_REQUIRED", parsed["clarification_prompt"], status="pending")
            elapsed = round((time.time() - start_time) * 1000, 2)
            return {
                "id": trace_id,
                "status": "clarification_required",
                "task": task,
                "mode": "orchestration",
                "answer": parsed["clarification_prompt"],
                "is_clarification_required": True,
                "missing_parameters": parsed["missing_parameters"],
                "suggested_examples": [
                    "Analyze Sentinel-2 NDVI change in Mumbai between 2023-01-01 and 2024-01-01",
                    "Compute Sentinel-1 RTC dual-pol DpRVI for Delhi [76.85, 28.40, 77.35, 28.88]",
                    "Assess radar canopy change in California between 2023 and 2024"
                ],
                "confidence": None,
                "confidence_label": "Unspecified Parameters",
                "models": [],
                "evidence": [],
                "trace": trace.to_dict(),
                "metadata": {"parsed_query": parsed},
                "execution_time_ms": elapsed,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }

        trace.add_step(
            "QUERY_PARSED",
            f"Target: Task={task}, Location={parsed.get('location_name') or 'From Workspace'}, Dates={parsed.get('start_date')} to {parsed.get('end_date')}"
        )
        trace.add_step(
            "PARAMETERS_CONFIGURED",
            f"Configured search: AOI={parsed.get('bbox')}, Cloud Cover Filter<={parsed.get('max_cloud_cover')}%, "
            f"Pre-Event Window={parsed.get('t1_start_date')} to {parsed.get('t1_end_date')}, "
            f"Post-Event Window={parsed.get('t2_start_date')} to {parsed.get('t2_end_date')}"
        )

        # Execution Graph State
        execution_steps = []
        cumulative_transfer_bytes = 0
        step_count = 0
        step_budget = MAX_ORCHESTRATION_STEPS
        analysis_metadata = {}
        analysis_result = None
        evidence_items = []
        answer = ""

        # Step 2: Formulate Multi-Step Plan
        plan_steps = self._formulate_plan(parsed, has_images)
        trace.add_step("PLAN_FORMULATED", f"Selected {len(plan_steps)} traceable tool step(s): {', '.join([s['tool'] for s in plan_steps])}")

        # Step 3: Execute Plan Steps
        retrieved_scenes = []
        search_scenes_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for step_def in plan_steps:
            if step_count >= step_budget:
                trace.add_step("STEP_LIMIT_EXCEEDED", f"Maximum orchestration step limit ({step_budget}) reached.", status="failed")
                break

            step_count += 1
            tool_name = step_def["tool"]
            tool_args = dict(step_def.get("args", {}))
            step_desc = step_def.get("description", tool_name)
            t_start = time.time()

            # Dynamic scene_id resolution for automated retrieval
            if tool_name == "stac_composite_retrieve" and tool_args.get("scene_id") == "auto":
                col = tool_args.get("collection", "sentinel-2-l2a")
                role = tool_args.get("temporal_role")
                search_key = f"{col}_{role}" if role else col
                candidates = search_scenes_by_key.get(search_key) or search_scenes_by_key.get(col) or []

                already_retrieved_ids = {s.get("scene_id") for s in retrieved_scenes if s.get("scene_id")}
                available_candidates = [c for c in candidates if c.get("scene_id") not in already_retrieved_ids]
                if not available_candidates and candidates:
                    available_candidates = candidates

                if available_candidates:
                    chosen = available_candidates[0]
                    tool_args["scene_id"] = chosen["scene_id"]
                    step_desc = f"Retrieve bounded {chosen.get('collection', col)} composite window ({chosen['scene_id']})"
                    trace.add_step(
                        "SCENE_SELECTED",
                        f"Selected live {chosen.get('collection', col)} scene '{chosen['scene_id']}' for bounded window acquisition.",
                        status="completed"
                    )
                elif tool_args.get("optional"):
                    trace.add_step(
                        "SCENE_SELECTION_SKIPPED",
                        f"No additional candidate scene available for {col} ({role}); continuing with retrieved scenes.",
                        status="completed"
                    )
                    continue
                elif parsed.get("allow_offline_fallback", False) or parsed.get("use_offline_fixtures"):
                    if "rtc" in col.lower() or "sar" in col.lower() or "sentinel-1" in col.lower():
                        tool_args["scene_id"] = "local_sample_risat_sar_mumbai"
                        fallback_label = "local offline SAR sample raster (Mumbai)"
                    elif role == "t2":
                        tool_args["scene_id"] = "local_sample_temporal_t2_post_monsoon"
                        fallback_label = "local offline temporal T2 optical sample raster"
                    elif role == "t1":
                        tool_args["scene_id"] = "local_sample_temporal_t1_pre_monsoon"
                        fallback_label = "local offline temporal T1 optical sample raster"
                    else:
                        tool_args["scene_id"] = "local_sample_cartosat_optical_bengaluru"
                        fallback_label = "local offline optical sample raster (Bengaluru)"

                    trace.add_step(
                        "OFFLINE_FIXTURE_FALLBACK",
                        f"Live STAC search yielded no available scenes for '{col}'. Falling back to {fallback_label}.",
                        status="completed"
                    )
                else:
                    err_msg = f"No satellite scenes discovered in STAC collection '{col}' for the requested AOI and date range."
                    trace.add_step(f"TOOL_{tool_name.upper()}", err_msg, status="failed")
                    execution_steps.append({
                        "step_number": step_count,
                        "tool": tool_name,
                        "status": "failed",
                        "latency_ms": 0.0,
                        "summary": step_desc,
                        "error": err_msg
                    })
                    analysis_metadata.setdefault("errors", []).append(err_msg)
                    continue

            tool = self.tool_registry.get(tool_name)
            if not tool:
                trace.add_step(f"TOOL_{tool_name.upper()}", f"Tool '{tool_name}' not registered.", status="failed")
                continue

            trace.add_model(tool_name)

            # Check transfer budget before running acquisition tools
            if "retrieve" in tool_name:
                rem_budget = max_bytes - cumulative_transfer_bytes
                if rem_budget <= 0:
                    if tool_args.get("optional"):
                        trace.add_step(
                            f"TOOL_{tool_name.upper()}_SKIPPED",
                            f"Transfer budget of {max_bytes} bytes reached; skipping optional multi-scene retrieval.",
                            status="completed"
                        )
                        continue
                    trace.add_step(f"TOOL_{tool_name.upper()}", f"Cumulative transfer budget of {max_bytes} bytes exhausted.", status="failed")
                    analysis_metadata["budget_exhausted"] = True
                    execution_steps.append({
                        "step_number": step_count,
                        "tool": tool_name,
                        "status": "failed",
                        "latency_ms": 0.0,
                        "summary": step_desc,
                        "error": f"Cumulative transfer budget of {max_bytes} bytes exceeded."
                    })
                    break
                tool_args["max_bytes"] = rem_budget

            # Wire outputs from prior retrieval steps into temporal analysis tools
            if tool_name == "temporal_optical_change" and not tool_args.get("t1_scenes") and not tool_args.get("t1_b04"):
                t1_retrieved = [
                    s for s in retrieved_scenes
                    if s.get("temporal_role") == "t1" and s.get("status") in ("success", None) and s.get("array") is not None
                ]
                t2_retrieved = [
                    s for s in retrieved_scenes
                    if s.get("temporal_role") == "t2" and s.get("status") in ("success", None) and s.get("array") is not None
                ]

                # Fallback if temporal_role was not explicitly tagged on items
                if not t1_retrieved or not t2_retrieved:
                    valid_opt = [
                        s for s in retrieved_scenes
                        if s.get("status") in ("success", None) and s.get("array") is not None
                    ]
                    if len(valid_opt) >= 2:
                        t1_retrieved = [valid_opt[0]]
                        t2_retrieved = [valid_opt[1]]

                if not t1_retrieved or not t2_retrieved:
                    err_msg = "Upstream Sentinel-2 optical scene retrieval failed or returned incomplete rasters; bi-temporal optical change calculation blocked."
                    trace.add_step(f"TOOL_{tool_name.upper()}", err_msg, status="blocked")
                    execution_steps.append({
                        "step_number": step_count,
                        "tool": tool_name,
                        "status": "blocked",
                        "latency_ms": 0.0,
                        "summary": step_desc,
                        "error": err_msg
                    })
                    analysis_metadata["temporal_optical_change"] = {"status": "blocked", "error": err_msg}
                    continue

                def extract_scene_dict(s, fallback_id="scene"):
                    arr = s.get("array")
                    meta = s.get("raster_metadata") or s.get("metadata") or {}
                    b04 = arr[:, :, 0] if arr.ndim == 3 else arr
                    b08 = arr[:, :, 1] if arr.ndim == 3 and arr.shape[2] > 1 else arr
                    scl = arr[:, :, 2] if arr.ndim == 3 and arr.shape[2] > 2 else None
                    return {
                        "scene_id": s.get("scene_id") or meta.get("scene_id") or fallback_id,
                        "datetime": s.get("datetime") or meta.get("datetime"),
                        "b04": b04,
                        "b08": b08,
                        "scl": scl,
                        "meta": meta,
                        "cloud_cover": s.get("cloud_cover"),
                        "provider": (s.get("provenance") or {}).get("provider")
                    }

                tool_args["t1_scenes"] = [extract_scene_dict(s, f"t1_{i+1}") for i, s in enumerate(t1_retrieved)]
                tool_args["t2_scenes"] = [extract_scene_dict(s, f"t2_{i+1}") for i, s in enumerate(t2_retrieved)]
                tool_args["cloud_threshold"] = parsed.get("max_cloud_cover", 20.0)
                tool_args["composite_method"] = "median"

                # Maintain legacy scalar args for compatibility
                tool_args["t1_b04"] = tool_args["t1_scenes"][0]["b04"]
                tool_args["t1_b08"] = tool_args["t1_scenes"][0]["b08"]
                tool_args["t2_b04"] = tool_args["t2_scenes"][0]["b04"]
                tool_args["t2_b08"] = tool_args["t2_scenes"][0]["b08"]
                tool_args["t1_scl"] = tool_args["t1_scenes"][0]["scl"]
                tool_args["t2_scl"] = tool_args["t2_scenes"][0]["scl"]
                tool_args["t1_meta"] = tool_args["t1_scenes"][0]["meta"]
                tool_args["t2_meta"] = tool_args["t2_scenes"][0]["meta"]
                tool_args["t1_datetime"] = parsed.get("t1_date") or tool_args["t1_scenes"][0]["datetime"] or "2023-01-01T10:00:00Z"
                tool_args["t2_datetime"] = parsed.get("t2_date") or tool_args["t2_scenes"][0]["datetime"] or "2024-01-01T10:00:00Z"

            elif tool_name == "temporal_sar_change" and not tool_args.get("t1_vv"):
                valid_sar = (
                    len(retrieved_scenes) >= 2 and
                    retrieved_scenes[0].get("status") in ("success", None) and
                    retrieved_scenes[1].get("status") in ("success", None) and
                    retrieved_scenes[0].get("array") is not None and
                    retrieved_scenes[1].get("array") is not None
                )
                if not valid_sar:
                    err_msg = "Upstream Sentinel-1 RTC SAR scene retrieval failed or returned incomplete rasters; bi-temporal SAR change calculation blocked."
                    trace.add_step(f"TOOL_{tool_name.upper()}", err_msg, status="blocked")
                    execution_steps.append({
                        "step_number": step_count,
                        "tool": tool_name,
                        "status": "blocked",
                        "latency_ms": 0.0,
                        "summary": step_desc,
                        "error": err_msg
                    })
                    analysis_metadata["temporal_sar_change"] = {"status": "blocked", "error": err_msg}
                    continue

                s1 = retrieved_scenes[0]
                s2 = retrieved_scenes[1]
                arr1 = s1.get("array")
                arr2 = s2.get("array")
                meta1 = s1.get("raster_metadata") or s1.get("metadata") or {}
                meta2 = s2.get("raster_metadata") or s2.get("metadata") or {}
                vv_1 = arr1[:, :, 0] if arr1.ndim == 3 else arr1
                vh_1 = arr1[:, :, 1] if arr1.ndim == 3 and arr1.shape[2] > 1 else arr1 * 0.2
                vv_2 = arr2[:, :, 0] if arr2.ndim == 3 else arr2
                vh_2 = arr2[:, :, 1] if arr2.ndim == 3 and arr2.shape[2] > 1 else arr2 * 0.2
                tool_args["t1_vv"] = vv_1
                tool_args["t1_vh"] = vh_1
                tool_args["t2_vv"] = vv_2
                tool_args["t2_vh"] = vh_2
                tool_args["t1_meta"] = meta1
                tool_args["t2_meta"] = meta2
                tool_args["t1_datetime"] = parsed.get("t1_date") or meta1.get("datetime") or "2023-01-01T10:00:00Z"
                tool_args["t2_datetime"] = parsed.get("t2_date") or meta2.get("datetime") or "2024-01-01T10:00:00Z"

            elif tool_name == "optical_sar_fusion" and not tool_args.get("images"):
                opt_scene = None
                sar_scene = None
                for s in retrieved_scenes:
                    coll = str(s.get("collection", "")).lower()
                    if ("sentinel-2" in coll or "optical" in coll or s.get("ndvi_result")) and opt_scene is None:
                        opt_scene = s
                    elif ("sentinel-1" in coll or "rtc" in coll or "sar" in coll or s.get("dprvi_result")) and sar_scene is None:
                        sar_scene = s

                if opt_scene is None and len(retrieved_scenes) >= 1:
                    opt_scene = retrieved_scenes[0]
                if sar_scene is None and len(retrieved_scenes) >= 2:
                    sar_scene = retrieved_scenes[1]

                valid_fusion = (
                    opt_scene is not None and
                    sar_scene is not None and
                    opt_scene.get("status") in ("success", None) and
                    sar_scene.get("status") in ("success", None) and
                    opt_scene.get("array") is not None and
                    sar_scene.get("array") is not None
                )
                if not valid_fusion:
                    err_msg = "Upstream optical or SAR scene retrieval failed or returned incomplete rasters; cross-sensor optical-SAR fusion blocked."
                    trace.add_step(f"TOOL_{tool_name.upper()}", err_msg, status="blocked")
                    execution_steps.append({
                        "step_number": step_count,
                        "tool": tool_name,
                        "status": "blocked",
                        "latency_ms": 0.0,
                        "summary": step_desc,
                        "error": err_msg
                    })
                    analysis_metadata["optical_sar_fusion"] = {"status": "blocked", "error": err_msg}
                    continue

                active_aoi = parsed.get("bbox")
                opt_meta = opt_scene.get("raster_metadata") or opt_scene.get("metadata") or {}
                sar_meta = sar_scene.get("raster_metadata") or sar_scene.get("metadata") or {}
                opt_prov = opt_scene.get("provenance") or opt_meta
                sar_prov = sar_scene.get("provenance") or sar_meta

                validation_aoi = active_aoi

                if (opt_prov.get("aoi_containment_status") == "out_of_bounds_clamped" or
                    sar_prov.get("aoi_containment_status") == "out_of_bounds_clamped"):
                    trace.add_step(
                        "AOI_CONTAINMENT_ADVISORY",
                        "Local demonstration fixture does not intersect requested AOI coordinates; clamped to raster extent for offline demonstration.",
                        status="completed"
                    )

                # Validate pair using CrossSensorValidator before execution (P1-1 integration)
                from app.acquisition.cross_sensor_validator import CrossSensorValidator
                cs_val = CrossSensorValidator.validate_pair(
                    optical_input=opt_scene["array"],
                    sar_input=sar_scene["array"],
                    optical_metadata=opt_meta,
                    sar_metadata=sar_meta,
                    optical_provenance=opt_prov,
                    sar_provenance=sar_prov,
                    aoi_bbox=validation_aoi
                )

                # Construct structured independent optical + SAR provenance for metadata & UI
                analysis_metadata["optical_sar_provenance"] = {
                    "optical": {
                        "catalog_id": opt_scene.get("scene_id"),
                        "collection": opt_scene.get("collection") or "cartosat-optical-l2",
                        "provider": opt_prov.get("provider") or ("Element84 Earth Search (AWS)" if "sentinel-2" in str(opt_scene.get("collection", "")).lower() else "Local Demonstration Archive"),
                        "sensor": opt_prov.get("sensor") or ("Sentinel-2 MSI" if "sentinel-2" in str(opt_scene.get("collection", "")).lower() else "Cartosat-style synthetic demonstration optical scene"),
                        "datetime": opt_scene.get("datetime") or opt_meta.get("datetime") or (opt_prov.get("datetime") if isinstance(opt_prov, dict) else None),
                        "asset_key": "B04_B08_SCL",
                        "crs": opt_meta.get("crs_display") or opt_meta.get("crs"),
                        "dimensions": f"{opt_meta.get('width', 0)}x{opt_meta.get('height', 0)}",
                        "validation_status": "validated" if opt_scene.get("status") == "success" else "unsupported",
                        "asset_hrefs": {k: SafeWindowedCOGReader._sanitize_url(v) for k, v in opt_prov.get("asset_hrefs", {}).items()} if opt_prov.get("asset_hrefs") else None
                    },
                    "sar": {
                        "catalog_id": sar_scene.get("scene_id"),
                        "collection": sar_scene.get("collection") or "risat-c-band-sar",
                        "provider": sar_prov.get("provider") or ("Microsoft Planetary Computer" if "sentinel-1" in str(sar_scene.get("collection", "")).lower() else "Local Demonstration Archive"),
                        "sensor": sar_prov.get("sensor") or ("Sentinel-1 C-SAR (RTC)" if "sentinel-1" in str(sar_scene.get("collection", "")).lower() else "RISAT-style synthetic demonstration SAR scene"),
                        "datetime": sar_scene.get("datetime") or sar_meta.get("datetime") or (sar_prov.get("datetime") if isinstance(sar_prov, dict) else None),
                        "asset_key": "VV_VH",
                        "polarization": sar_prov.get("polarization") or "DUAL_VV_VH",
                        "calibration_status": sar_prov.get("calibration_quantity") or "linear_power",
                        "terrain_correction_status": sar_prov.get("sar_terrain_layover_shadow_mask") or "unavailable",
                        "crs": sar_meta.get("crs_display") or sar_meta.get("crs"),
                        "dimensions": f"{sar_meta.get('width', 0)}x{sar_meta.get('height', 0)}",
                        "validation_status": "validated" if sar_scene.get("status") == "success" else "unsupported",
                        "asset_hrefs": {k: SafeWindowedCOGReader._sanitize_url(v) for k, v in sar_prov.get("asset_hrefs", {}).items()} if sar_prov.get("asset_hrefs") else None
                    },
                    "cross_sensor_verification": {
                        "spatial_overlap_pct": cs_val.get("spatial_overlap_pct") if cs_val.get("spatial_overlap_pct") is not None else 0.0,
                        "crs_matching": bool(opt_meta.get("crs") and sar_meta.get("crs") and opt_meta.get("crs") == sar_meta.get("crs")),
                        "temporal_delta_days": cs_val.get("temporal_delta_days"),
                        "co_registered": cs_val.get("co_registered", False)
                    },
                    "optical_cloud_filter": {
                        "max_cloud_cover_pct": parsed.get("max_cloud_cover", 20.0),
                        "actual_cloud_cover_pct": opt_scene.get("cloud_cover") or opt_meta.get("cloud_cover")
                    },
                    "scientific_distinction": {
                        "optical_measurement": "Solar spectral surface reflectance (albedo across visible and NIR wavelengths).",
                        "sar_measurement": "Coherent microwave electromagnetic backscatter (dielectric permittivity, moisture, and surface roughness).",
                        "physical_assertion": "Optical reflectance and SAR backscatter are non-interchangeable remote sensing measurements."
                    }
                }

                if not cs_val.get("is_compatible", True):
                    rejection_reason = cs_val.get("rejection_reason") or "Cross-sensor validation failed."
                    err_msg = f"Cross-sensor validation rejected optical-SAR pair: {rejection_reason}"
                    trace.add_step(f"TOOL_{tool_name.upper()}", err_msg, status="unsupported")
                    execution_steps.append({
                        "step_number": step_count,
                        "tool": tool_name,
                        "status": "unsupported",
                        "latency_ms": 0.0,
                        "summary": step_desc,
                        "error": err_msg
                    })
                    analysis_metadata["optical_sar_fusion"] = {
                        "status": "unsupported",
                        "analysis_status": "unsupported_validation",
                        "answer": f"Optical-SAR cross-sensor fusion analysis rejected: {rejection_reason}",
                        "cross_sensor_validation": cs_val,
                        "confidence": None,
                        "confidence_label": "Not available (Cross-Sensor Validation Rejected)",
                        "water_pct": None,
                        "urban_pct": None,
                        "veg_pct": None,
                        "limitations": cs_val.get("scientific_limitations", []),
                        "warnings": cs_val.get("warnings", [])
                    }
                    continue

                if cs_val.get("warnings"):
                    for w in cs_val["warnings"]:
                        trace.add_step(f"TOOL_{tool_name.upper()}_ADVISORY", w, status="completed")

                tool_args["images"] = [opt_scene["array"], sar_scene["array"]]
                tool_args["metadata"] = {
                    "optical": opt_meta,
                    "sar": sar_meta,
                    "aoi_bbox": active_aoi,
                    "cross_sensor_validation": cs_val
                }
                tool_args["query"] = query

            elif tool_name == "land_cover_classification" and not tool_args.get("images"):
                arr = None
                if retrieved_scenes:
                    arr = retrieved_scenes[0].get("array")
                elif has_images and len(images) > 0:
                    arr = images[0]

                if arr is not None:
                    tool_args["images"] = [arr]
                else:
                    tool_args["images"] = []
                tool_args["query"] = query

            # Execute tool with bounded retries (MAX_TOOL_RETRIES)
            tool_out = None
            step_status = "failed"
            step_error = None
            t_latency = 0.0

            for attempt in range(1, MAX_TOOL_RETRIES + 1):
                t_start = time.time()
                try:
                    tool_out = tool.run(**tool_args)
                    t_latency = round((time.time() - t_start) * 1000, 2)

                    if tool_out.get("status") in ("success", "unsupported", None) or str(tool_out.get("status", "")).startswith("unsupported"):
                        step_status = "completed"
                        step_error = None
                        if attempt > 1:
                            trace.add_step(
                                f"TOOL_{tool_name.upper()}_RETRY",
                                f"Tool '{tool_name}' succeeded on retry attempt {attempt}/{MAX_TOOL_RETRIES} ({t_latency} ms)"
                            )
                        break
                    else:
                        step_error = tool_out.get("error") or f"Tool returned failure status '{tool_out.get('status')}'"
                        if "budget" in step_error.lower():
                            analysis_metadata["budget_exhausted"] = True
                        is_non_retryable = any(kw in step_error.lower() for kw in [
                            "validation", "degenerate", "identical", "unsupported", "invalid bbox", "crs mismatch", "budget"
                        ])
                        if is_non_retryable or attempt >= MAX_TOOL_RETRIES:
                            step_status = "failed"
                            break
                        trace.add_step(
                            f"TOOL_{tool_name.upper()}",
                            f"Attempt {attempt}/{MAX_TOOL_RETRIES} failed: {step_error}. Retrying...",
                            status="retrying"
                        )
                        time.sleep(0.05 * attempt)
                except Exception as tool_err:
                    t_latency = round((time.time() - t_start) * 1000, 2)
                    step_error = str(tool_err)
                    if "budget" in step_error.lower():
                        analysis_metadata["budget_exhausted"] = True
                    is_non_retryable = any(kw in step_error.lower() for kw in [
                        "validation", "degenerate", "identical", "unsupported", "invalid bbox", "crs mismatch", "budget"
                    ])
                    if is_non_retryable or attempt >= MAX_TOOL_RETRIES:
                        step_status = "failed"
                        logger.error(f"Error executing tool '{tool_name}' after {attempt} attempt(s): {tool_err}")
                        break
                    trace.add_step(
                        f"TOOL_{tool_name.upper()}",
                        f"Attempt {attempt}/{MAX_TOOL_RETRIES} encountered error: {step_error}. Retrying...",
                        status="retrying"
                    )
                    time.sleep(0.05 * attempt)

            # Record step in DAG execution steps
            step_record = {
                "step_number": step_count,
                "tool": tool_name,
                "status": step_status,
                "latency_ms": t_latency,
                "summary": step_desc
            }
            if step_error:
                step_record["error"] = step_error
            execution_steps.append(step_record)

            trace.add_step(
                f"TOOL_{tool_name.upper()}",
                f"{step_desc} -> {step_status} ({t_latency} ms)" + (f": {step_error}" if step_error else ""),
                status=step_status
            )

            if tool_out:
                # Update transfer telemetry
                if "size_bytes" in tool_out:
                    cumulative_transfer_bytes += tool_out.get("size_bytes", 0)

                if tool_name == "stac_search" and tool_out.get("status") == "success":
                    col = tool_args.get("collection", "default")
                    role = tool_args.get("temporal_role")
                    search_key = f"{col}_{role}" if role else col
                    scenes_found = tool_out.get("scenes", [])
                    search_scenes_by_key[search_key] = scenes_found
                    search_scenes_by_key.setdefault(col, []).extend(scenes_found)

                    disc_list = analysis_metadata.setdefault("discovered_scenes", [])
                    for sc in scenes_found:
                        disc_list.append({
                            "scene_id": sc.get("scene_id"),
                            "datetime": sc.get("datetime"),
                            "collection": sc.get("collection") or col,
                            "provider": tool_out.get("provider"),
                            "cloud_cover": sc.get("cloud_cover"),
                            "sensor": sc.get("sensor")
                        })

                if tool_name == "stac_composite_retrieve":
                    retrieved_scenes.append(tool_out)
                    scene_prov = tool_out.get("provenance") or {}
                    if scene_prov.get("aoi_containment_status") == "out_of_bounds_clamped":
                        analysis_metadata["aoi_containment_status"] = "out_of_bounds_clamped"
                        analysis_metadata["actual_fixture_bounds_wgs84"] = scene_prov.get("actual_fixture_bounds_wgs84")
                    if "scientific_limitations" in tool_out:
                        lims = analysis_metadata.setdefault("scientific_limitations", [])
                        for l in tool_out.get("scientific_limitations", []):
                            if l not in lims:
                                lims.append(l)

                # Integrate tool outputs into orchestration state
                if "ndvi_result" in tool_out and tool_out["ndvi_result"]:
                    analysis_metadata["ndvi_result"] = tool_out["ndvi_result"]
                if "dprvi_result" in tool_out and tool_out["dprvi_result"]:
                    analysis_metadata["dprvi_result"] = tool_out["dprvi_result"]
                if "delta_ndvi_mean" in tool_out:
                    tool_out["parameters"] = {
                        "t1_window": (parsed.get("t1_start_date"), parsed.get("t1_end_date")),
                        "t2_window": (parsed.get("t2_start_date"), parsed.get("t2_end_date")),
                        "max_cloud_cover_pct": parsed.get("max_cloud_cover", 20.0),
                        "composite_method": tool_out.get("composite_method", "valid_pixel_temporal_median"),
                        "scl_mask_policy": tool_out.get("scl_mask_policy"),
                        "t1_scenes_contributing": tool_out.get("contributing_scene_ids_t1", []),
                        "t2_scenes_contributing": tool_out.get("contributing_scene_ids_t2", []),
                        "rejected_scenes": tool_out.get("rejected_scenes", []),
                        "cumulative_transfer_bytes": cumulative_transfer_bytes
                    }
                    analysis_metadata["temporal_optical_change"] = tool_out
                    if tool_out.get("colorized_preview_base64"):
                        evidence_items.append({
                            "id": "ev_temporal_optical_change_map",
                            "type": "change_map",
                            "title": "Sentinel-2 Calibrated Bi-Temporal NDVI Change Map",
                            "description": "Multi-scene SCL-screened Delta-NDVI composite change map (Red: Vegetation Loss, Green: Vegetation Gain, Gray: Stable)",
                            "data_base64": tool_out.get("colorized_preview_base64"),
                            "statistics": {
                                "mean_delta_ndvi": tool_out.get("delta_ndvi_mean"),
                                "vegetation_loss_pct": tool_out.get("vegetation_loss_pct"),
                                "vegetation_gain_pct": tool_out.get("vegetation_gain_pct"),
                                "joint_clear_pixels": tool_out.get("joint_valid_pixel_count"),
                                "composite_method": tool_out.get("composite_method"),
                                "t1_obs_depth": tool_out.get("t1_mean_observation_count"),
                                "t2_obs_depth": tool_out.get("t2_mean_observation_count")
                            }
                        })
                    if tool_out.get("visual_inspector") and tool_out["visual_inspector"].get("status") == "success":
                        vi = tool_out["visual_inspector"]
                        t1_insp = vi.get("t1", {})
                        evidence_items.append({
                            "id": "ev_temporal_composite_inspector",
                            "type": "mask",
                            "title": "Sentinel-2 Multi-Scene Composite & SCL Mask Inspector",
                            "description": "Interactive visual inspection of Red (B04), NIR (B08), observation depth, and SCL cloud masking for T1/T2 composites.",
                            "data_base64": t1_insp.get("layers", {}).get("obs_depth") or tool_out.get("colorized_preview_base64"),
                            "statistics": {
                                "t1_clear_coverage_pct": t1_insp.get("stats", {}).get("clear_coverage_pct"),
                                "t1_mean_obs_depth": t1_insp.get("stats", {}).get("mean_obs_depth"),
                                "t2_clear_coverage_pct": vi.get("t2", {}).get("stats", {}).get("clear_coverage_pct"),
                                "t2_mean_obs_depth": vi.get("t2", {}).get("stats", {}).get("mean_obs_depth")
                            },
                            "inspector_data": vi
                        })
                if "delta_dprvi_mean" in tool_out:
                    tool_out["parameters"] = {
                        "t1_window": (parsed.get("t1_start_date"), parsed.get("t1_end_date")),
                        "t2_window": (parsed.get("t2_start_date"), parsed.get("t2_end_date"))
                    }
                    analysis_metadata["temporal_sar_change"] = tool_out
                    if tool_out.get("colorized_preview_base64"):
                        evidence_items.append({
                            "id": "ev_temporal_sar_change_map",
                            "type": "change_map",
                            "title": "Sentinel-1 RTC Calibrated Bi-Temporal DpRVI Change Map",
                            "description": "Calibrated Delta-DpRVI change map (Red-Orange: Radar Canopy Loss, Green: Radar Canopy Gain, Gray: Stable)",
                            "data_base64": tool_out.get("colorized_preview_base64"),
                            "statistics": {
                                "mean_delta_dprvi": tool_out.get("delta_dprvi_mean"),
                                "radar_canopy_loss_pct": tool_out.get("radar_canopy_loss_pct"),
                                "radar_canopy_gain_pct": tool_out.get("radar_canopy_gain_pct"),
                                "delta_vv_db": tool_out.get("delta_vv_db")
                            }
                        })
                if tool_name == "optical_sar_fusion" and step_status == "completed":
                    analysis_metadata["optical_sar_fusion"] = tool_out
                    if tool_out.get("evidence") and isinstance(tool_out["evidence"], list):
                        evidence_items.extend(tool_out["evidence"])
                    fused_b64 = (
                        tool_out.get("fused_evidence")
                        or tool_out.get("fused_preview_base64")
                        or tool_out.get("colorized_preview_base64")
                        or tool_out.get("preview_base64")
                    )
                    if fused_b64 and not any(e.get("type") == "fused" for e in evidence_items):
                        evidence_items.append({
                            "id": "ev_optical_sar_fused_composite",
                            "type": "fused",
                            "title": "Sentinel-2 & Sentinel-1 RTC Optical-SAR Cross-Sensor Composite",
                            "description": "Co-registered multimodal optical surface reflectance and calibrated RTC SAR backscatter",
                            "data_base64": fused_b64,
                            "statistics": tool_out.get("statistics") or {}
                        })
                if "quality_screening" in tool_out:
                    analysis_metadata["quality_screening"] = tool_out["quality_screening"]
                if "provenance" in tool_out:
                    analysis_metadata["provenance"] = tool_out["provenance"]
                if tool_name == "land_cover_classification":
                    analysis_metadata["land_cover_classification"] = tool_out
                    top_preds = tool_out.get("top_predictions", [])
                    evidence_items.append({
                        "id": "ev_bigearthnet_classification",
                        "type": "classification",
                        "title": "BigEarthNet ResNet-50 CORINE Land Cover Classification",
                        "description": "Multi-label classification of 10-band Sentinel-2 imagery into 19 CORINE land-cover classes",
                        "data_base64": None,
                        "statistics": {
                            "predicted_classes": tool_out.get("predicted_classes", []),
                            "top_predictions": top_preds,
                            "class_probabilities": tool_out.get("class_probabilities", {})
                        }
                    })

                analysis_result = tool_out

        analysis_metadata["retrieved_scenes"] = [
            {
                "scene_id": s.get("scene_id"),
                "collection": s.get("collection"),
                "provider": (s.get("provenance") or {}).get("provider") or ("Element84 Earth Search (AWS)" if "sentinel-2" in str(s.get("collection")).lower() else "Microsoft Planetary Computer"),
                "bands": s.get("bands_retrieved") or ["b04", "b08", "scl"],
                "size_bytes": s.get("size_bytes", 0)
            }
            for s in retrieved_scenes
        ]

        # Step 4: Synthesize Answer Text & Formulate Claims
        overall_status, confidence, confidence_label, answer, candidate_claims = self._synthesize_claims(
            task=task,
            analysis_metadata=analysis_metadata,
            execution_steps=execution_steps,
            max_bytes=max_bytes,
            analysis_result=analysis_result
        )

        # Step 5: Evidence Package Building & Deterministic Verification / Contradiction Detection
        analysis_dict = {
            "id": trace_id,
            "query": query,
            "task": task,
            "metadata": analysis_metadata,
            "evidence": evidence_items
        }
        pkg = EvidencePackageBuilder.from_analysis_response(analysis_dict)
        trace.add_step("EVIDENCE_ASSEMBLED", f"Extracted {len(pkg.physical_measurements)} physical measurement(s) and {len(pkg.evidence_artifacts)} artifact(s)")

        conflict_info = ConflictInfoSchema(conflict_detected=False)

        if overall_status == "success":
            # 5a. Cross-Evidence Contradiction Evaluation
            conflict_eval = ConflictDetector.evaluate(
                task=task,
                query=query,
                primary_result=analysis_result or {"answer": answer, "evidence": evidence_items},
                metadata=analysis_metadata,
                evidence_package=pkg,
                candidate_claims=candidate_claims
            )

            # 5b. Deterministic Verification
            v_out = None
            if candidate_claims:
                verifier_tool = self.tool_registry.get("deterministic_verifier")
                v_out = verifier_tool.run(evidence_package=pkg, candidate_claims=candidate_claims, answer_text=answer)

            contradictions = list(v_out.get("contradictions", [])) if v_out else []
            det_passed = v_out.get("passed", True) if v_out else True

            # If conflict detector detected conflict OR deterministic verifier found contradictions
            if conflict_eval.conflict_detected or len(contradictions) > 0:
                conf_type = conflict_eval.conflict_type or "DETERMINISTIC_CLAIM_CONTRADICTION"
                conf_details = conflict_eval.reanalysis_reason or "; ".join(contradictions)
                trace.add_step(
                    "CONFLICT_DETECTED",
                    f"Contradiction identified [{conf_type}]: {conf_details}"
                )
                conflict_info = ConflictInfoSchema(
                    conflict_detected=True,
                    conflict_type=conf_type,
                    conflict_details=conf_details,
                    reanalysis_performed=False,
                    reanalysis_tool=conflict_eval.reanalysis_tool
                )
                overall_status = "contradicted"
                confidence = 0.35
                confidence_label = "Contradiction Detected"
                answer += f"\n\n[CONTRADICTION DETECTED]: {conf_details}"
                if v_out is None:
                    v_out = {
                        "passed": False,
                        "deterministic_passed": False,
                        "verification_status": "contradicted",
                        "failures": [conf_details],
                        "deterministic_failures": [conf_details],
                        "claims": [c.dict() if hasattr(c, "dict") else c.model_dump() for c in candidate_claims],
                        "contradictions": [conf_details],
                        "unsupported_claims": [],
                        "summary_explanation": f"Contradiction identified: {conf_details}",
                        "missing_evidence": [],
                        "scientific_limitations": getattr(pkg, "known_limitations", []) or [],
                        "recommended_checks": []
                    }
                else:
                    v_out["passed"] = False
                    v_out["deterministic_passed"] = False
                    v_out["verification_status"] = "contradicted"
                    if conf_details not in v_out.get("contradictions", []):
                        v_out.setdefault("contradictions", []).append(conf_details)
            elif conflict_eval.status in ("unsupported", "insufficient_evidence"):
                trace.add_step(
                    "INSUFFICIENT_EVIDENCE",
                    f"Cross-evidence evaluation uncertain: {conflict_eval.uncertainty_reason}"
                )
                conflict_info = ConflictInfoSchema(
                    conflict_detected=False,
                    conflict_details=conflict_eval.uncertainty_reason
                )
                if task in ("optical_sar", "optical_sar_fusion", "optical_sar_joint"):
                    overall_status = "unsupported"
                    confidence = None
                    confidence_label = "Insufficient Evidence"
                    if v_out:
                        v_out["passed"] = False
                        v_out["verification_status"] = "unsupported"
            else:
                # No conflict detected
                conflict_info = ConflictInfoSchema(conflict_detected=False)
                if v_out:
                    confidence = 0.95 if det_passed else 0.40
                    confidence_label = "Deterministic Verified" if det_passed else "Verification Flagged"
                    trace.add_step("DETERMINISTIC_VERIFIED", f"Checks A-L: {'PASSED' if det_passed else 'VIOLATIONS DETECTED'} ({len(v_out.get('failures', []))} violation(s))")
                else:
                    confidence = 0.95
                    confidence_label = "Deterministic Verified"
        else:
            # overall_status != "success" (e.g. failed, unsupported, budget_exhausted)
            conflict_info = ConflictInfoSchema(conflict_detected=False)
            v_out = {
                "passed": False,
                "deterministic_passed": False,
                "verification_status": "unsupported",
                "failures": [answer] if overall_status != "success" else [],
                "deterministic_failures": [answer] if overall_status != "success" else [],
                "claims": [],
                "contradictions": [],
                "unsupported_claims": [],
                "summary_explanation": f"Workflow ended with status '{overall_status}'. Quantitative verification not run.",
                "missing_evidence": [],
                "scientific_limitations": getattr(pkg, "known_limitations", []) or [],
                "recommended_checks": []
            }
            trace.add_step("VERIFICATION_SKIPPED", f"Verification skipped for status '{overall_status}'", status="completed")

        # Step 6: Scientific Report Generation
        report = ScientificReportBuilder.build_report(analysis_dict, evidence_package=pkg)
        trace.add_step("REPORT_GENERATED", f"Structured Scientific Report '{report.report_id}' compiled")

        elapsed_total = round((time.time() - start_time) * 1000, 2)
        trace_data = trace.to_dict()
        trace_data["execution_steps"] = execution_steps
        trace_data["cumulative_transfer_bytes"] = cumulative_transfer_bytes
        trace_data["transfer_budget_bytes"] = max_bytes

        final_res = {
            "id": trace_id,
            "status": overall_status,
            "task": task,
            "mode": "orchestration",
            "answer": answer,
            "is_clarification_required": False,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "models": trace.models_selected,
            "evidence": evidence_items,
            "trace": trace_data,
            "metadata": analysis_metadata,
            "conflict_info": conflict_info.dict() if hasattr(conflict_info, "dict") else conflict_info.model_dump(),
            "verification": v_out,
            "reproducible_report": report.dict() if hasattr(report, "dict") else report.model_dump(),
            "execution_time_ms": elapsed_total,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        return self._sanitize_for_json(final_res)

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: MultiStepOrchestrator._sanitize_for_json(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
        elif isinstance(obj, list):
            return [MultiStepOrchestrator._sanitize_for_json(v) for v in obj]
        return obj

    def _formulate_plan(self, parsed: Dict[str, Any], has_images: bool) -> List[Dict[str, Any]]:
        """Formulates an ordered, bounded list of tools to execute based on parsed query."""
        task = parsed["task"]
        bbox = parsed.get("bbox") or [72.8, 18.9, 72.9, 19.0]
        use_offline = parsed.get("use_offline_fixtures", False)
        cloud_thresh = parsed.get("max_cloud_cover")
        if cloud_thresh is None:
            cloud_thresh = 20.0
        steps = []

        if task == "temporal_optical_change":
            if not has_images and not use_offline:
                (t1_start, t1_end), (t2_start, t2_end) = self._resolve_temporal_search_windows(parsed)
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-2-l2a",
                        "bbox": bbox,
                        "start_date": t1_start,
                        "end_date": t1_end,
                        "max_cloud_cover": cloud_thresh,
                        "temporal_role": "t1",
                        "limit": 3
                    },
                    "description": f"Discover Pre-Event (T1) Sentinel-2 optical scenes ({t1_start} to {t1_end}; cloud<={cloud_thresh}%)"
                })
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-2-l2a",
                        "bbox": bbox,
                        "start_date": t2_start,
                        "end_date": t2_end,
                        "max_cloud_cover": cloud_thresh,
                        "temporal_role": "t2",
                        "limit": 3
                    },
                    "description": f"Discover Post-Event (T2) Sentinel-2 optical scenes ({t2_start} to {t2_end}; cloud<={cloud_thresh}%)"
                })
                # Primary and secondary candidate retrieval for Pre-Event (T1) multi-scene compositing
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "temporal_role": "t1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire Pre-Event (T1) Sentinel-2 optical composite with SCL (Candidate 1)"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "temporal_role": "t1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True,
                        "optional": True
                    },
                    "description": "Acquire Pre-Event (T1) Sentinel-2 optical composite with SCL (Candidate 2, Multi-Scene)"
                })
                # Primary and secondary candidate retrieval for Post-Event (T2) multi-scene compositing
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "temporal_role": "t2",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire Post-Event (T2) Sentinel-2 optical composite with SCL (Candidate 1)"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "temporal_role": "t2",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True,
                        "optional": True
                    },
                    "description": "Acquire Post-Event (T2) Sentinel-2 optical composite with SCL (Candidate 2, Multi-Scene)"
                })
                steps.append({
                    "tool": "temporal_optical_change",
                    "args": {
                        "composite_method": "median"
                    },
                    "description": "Compute multi-scene bi-temporal Sentinel-2 optical change and SCL filtering"
                })
            else:
                # Search / retrieve / process T1 and T2 using local demo fixtures or workspace images
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_temporal_t1_pre_monsoon" if not has_images else "workspace_img_1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire Pre-Event (T1) Sentinel-2 optical composite with SCL"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_temporal_t2_post_monsoon" if not has_images else "workspace_img_2",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire Post-Event (T2) Sentinel-2 optical composite with SCL"
                })
                steps.append({
                    "tool": "temporal_optical_change",
                    "args": {},
                    "description": "Compute bi-temporal Sentinel-2 optical change and SCL filtering"
                })
        elif task == "temporal_sar_change":
            if not has_images and not use_offline:
                (t1_start, t1_end), (t2_start, t2_end) = self._resolve_temporal_search_windows(parsed)
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-1-rtc",
                        "bbox": bbox,
                        "start_date": t1_start,
                        "end_date": t1_end,
                        "temporal_role": "t1",
                        "limit": 3
                    },
                    "description": f"Discover Pre-Event (T1) Sentinel-1 RTC SAR scenes ({t1_start} to {t1_end})"
                })
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-1-rtc",
                        "bbox": bbox,
                        "start_date": t2_start,
                        "end_date": t2_end,
                        "temporal_role": "t2",
                        "limit": 3
                    },
                    "description": f"Discover Post-Event (T2) Sentinel-1 RTC SAR scenes ({t2_start} to {t2_end})"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "temporal_role": "t1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire Pre-Event (T1) Sentinel-1 RTC dual-pol composite"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "temporal_role": "t2",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire Post-Event (T2) Sentinel-1 RTC dual-pol composite"
                })
                steps.append({
                    "tool": "temporal_sar_change",
                    "args": {},
                    "description": "Compute bi-temporal Sentinel-1 RTC SAR backscatter and DpRVI change"
                })
            else:
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_risat_sar_mumbai" if not has_images else "workspace_img_1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire Pre-Event (T1) Sentinel-1 RTC dual-pol composite"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_risat_sar_mumbai" if not has_images else "workspace_img_2",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire Post-Event (T2) Sentinel-1 RTC dual-pol composite"
                })
                steps.append({
                    "tool": "temporal_sar_change",
                    "args": {},
                    "description": "Compute bi-temporal Sentinel-1 RTC SAR backscatter and DpRVI change"
                })
        elif task == "optical_sar":
            if not has_images and not use_offline:
                s_date = parsed.get("start_date") or "2023-01-01"
                e_date = parsed.get("end_date") or "2024-01-01"
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-2-l2a",
                        "bbox": bbox,
                        "start_date": s_date,
                        "end_date": e_date,
                        "max_cloud_cover": cloud_thresh,
                        "limit": 5
                    },
                    "description": f"Discover Sentinel-2 L2A optical scenes via Element84 Earth Search (cloud<={cloud_thresh}%)"
                })
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-1-rtc",
                        "bbox": bbox,
                        "start_date": s_date,
                        "end_date": e_date,
                        "limit": 5
                    },
                    "description": "Discover Sentinel-1 RTC SAR scenes via Microsoft Planetary Computer"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire bounded Sentinel-2 optical composite with SCL"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire bounded Sentinel-1 RTC SAR composite"
                })
                steps.append({
                    "tool": "optical_sar_fusion",
                    "args": {},
                    "description": "Execute calibrated Optical and SAR cross-sensor alignment and fusion"
                })
            else:
                # Disentangled sensor-appropriate local fixtures:
                # Optical: local_sample_cartosat_optical_bengaluru (optical raster)
                # SAR: local_sample_risat_sar_mumbai (SAR raster)
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_cartosat_optical_bengaluru" if not has_images else "workspace_img_1",
                        "aoi_bbox": bbox,
                        "collection": "cartosat-optical-l2" if not has_images else "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire optical composite for cross-sensor fusion"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_risat_sar_mumbai" if not has_images else "workspace_img_2",
                        "aoi_bbox": bbox,
                        "collection": "risat-c-band-sar" if not has_images else "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire SAR composite for cross-sensor fusion"
                })
                steps.append({
                    "tool": "optical_sar_fusion",
                    "args": {},
                    "description": "Execute calibrated Optical and SAR cross-sensor alignment and fusion"
                })
        elif task == "sentinel1_rtc_dprvi":
            if not has_images and not use_offline:
                s_date = parsed.get("start_date") or "2023-01-01"
                e_date = parsed.get("end_date") or "2024-01-01"
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-1-rtc",
                        "bbox": bbox,
                        "start_date": s_date,
                        "end_date": e_date,
                        "limit": 5
                    },
                    "description": "Discover Sentinel-1 RTC SAR scenes via Microsoft Planetary Computer"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire bounded Sentinel-1 RTC dual-pol composite and compute calibrated DpRVI"
                })
            else:
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_risat_sar_mumbai" if not has_images else "workspace_img_1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-1-rtc",
                        "bands": ["vv", "vh"],
                        "compute_dprvi": True
                    },
                    "description": "Acquire Sentinel-1 RTC dual-pol window and compute calibrated DpRVI"
                })
        elif task == "sentinel2_ndvi":
            if not has_images and not use_offline:
                s_date = parsed.get("start_date") or "2023-01-01"
                e_date = parsed.get("end_date") or "2024-01-01"
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-2-l2a",
                        "bbox": bbox,
                        "start_date": s_date,
                        "end_date": e_date,
                        "max_cloud_cover": cloud_thresh,
                        "limit": 5
                    },
                    "description": f"Discover Sentinel-2 L2A optical scenes via Element84 Earth Search (cloud<={cloud_thresh}%)"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire bounded Sentinel-2 L2A optical composite and compute calibrated NDVI"
                })
            else:
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "local_sample_cartosat_optical_bengaluru" if not has_images else "workspace_img_1",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b04", "b08", "scl"],
                        "include_scl": True,
                        "compute_ndvi": True
                    },
                    "description": "Acquire Sentinel-2 L2A window and compute calibrated NDVI with SCL"
                })
        elif task == "land_cover_classification":
            if not has_images and not use_offline:
                s_date = parsed.get("start_date") or "2023-01-01"
                e_date = parsed.get("end_date") or "2024-01-01"
                steps.append({
                    "tool": "stac_search",
                    "args": {
                        "collection": "sentinel-2-l2a",
                        "bbox": bbox,
                        "start_date": s_date,
                        "end_date": e_date,
                        "max_cloud_cover": cloud_thresh,
                        "limit": 3
                    },
                    "description": f"Discover Sentinel-2 L2A optical scenes for land cover classification (cloud<={cloud_thresh}%)"
                })
                steps.append({
                    "tool": "stac_composite_retrieve",
                    "args": {
                        "scene_id": "auto",
                        "aoi_bbox": bbox,
                        "collection": "sentinel-2-l2a",
                        "bands": ["b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"],
                        "include_scl": False,
                        "compute_ndvi": False
                    },
                    "description": "Acquire bounded Sentinel-2 10-band raster for BigEarthNet land cover classification"
                })
            steps.append({
                "tool": "land_cover_classification",
                "args": {
                    "images": [],
                    "query": parsed.get("raw_query", "Classify land cover")
                },
                "description": "Execute BigEarthNet ResNet-50 multi-label 19-class land-cover classification"
            })
        else:
            steps.append({
                "tool": "vqa_captioning",
                "args": {
                    "images": [],
                    "query": parsed.get("task", "Analyze scene")
                },
                "description": "Execute vision-language specialist analysis"
            })

        return steps


# Singleton instance
agent_orchestrator = MultiStepOrchestrator()
