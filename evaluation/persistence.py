import os
import json
import logging
from typing import List, Dict, Any, Optional
from app.evaluation.schemas import BenchmarkResultItem, EvaluationStatus

logger = logging.getLogger("satquery.evaluation.persistence")

RESULTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "benchmark_results.json")
)

def load_persisted_benchmark_results() -> List[BenchmarkResultItem]:
    """
    Loads verified benchmark results from persisted disk storage.
    If no evaluation has been run, returns empty list (never fabricates scores).
    """
    if not os.path.isfile(RESULTS_PATH):
        logger.info(f"No persisted benchmark results found at {RESULTS_PATH}.")
        return []

    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        results: List[BenchmarkResultItem] = []
        for item in raw_data:
            # Only include results with legitimate status
            if item.get("evaluation_status") == EvaluationStatus.COMPLETED:
                results.append(BenchmarkResultItem(**item))
        return results
    except Exception as e:
        logger.error(f"Error loading persisted benchmark results: {e}")
        return []

def persist_benchmark_result(result: BenchmarkResultItem) -> bool:
    """
    Persists an executed benchmark evaluation artifact to disk.
    """
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    existing_results = []
    if os.path.isfile(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    # Update or append
    updated = False
    for idx, r in enumerate(existing_results):
        if r.get("dataset") == result.dataset and r.get("task") == result.task:
            existing_results[idx] = result.model_dump()
            updated = True
            break
    if not updated:
        existing_results.append(result.model_dump())

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2)
    logger.info(f"Persisted verified evaluation result for {result.dataset} -> {result.metric}: {result.score}")
    return True
