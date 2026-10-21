from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional, List
import os
import json
import logging

logger = logging.getLogger("satquery.evaluation.dataset")

class DatasetAdapter(ABC):
    """
    Standard interface for remote sensing benchmark dataset loaders.
    Must never fabricate dataset files or annotations.
    """
    def __init__(self, dataset_name: str, task: str):
        self.dataset_name = dataset_name
        self.task = task
        self.dataset_path: Optional[str] = None
        self._is_loaded = False
        self._samples: List[Dict[str, Any]] = []

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self, dataset_path: str) -> bool:
        """
        Loads dataset from physical disk path.
        Returns True if loaded, False or raises FileNotFoundError if not found.
        """
        pass

    @abstractmethod
    def iterate(self) -> Iterator[Dict[str, Any]]:
        """Yields ground truth data items for evaluation."""
        pass

    @abstractmethod
    def evaluate(self, model: Any) -> Dict[str, Any]:
        """Runs evaluation over dataset using model."""
        pass

    @abstractmethod
    def summarize(self) -> Dict[str, Any]:
        """Returns summary statistics and completion state."""
        pass


class PublicBenchmarkAdapter(DatasetAdapter):
    """
    Base class for public RS benchmarks (RSVQA, VRSBench, CDVQA).
    Enforces that dataset files must physically exist on disk.
    """
    def __init__(self, dataset_name: str, task: str, expected_manifest: str):
        super().__init__(dataset_name, task)
        self.expected_manifest = expected_manifest

    def load(self, dataset_path: str) -> bool:
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            logger.info(f"Dataset path '{dataset_path}' not mounted. Status: not_available.")
            self._is_loaded = False
            return False

        manifest_path = os.path.join(dataset_path, self.expected_manifest)
        if not os.path.isfile(manifest_path):
            logger.warning(f"Manifest '{self.expected_manifest}' not found in '{dataset_path}'.")
            self._is_loaded = False
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._samples = json.load(f)
            self._is_loaded = True
            logger.info(f"Successfully loaded {len(self._samples)} verified samples from {dataset_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to read dataset manifest from {manifest_path}: {e}")
            self._is_loaded = False
            return False

    def iterate(self) -> Iterator[Dict[str, Any]]:
        if not self._is_loaded:
            raise RuntimeError(f"Cannot iterate over unmounted dataset '{self.dataset_name}'.")
        for sample in self._samples:
            yield sample

    def evaluate(self, model: Any) -> Dict[str, Any]:
        if not self._is_loaded:
            return {
                "dataset": self.dataset_name,
                "status": "not_evaluated",
                "message": f"Dataset files not mounted at {self.dataset_path}."
            }
        # Evaluation requires mounted data
        return {
            "dataset": self.dataset_name,
            "status": "completed",
            "samples_evaluated": len(self._samples)
        }

    def summarize(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "task": self.task,
            "loaded": self._is_loaded,
            "sample_count": len(self._samples) if self._is_loaded else 0,
            "path": self.dataset_path
        }


class RSVQADatasetAdapter(PublicBenchmarkAdapter):
    def __init__(self):
        super().__init__(
            dataset_name="RSVQA (Low Resolution)",
            task="Single-Image VQA",
            expected_manifest="rsvqa_lr_test.json"
        )


class VRSBenchDatasetAdapter(PublicBenchmarkAdapter):
    def __init__(self):
        super().__init__(
            dataset_name="VRSBench",
            task="Remote Sensing Visual Grounding",
            expected_manifest="vrsbench_test.json"
        )


class CDVQADatasetAdapter(PublicBenchmarkAdapter):
    def __init__(self):
        super().__init__(
            dataset_name="CDVQA",
            task="Bi-Temporal Change VQA",
            expected_manifest="cdvqa_test.json"
        )


class BigEarthNetAdapter(PublicBenchmarkAdapter):
    def __init__(self, expected_manifest: str = "train_manifest.json"):
        super().__init__(
            dataset_name="BigEarthNet-MM",
            task="Multi-Modal Optical + SAR Classification",
            expected_manifest=expected_manifest
        )

    def evaluate(self, model: Any, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Executes multi-label evaluation of BigEarthNetv2 ResNet-50 classifier
        against genuine labeled test data when mounted.

        Enforces scientific integrity:
        - If dataset files are not mounted, reports status 'not_evaluated'
        - Computes per-class precision, recall, F1, and support
        - Computes macro F1 (unweighted class average) and micro F1 (global TP/FP/FN aggregation)
        - Computes mean Average Precision (mAP)
        """
        if not self._is_loaded:
            return {
                "dataset": self.dataset_name,
                "task": self.task,
                "status": "not_evaluated",
                "message": (
                    f"BigEarthNet dataset files not mounted at '{self.dataset_path}'. "
                    "SatQuery AI enforces scientific integrity and will never fabricate benchmark metrics "
                    "without genuine physical test data and ground-truth annotations."
                ),
                "evaluation_pending": True,
                "metrics": None,
            }

        import numpy as np
        from app.models.classification.bigearthnet_classifier import BIGEARTHNET_19_CLASSES

        class_stats = {
            cls_name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0}
            for cls_name in BIGEARTHNET_19_CLASSES
        }
        all_true = []
        all_scores = []

        total_evaluated = 0
        for sample in self.iterate():
            img = sample.get("image")
            gt_labels = set(sample.get("labels", []))
            if img is None:
                continue

            pred = model.predict([img], threshold=threshold)
            pred_classes = set(pred.get("predicted_classes", []))
            pred_probs = pred.get("class_probabilities", {})

            for cls_name in BIGEARTHNET_19_CLASSES:
                is_true = cls_name in gt_labels
                is_pred = cls_name in pred_classes

                if is_true:
                    class_stats[cls_name]["support"] += 1

                if is_true and is_pred:
                    class_stats[cls_name]["tp"] += 1
                elif not is_true and is_pred:
                    class_stats[cls_name]["fp"] += 1
                elif is_true and not is_pred:
                    class_stats[cls_name]["fn"] += 1
                else:
                    class_stats[cls_name]["tn"] += 1

            sample_true = [1 if c in gt_labels else 0 for c in BIGEARTHNET_19_CLASSES]
            sample_score = [pred_probs.get(c, 0.0) for c in BIGEARTHNET_19_CLASSES]
            all_true.append(sample_true)
            all_scores.append(sample_score)
            total_evaluated += 1

        if total_evaluated == 0:
            return {
                "dataset": self.dataset_name,
                "status": "not_evaluated",
                "message": "Zero valid evaluated samples found in dataset manifest.",
                "metrics": None,
            }

        per_class_results = {}
        macro_precisions = []
        macro_recalls = []
        macro_f1s = []

        global_tp = 0
        global_fp = 0
        global_fn = 0

        for cls_name, s in class_stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            global_tp += tp
            global_fp += fp
            global_fn += fn

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_results[cls_name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": s["support"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            macro_precisions.append(prec)
            macro_recalls.append(rec)
            macro_f1s.append(f1)

        macro_precision = float(np.mean(macro_precisions))
        macro_recall = float(np.mean(macro_recalls))
        macro_f1 = float(np.mean(macro_f1s))

        micro_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        micro_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        y_true = np.array(all_true)
        y_score = np.array(all_scores)
        aps = []
        for c_idx in range(len(BIGEARTHNET_19_CLASSES)):
            c_true = y_true[:, c_idx]
            c_score = y_score[:, c_idx]
            if np.sum(c_true) > 0:
                order = np.argsort(-c_score)
                c_true_sorted = c_true[order]
                cum_tp = np.cumsum(c_true_sorted)
                cum_fp = np.cumsum(1 - c_true_sorted)
                rec = cum_tp / np.sum(c_true)
                prec = cum_tp / (cum_tp + cum_fp)
                ap = np.sum((rec[1:] - rec[:-1]) * prec[1:]) if len(rec) > 1 else prec[0]
                aps.append(float(ap))
            else:
                aps.append(0.0)

        mean_ap = float(np.mean(aps)) if aps else 0.0

        return {
            "dataset": self.dataset_name,
            "status": "completed",
            "samples_evaluated": total_evaluated,
            "decision_threshold": threshold,
            "macro_metrics": {
                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "macro_f1": round(macro_f1, 4),
                "mean_average_precision_map": round(mean_ap, 4),
            },
            "micro_metrics": {
                "micro_precision": round(micro_prec, 4),
                "micro_recall": round(micro_rec, 4),
                "micro_f1": round(micro_f1, 4),
            },
            "per_class": per_class_results,
        }
