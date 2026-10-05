from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np

class BaseModel(ABC):
    """
    Abstract base adapter class for remote sensing models and baselines.
    Enforces scientific integrity: models must declare their actual implementation
    status (baseline, pretrained_model, production_model, demo, unavailable)
    and must never claim to be trained or loaded without authentic weights.
    """
    def __init__(
        self, 
        model_name: str, 
        task_type: str,
        implementation_status: str = "baseline",
        is_trained: bool = False,
        is_remote_sensing_adapted: bool = False,
        license: str = "Apache-2.0",
        version: str = "1.0.0"
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.implementation_status = implementation_status
        self.is_trained = is_trained
        self.is_remote_sensing_adapted = is_remote_sensing_adapted
        self.license = license
        self.version = version
        self._is_loaded = False
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def status(self) -> str:
        if self._is_loaded:
            return "loaded"
        if self.implementation_status in ("baseline", "demo", "unavailable"):
            return self.implementation_status
        return "baseline"

    def load(self) -> None:
        """
        Loads actual neural network weights or resource artifacts into memory/device.
        Must only set _is_loaded = True if actual model resources are ready.
        Baselines, demo API orchestrators, and unmounted models never set _is_loaded = True.
        """
        if self.implementation_status in ("baseline", "demo", "unavailable"):
            self._is_loaded = False
            return
        
        # Subclasses for real local models override this method to load weights
        self._is_loaded = True

    @abstractmethod
    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute prediction on given image array(s) and query.
        Returns a dict matching specialist output contracts with implementation_status.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "task": self.task_type,
            "implementation_status": self.implementation_status,
            "status": self.status,
            "is_trained": self.is_trained,
            "is_remote_sensing_adapted": self.is_remote_sensing_adapted,
            "license": self.license,
            "version": self.version
        }
