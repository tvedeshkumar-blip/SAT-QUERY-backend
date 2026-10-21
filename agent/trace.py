from datetime import datetime
import time
from typing import List, Dict, Any

class ExecutionTraceTracker:
    def __init__(self, task: str):
        self.task = task
        self.models_selected: List[str] = []
        self.steps: List[Dict[str, Any]] = []
        self.parameters: Dict[str, Any] = {}
        self.start_time = time.time()

    def add_step(self, step: str, detail: str, status: str = "completed"):
        self.steps.append({
            "step": step,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detail": detail,
            "status": status
        })

    def add_model(self, model_name: str):
        if model_name not in self.models_selected:
            self.models_selected.append(model_name)

    def set_parameters(self, params: Dict[str, Any]):
        self.parameters.update(params)

    def to_dict(self) -> Dict[str, Any]:
        execution_time_ms = round((time.time() - self.start_time) * 1000, 2)
        return {
            "task": self.task,
            "models_selected": self.models_selected,
            "steps": self.steps,
            "parameters": self.parameters,
            "execution_time_ms": execution_time_ms
        }
