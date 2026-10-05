from fastapi import APIRouter
from app.agent.registry import model_registry

router = APIRouter()

@router.get("/models")
def list_models():
    """
    Returns registered model adapters and baselines with authentic implementation statuses.
    """
    return {
        "models": model_registry.list_models_detailed(),
        "registered_models": model_registry.list_models(),
        "status": "active"
    }
