import os
from fastapi import APIRouter
from app.schemas.analysis import HealthResponseSchema, ModelHealthItem
from app.agent.registry import model_registry

router = APIRouter()

@router.get("/health", response_model=HealthResponseSchema)
def get_health():
    try:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device_str = "cpu"

    models_health_list = []
    loaded_names = []

    for task, model in model_registry._models.items():
        st = model.status
        models_health_list.append(ModelHealthItem(
            name=model.model_name,
            status=st,
            task=task
        ))
        if st == "loaded":
            loaded_names.append(model.model_name)

    return HealthResponseSchema(
        status="ok",
        version="1.1.0",
        models_loaded=loaded_names,
        models=models_health_list,
        device=device_str
    )

@router.get("/models")
def get_models_detailed():
    """Returns honest, detailed information for all registered models and baselines."""
    return {"models": model_registry.list_models_detailed()}
