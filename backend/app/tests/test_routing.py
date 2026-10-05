import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.agent.router import TaskRouter

def test_single_image_captioning_routing():
    task = TaskRouter.classify_task("Describe this satellite scene and terrain.", image_count=1, modalities=["OPTICAL"])
    assert task == "captioning"

def test_single_image_grounding_routing():
    task = TaskRouter.classify_task("Highlight the water body and shoreline.", image_count=1, modalities=["OPTICAL"])
    assert task == "grounding"

def test_single_image_vqa_routing():
    task = TaskRouter.classify_task("What type of land cover dominates this image?", image_count=1, modalities=["OPTICAL"])
    assert task == "vqa"

def test_bitemporal_change_routing():
    task = TaskRouter.classify_task("What changed between these two dates?", image_count=2, modalities=["OPTICAL", "OPTICAL"])
    assert task == "change_vqa"

def test_optical_sar_routing():
    task = TaskRouter.classify_task(
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        image_count=2,
        modalities=["OPTICAL", "SAR"]
    )
    assert task == "optical_sar"
