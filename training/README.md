# SatQuery AI — Remote Sensing Fine-Tuning & Adaptation Subsystem

This subsystem provides Google Colab-compatible training pipelines for fine-tuning Vision-Language Models (VLMs) and change detection models on remote sensing datasets (BigEarthNet, RSVQA, CDVQA).

---

## 1. Subsystem Architecture

```
training/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_remote_sensing_adaptation.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
├── scripts/
│   ├── train_adapter.py
│   ├── preprocess_bigearthnet.py
│   └── evaluate_rs_vlm.py
├── configs/
│   └── training_config.yaml
└── README.md
```

---

## 2. Adaptation Techniques

To ensure efficient training on Google Colab T4 / A100 GPUs without requiring hundreds of gigabytes of VRAM, SatQuery AI uses **LoRA (Low-Rank Adaptation)** and **Projection Adapter Tuning**:

- **Backbone Model:** Qwen2-VL / SigLIP / ResNet-50
- **Target Modules:** Vision-to-language projection layer (`multi_modal_projector`) and LoRA rank 16 on self-attention projection matrices (`q_proj`, `v_proj`).
- **Dataset Target:** BigEarthNet-MM (Multispectral & Sentinel-1 SAR pair rasters).

---

## 3. Quickstart: Google Colab Workflow

1. Open `notebooks/03_remote_sensing_adaptation.ipynb` in Google Colab.
2. Select GPU runtime (T4 / V100 / A100).
3. Execute setup cell:
   ```bash
   pip install torch torchvision transformers peft datasets rasterio
   ```
4. Run `python scripts/train_adapter.py --config configs/training_config.yaml`.
5. Saved checkpoints will be exported to `training/checkpoints/satquery_adapter_latest.pt`.
