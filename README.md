# AI-Generated Image Detection Backend

FastAPI backend and EfficientNet-B0 inference service for **REAL IMAGE** versus **AI-GENERATED IMAGE** classification. The upload, prediction, confidence, and Grad-CAM user flow is unchanged.

## CIFAKE pipeline

The training workflow uses the CIFAKE Real and AI-Generated Synthetic Images dataset and preserves its official test set. See `ml_pipeline/README.md` for the dataset layout, verification process, training command, evaluation command, exported model, and known limitations.

The app loads `models/ai_image_detector.keras` (or `MODEL_PATH`) with the same 224×224 RGB preprocessing used in training. `models/model_config.json` stores the explicit mapping: `REAL = 0`, `AI GENERATED = 1`.

## Run

```powershell
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
