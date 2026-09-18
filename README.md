# AI-Generated Image Detection Backend

FastAPI backend and EfficientNet-B0 inference service for **REAL IMAGE** versus **AI-GENERATED IMAGE** classification. The upload, prediction, confidence, and Grad-CAM user flow is unchanged.

## CIFAKE pipeline

The training workflow uses the CIFAKE Real and AI-Generated Synthetic Images dataset and preserves its official test set. See `ml_pipeline/README.md` for the dataset layout, verification process, training command, evaluation command, exported model, and known limitations.

The app loads `models/custom_fused_parameter_model.pth` (or `MODEL_PATH`) and uses a 7-channel `32 × 32` PyTorch tensor: RGB (3), Laplacian texture (3), and FFT magnitude (1). Class mapping is `AI-GENERATED = 0`, `REAL = 1`.

## Run

```powershell
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
