# Image Detection Backend (v4.0)

A high-performance, production-ready FastAPI backend and deep learning inference engine for real-time image authenticity classification, Grad-CAM attention visualization, supporting signal analysis, and authenticated PDF report generation.

---

## 🚀 Key Features

- **Deep Learning Classification**:
  - Fine-tuned **EfficientNet-B0** architecture trained on balanced Real vs. AI-generated image datasets.
  - Generates exact probabilistic scores for `ai_probability`, `real_probability`, and calibrated `confidence`.
- **Grad-CAM Model Focus**:
  - Computes Gradient-weighted Class Activation Mapping directly from the final convolutional feature maps.
  - Returns bounding boxes and spatial heatmaps showing which regions influenced the classification.
- **Structured Explanations**:
  - Clean, professional explanations answering *"Why was this image classified this way?"*.
  - Breaks down model basis, primary factors, supporting signal indicators, and limitations.
  - **Zero forensic terminology**; 100% focused on objective technical analysis.
- **Hidden Evaluation Mode**:
  - Independent verification testing.
  - Accepts optional `ground_truth` parameter.
  - **Strict Model Isolation**: The model predicts 100% independently prior to inspecting ground truth. The backend calculates `is_correct` post-prediction without biasing the model.
- **Supabase Authentication & Private Storage**:
  - Validates Google OAuth JWT tokens via Supabase Auth API (`/auth/v1/user`).
  - Stores uploaded assets in private buckets under `{user_id}/uploads/{analysis_id}.{ext}`.
  - Issues time-limited signed URLs (1-hour validity) for secure image viewing.
  - High-resilience persistence: Supabase PostgREST sync with local SQLite backup.
- **Automated PDF Reports**:
  - Generates downloadable Detection Summary certificates via ReportLab.
  - Includes model metrics, visual breakdown, evaluation benchmark (if tested), and disclaimer.

---

## 🛠 Tech Stack

- **Framework**: FastAPI (Python 3.10+)
- **ASGI Server**: Uvicorn
- **Deep Learning**: PyTorch, Torchvision, TIMM, Keras / TensorFlow
- **Image Processing**: Pillow, NumPy, SciPy, Matplotlib
- **Document Generation**: ReportLab
- **Cloud & Auth**: Supabase (Auth, PostgreSQL, Private Storage)
- **Validation**: Pydantic v2 & Pydantic-Settings
- **Testing**: Pytest & Pytest-AsyncIO

---

## 📁 Directory Structure

```
Image-Detection-Backend/
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes/          # health, upload, analysis, history, reports, storage
│       ├── core/                # config.py (Pydantic settings), security.py (token verification)
│       ├── schemas/             # Pydantic request & response models
│       ├── services/            # predictor, explanation, image_analysis, supabase, report
│       └── main.py              # FastAPI app initialization, CORS, lifespan
├── ml_pipeline/
│   ├── data_loader.py           # Balanced dataset loaders & augmentations
│   ├── manipulation_detector.py # ELA compression, noise variance, metadata analysis
│   ├── model_builder.py         # EfficientNet-B0 architecture & transfer learning
│   ├── predictor.py             # Inference pipeline & Grad-CAM visualizer
│   ├── trainer.py               # Model training loop & checkpointing
│   └── utils.py                 # File & image helper utilities
├── models/
│   ├── image_detection_v1.keras # Trained model weights
│   ├── model_config.json        # Hyperparameters & normalization thresholds
│   └── evaluation_results.json  # Benchmark metrics
├── tests/                       # Complete Pytest integration test suite
├── .env.example                 # Template for environment variables
├── pytest.ini                   # Pytest test runner settings
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # Project documentation & deployment guides
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Set your configuration:

```env
# Application
APP_NAME=Image Detection
APP_VERSION=4.0.0
API_PREFIX=/api
DEBUG=False

# Supabase Credentials
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# Storage Bucket
SUPABASE_STORAGE_BUCKET=imageguard
SIGNED_URL_EXPIRES_IN=3600

# Model Path
MODEL_PATH=models/image_detection_v1.keras
MODEL_VERSION=v1.0

# CORS & Upload Limits
MAX_UPLOAD_SIZE_MB=25
CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]
```

---

## 💻 Local Development

### 1. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Development Server
```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

---

## 🧪 Running Tests

Execute the full automated test suite (including model inference, authentication, storage lifecycle, and schema validation):

```bash
python -m pytest tests/ -v
```

---

## 🌐 Deployment Guide

### Deploying to Render
1. Connect this repository to [Render](https://render.com).
2. Create a new **Web Service**.
3. Set Environment: **Python 3**.
4. Set Build Command:
   ```bash
   pip install -r requirements.txt
   ```
5. Set Start Command:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
   ```
6. Add Environment Variables from `.env` in the Render dashboard.

### Deploying to Railway
1. Create a new project in [Railway](https://railway.app).
2. Select **Deploy from GitHub repo**.
3. Railway automatically detects Python and `requirements.txt`.
4. Add environment variables.
5. Railway assigns a public HTTPS domain.

---

## 📄 License
MIT License. Built for production-grade deep learning classification systems.
