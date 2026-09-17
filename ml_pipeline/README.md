# ImageGuard ML Pipeline Architecture & Preprocessing Conventions

## 1. Dataset Balancing Strategy
- **Raw Distribution**: 8,803 Real vs 48,786 Fake (~15.3% Real vs 84.7% Fake).
- **Class Balancing**: Formed a balanced subset of 8,803 Real and 8,803 Fake images (total 17,606 images) via deterministic sampling with `seed=42`.
- **Stratified Partition**:
  - **70% Training**: 6,162 Real, 6,162 Fake (12,324 images)
  - **15% Validation**: 1,320 Real, 1,320 Fake (2,640 images)
  - **15% Test**: 1,321 Real, 1,321 Fake (2,642 images)
- **Leakage Prevention**: Isolated directories in `data/processed/` verify zero overlap across train, val, and test splits. Original images in `data/real/` and `data/fake/` remain untouched.

## 2. EfficientNet-B0 Preprocessing & Normalization Verification
- **Backbone**: `tf.keras.applications.EfficientNetB0` (ImageNet pretrained weights).
- **Internal Layers**: Inspection confirms `EfficientNetB0` includes an internal `Rescaling(scale=1./255.0, offset=0.0)` layer and a `Normalization` layer at index 1 and 2.
- **Critical Normalization Rule**:
  - Input images **MUST be passed in the `[0, 255]` pixel range** as `float32`.
  - Applying manual `rescaling(x) / 255.0` prior to the model causes **accidental double-normalization** (scaling down to `[0, 0.0039]`), severely corrupting feature representations.
  - Training and inference pipelines use identical preprocessing: convert to RGB, bilinear resize to (224, 224), and provide raw `[0, 255]` float32 tensors.

## 3. Architecture Specification
```
Input Image (224 × 224 × 3, RGB, [0, 255])
       ↓
tf.keras.applications.EfficientNetB0 (ImageNet Pretrained, Feature Extractor)
       ↓
GlobalAveragePooling2D
       ↓
BatchNormalization (head_bn1)
       ↓
Dense(256, activation='relu', name='dense_256')
       ↓
Dropout(0.5, name='dropout_1')
       ↓
Dense(128, activation='relu', name='dense_128')
       ↓
Dropout(0.3, name='dropout_2')
       ↓
Dense(1, activation='sigmoid', name='ai_probability_output')
```

## 4. Decision Thresholds & Interpretation
- **AI Probability > 55%**: `AI-GENERATED`
- **AI Probability < 45%**: `AUTHENTIC`
- **45% ≤ AI Probability ≤ 55%**: `NEEDS REVIEW` (borderline prediction, requires manual inspection)
- **Confidence Levels**:
  - `High`: dominant probability ≥ 90%
  - `Medium`: dominant probability ≥ 70%
  - `Low`: dominant probability < 70%
- **Terminology**: Labeled as "Model Confidence" (rather than "Calibrated Confidence") until explicit empirical Platt scaling or isotonic calibration is performed.

## 5. Model Explainability
- Grad-CAM heatmap generation highlights pixel regions with highest activation gradients towards the prediction.
- Heatmaps are explicitly presented as *supporting explanatory signals* rather than proof of forgery.
