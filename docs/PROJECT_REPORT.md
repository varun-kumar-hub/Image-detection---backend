# AI-Generated Image Detection

The classifier distinguishes **REAL IMAGE** from **AI-GENERATED IMAGE** with ImageNet-pretrained EfficientNet-B0. Training and evaluation use CIFAKE manifests generated from the dataset's supplied labels; no images are manually relabeled.

The official CIFAKE test set remains separate from training and validation. Evaluation saves accuracy, precision, recall, F1, ROC-AUC, confusion-matrix values, per-class accuracy, false-positive rate, and false-negative rate to `models/evaluation_results.json`.

Grad-CAM is computed from the actual loaded EfficientNet-B0 model and visualizes regions that contributed to the predicted class. It is an interpretability aid, not proof that an image is AI generated.

CIFAKE includes Stable Diffusion synthetic images at CIFAR-10-like resolution, so this model should not be represented as detecting every modern AI image generator.
