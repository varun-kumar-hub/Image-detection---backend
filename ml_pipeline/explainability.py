"""
Model Explainability Module (Grad-CAM)
======================================
Produces Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
for EfficientNet-B0 predictions.

Guiding Principles:
- The heatmap indicates regions that contributed most strongly to the model's prediction.
- Heatmaps provide interpretability and supporting visual evidence, NOT definitive forensic proof.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import io
from pathlib import Path
from typing import Tuple, Optional

def get_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for a given input image tensor.

    Args:
        model: Trained Keras model (ImageGuard EfficientNet)
        image_tensor: Normalized/preprocessed input tensor with shape (1, 224, 224, 3)
        last_conv_layer_name: Name of target convolutional layer (defaults to last conv in backbone)
        pred_index: Target class index (0 for Real, 1 for AI-Generated). None uses top prediction.

    Returns:
        2D numpy array heatmap of shape (224, 224) with values in [0, 1].
    """
    # Locate backbone and target conv layer
    # For EfficientNetB0 in Keras, the backbone is usually the first major sub-model or layer
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is not None and last_conv_layer_name is None:
        # Find last conv2d in backbone
        for layer in reversed(backbone.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or "conv" in layer.name:
                last_conv_layer_name = layer.name
                break

    try:
        if backbone is not None and last_conv_layer_name is not None:
            # Build grad model extracting feature map from backbone conv layer and classification output
            conv_layer = backbone.get_layer(last_conv_layer_name)
            
            # Create sub-model mapping backbone inputs to (conv_output, backbone_output)
            backbone_submodel = tf.keras.Model(
                inputs=backbone.inputs,
                outputs=[conv_layer.output, backbone.output]
            )

            # Reconstruct forward pass for gradient computation
            with tf.GradientTape() as tape:
                inputs = tf.cast(image_tensor, tf.float32)
                conv_outputs, backbone_features = backbone_submodel(inputs)
                
                # Feed backbone_features through classification head
                x = backbone_features
                for layer in model.layers:
                    if layer.name in [backbone.name, "image_input"]:
                        continue
                    x = layer(x)
                
                prediction = x
                if pred_index is None:
                    loss = prediction[:, 0]
                else:
                    loss = prediction[:, pred_index]

            # Gradient of output with respect to feature maps
            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                return np.zeros((224, 224), dtype=np.float32)

            # Global average pooling of gradients to obtain importance weights
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight convolutional feature maps
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # ReLU on heatmap to keep only positive contributions
            heatmap = tf.maximum(heatmap, 0.0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            
            heatmap_np = heatmap.numpy()

            # Resize heatmap to target 224x224
            img_heatmap = Image.fromarray(np.uint8(255 * heatmap_np))
            img_heatmap = img_heatmap.resize((image_tensor.shape[2], image_tensor.shape[1]), Image.Resampling.BILINEAR)
            return np.array(img_heatmap, dtype=np.float32) / 255.0

    except Exception as e:
        print(f"[Grad-CAM] Error generating heatmap: {e}")
        return np.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=np.float32)

    return np.zeros((224, 224), dtype=np.float32)

def generate_gradcam_overlay(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> Image.Image:
    """
    Overlays Grad-CAM heatmap onto the original image using a neutral grayscale/amber colormap.
    Avoids garish rainbow / neon styling.
    """
    resized_orig = original_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    orig_np = np.array(resized_orig, dtype=np.float32) / 255.0

    # Minimal, technical monochrome/subtle warm map (Linear/Apple style)
    # Highlight high-activation regions with subtle contrast enhancement
    heatmap_colored = np.zeros_like(orig_np)
    heatmap_colored[..., 0] = heatmap * 1.0  # Red/amber emphasis
    heatmap_colored[..., 1] = heatmap * 0.4
    heatmap_colored[..., 2] = heatmap * 0.1

    overlay = (orig_np * (1.0 - alpha * heatmap[..., np.newaxis])) + (heatmap_colored * alpha)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
