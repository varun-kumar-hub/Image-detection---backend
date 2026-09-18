"""Grad-CAM for the actual EfficientNet-B0 binary classifier output."""
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image


def get_gradcam_heatmap(model: tf.keras.Model, image_tensor: np.ndarray, last_conv_layer_name: Optional[str] = None, pred_index: Optional[int] = None) -> np.ndarray:
    """Return a genuine Grad-CAM map for REAL (0) or AI GENERATED (1)."""
    backbone = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()), None)
    if backbone is None:
        raise ValueError("Grad-CAM requires the EfficientNet-B0 backbone in the loaded model.")
    if last_conv_layer_name is None:
        last_conv_layer_name = next(layer.name for layer in reversed(backbone.layers) if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)))
    conv_layer = backbone.get_layer(last_conv_layer_name)
    backbone_to_conv = tf.keras.Model(backbone.inputs, [conv_layer.output, backbone.output])
    input_layer_name = model.inputs[0].name.split(":")[0]
    head_layers = [layer for layer in model.layers if layer.name not in {input_layer_name, backbone.name}]
    inputs = tf.cast(image_tensor, tf.float32)
    with tf.GradientTape() as tape:
        conv_output, features = backbone_to_conv(inputs, training=False)
        output = features
        for layer in head_layers:
            output = layer(output, training=False)
        ai_probability = output[:, 0]
        target = pred_index if pred_index is not None else tf.cast(ai_probability >= 0.5, tf.int32)
        class_score = tf.where(tf.cast(target, tf.bool), ai_probability, 1.0 - ai_probability)
    gradients = tape.gradient(class_score, conv_output)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be computed for this model.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image_tensor.shape[1], image_tensor.shape[2])).numpy().squeeze()
    return heatmap.astype(np.float32)


def generate_gradcam_overlay(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    original = np.asarray(original_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    color = np.zeros_like(original)
    color[..., 0], color[..., 1], color[..., 2] = heatmap, heatmap * 0.4, heatmap * 0.1
    return Image.fromarray(np.clip((original * (1 - alpha * heatmap[..., None]) + color * alpha) * 255, 0, 255).astype(np.uint8))
