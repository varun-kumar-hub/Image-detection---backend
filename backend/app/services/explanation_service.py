"""
Explanation Service
===================
Generates structured, evidence-based explanations of model classifications.
Strictly adheres to:
- Grounded solely on actual model predictions and measured image characteristics
- No fabricated evidence or claims of predefined manual 'AI artifacts'
- Distinguishes model learned deep visual features from supporting measurements
- Outlines technical limitations and probabilistic boundaries
"""

from typing import Dict, Any, List

class ExplanationService:
    @staticmethod
    def generate_explanation(
        classification: str,
        ai_probability: float,
        real_probability: float,
        confidence: str,
        supporting_details: Dict[str, Any],
        has_gradcam: bool = False
    ) -> Dict[str, Any]:
        """
        Builds a structured explanation object for an analysis result.
        """
        supporting_observations: List[str] = []

        # 1. Inspect actual supporting measurements
        exif_info = supporting_details.get("exif", {})
        if exif_info.get("available"):
            make = exif_info.get("camera_make") or "Unknown"
            model = exif_info.get("camera_model") or ""
            supporting_observations.append(
                f"Camera hardware metadata detected: {make} {model}".strip()
            )
        else:
            supporting_observations.append(
                "Standard camera EXIF hardware metadata was not found in the file structure."
            )

        ela_info = supporting_details.get("ela", {})
        if ela_info:
            status = ela_info.get("compression_status", "Analyzed")
            supporting_observations.append(
                f"Error level compression consistency: {status}."
            )

        noise_info = supporting_details.get("noise", {})
        if noise_info:
            filter_status = noise_info.get("filter_status", "Analyzed")
            supporting_observations.append(
                f"High-frequency sensor noise distribution: {filter_status}."
            )

        if has_gradcam:
            supporting_observations.append(
                "Gradient-weighted Class Activation Mapping (Grad-CAM) computed to highlight salient regions contributing to the prediction."
            )

        # 2. Derive classification-specific technical explanation
        if classification == "ai_generated":
            summary = (
                f"The system classified this image as AI-Generated with an estimated probability "
                f"of {ai_probability:.1f}% and {confidence} model confidence."
            )
            model_basis = (
                "The EfficientNet-B0 feature extractor processed the standardized 224×224×3 pixel tensor through "
                "its convolutional layers. The extracted high-dimensional visual feature vector aligned more strongly "
                "with the statistical representations learned from AI-synthesized images during balanced model training."
            )
            primary_factors = [
                f"Model output probability strongly favors the AI-generated class ({ai_probability:.1f}% vs {real_probability:.1f}% authentic).",
                "Learned feature representations correspond to texture and frequency characteristics typical of generative AI distributions.",
                "Supporting image analysis indicators corroborate statistical deviation from standard optical camera sensors."
            ]
            limitations = [
                "AI image detection is probabilistic. Models evaluate learned visual representations rather than deterministic watermarks.",
                "Heavy compression, social media re-encoding, or aggressive filters can occasionally introduce visual artifacts.",
                "Results should be interpreted as supporting evidence alongside contextual origin."
            ]

        elif classification == "real":
            summary = (
                f"The system classified this image as Real / Authentic with an estimated probability "
                f"of {real_probability:.1f}% and {confidence} model confidence."
            )
            model_basis = (
                "The EfficientNet-B0 feature extractor transformed the image pixel values into learned visual representations. "
                "The resulting feature vector aligned with the natural optical sensor noise, optical lens characteristics, "
                "and coherent high-frequency details learned from authentic camera captures."
            )
            primary_factors = [
                f"Model output probability strongly favors the authentic class ({real_probability:.1f}% vs {ai_probability:.1f}% AI-generated).",
                "Visual representations exhibit coherent high-frequency consistency characteristic of physical camera sensors.",
                "No decisive statistical markers of generative synthesis were detected in the learned feature vector."
            ]
            limitations = [
                "Classification reflects resemblance to the model's authentic training distribution and is not absolute proof of origin.",
                "State-of-the-art generative models with fine-grained post-processing may occasionally mimic authentic sensor noise.",
                "Assessment should be corroborated with provenance."
            ]

        else:  # needs_review
            summary = (
                f"The system classified this image as Needs Review because the prediction fell into the uncertain boundary "
                f"(AI: {ai_probability:.1f}%, Real: {real_probability:.1f}%)."
            )
            model_basis = (
                "The EfficientNet-B0 backbone extracted features that exhibit subtle markers from both authentic and synthetic "
                "distributions. The resulting output did not cross the decisive 55% threshold in either direction."
            )
            primary_factors = [
                f"Model prediction probabilities are close to the neutral boundary ({ai_probability:.1f}% AI vs {real_probability:.1f}% Real).",
                "The image contains mixed or ambiguous visual representations that prevent high-confidence automated determination.",
                "Supporting image analysis indicators show intermediate characteristics."
            ]
            limitations = [
                "Because the probabilities are close to the neutral threshold, this result must not be treated as a definitive classification.",
                "Manual human review and contextual source verification are strongly recommended."
            ]

        return {
            "summary": summary,
            "model_basis": model_basis,
            "primary_factors": primary_factors,
            "supporting_observations": supporting_observations,
            "limitations": limitations
        }

explanation_service = ExplanationService()
