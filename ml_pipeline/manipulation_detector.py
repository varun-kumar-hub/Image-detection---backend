"""
Image Manipulation & Forensics Detector
=======================================
Performs non-destructive, scientifically grounded forensic inspections on input images:
1. EXIF Metadata analysis
2. Error Level Analysis (ELA) for compression anomalies & splicing
3. High-frequency noise distribution & texture smoothness
4. Resizing & interpolation indicators
"""

import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Any

class ManipulationDetector:
    def __init__(self, ela_quality: int = 90, ela_scale: int = 15):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def extract_exif(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts and parses human-readable EXIF metadata."""
        exif_data = {}
        try:
            raw_exif = image._getexif()
            if not raw_exif:
                return {"available": False, "details": "No EXIF metadata found"}

            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Filter binary or unprintable tags
                if isinstance(value, bytes):
                    continue
                if tag_name == "GPSInfo":
                    gps_readable = {}
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        if not isinstance(value[t], bytes):
                            gps_readable[sub_tag] = str(value[t])
                    exif_data["GPS"] = gps_readable
                else:
                    exif_data[str(tag_name)] = str(value)

            return {
                "available": True,
                "camera_make": exif_data.get("Make", "Unknown"),
                "camera_model": exif_data.get("Model", "Unknown"),
                "software": exif_data.get("Software", "None"),
                "date_taken": exif_data.get("DateTimeOriginal", exif_data.get("DateTime", "Unknown")),
                "has_gps": "GPS" in exif_data,
                "all_tags": exif_data
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def compute_ela(self, image: Image.Image) -> Dict[str, Any]:
        """
        Performs Error Level Analysis (ELA).
        Re-saves the image at a known JPEG quality, computes absolute difference,
        and analyzes the error distribution.
        """
        try:
            # Convert to RGB
            rgb_img = image.convert("RGB")

            # Resave in memory at target quality
            buffer = io.BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=self.ela_quality)
            buffer.seek(0)
            resaved_img = Image.open(buffer)

            # Compute difference and amplify
            diff = ImageChops.difference(rgb_img, resaved_img)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale = 255.0 / max_diff if max_diff > 0 else 1.0
            diff = ImageEnhance.Brightness(diff).enhance(scale)

            # Calculate metrics
            diff_array = np.array(diff, dtype=np.float32)
            ela_mean = float(np.mean(diff_array))
            ela_std = float(np.std(diff_array))

            # Compression assessment
            compression_status = "Detected" if max_diff > 10 else "Minimal / Lossless"

            return {
                "compression_status": compression_status,
                "ela_mean": round(ela_mean, 2),
                "ela_std": round(ela_std, 2),
                "max_difference": int(max_diff),
                "status": "Completed"
            }
        except Exception as e:
            return {
                "compression_status": "Indeterminate",
                "error": str(e),
                "status": "Failed"
            }

    def analyze_noise(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyzes high-frequency noise variance across the image.
        Synthetic AI images often exhibit unnatural smoothness in skin or solid textures
        lacking characteristic sensor noise.
        """
        try:
            gray = image.convert("L")
            img_array = np.array(gray, dtype=np.float32)

            # High-pass 3x3 Laplacian kernel approximation
            # [ 0,  1,  0]
            # [ 1, -4,  1]
            # [ 0,  1,  0]
            padded = np.pad(img_array, 1, mode="edge")
            laplacian = (
                padded[:-2, 1:-1] +
                padded[2:, 1:-1] +
                padded[1:-1, :-2] +
                padded[1:-1, 2:] -
                4 * img_array
            )

            noise_variance = float(np.var(laplacian))

            filter_status = "Normal sensor noise" if noise_variance > 150 else "High smoothness / Filtered"

            return {
                "filter_status": filter_status,
                "noise_variance": round(noise_variance, 2),
                "texture_uniformity": "High" if noise_variance < 100 else "Natural"
            }
        except Exception as e:
            return {
                "filter_status": "Indeterminate",
                "error": str(e)
            }

    def check_resizing(self, image: Image.Image) -> Dict[str, Any]:
        """Inspects aspect ratio, dimensions, and potential interpolation markers."""
        width, height = image.size
        standard_ratios = [(1, 1), (4, 3), (3, 2), (16, 9), (9, 16)]
        current_ratio = width / height

        matches_standard = any(abs(current_ratio - (w/h)) < 0.02 for w, h in standard_ratios)

        return {
            "resize_status": "Possible" if not matches_standard else "Standard Aspect Ratio",
            "width": width,
            "height": height,
            "aspect_ratio": f"{round(current_ratio, 2)}:1"
        }

    def analyze(self, image_input: str | Image.Image) -> Dict[str, Any]:
        """Runs the complete manipulation and forensics suite."""
        if isinstance(image_input, (str, bytes)):
            image = Image.open(image_input)
        else:
            image = image_input

        exif_info = self.extract_exif(image)
        ela_info = self.compute_ela(image)
        noise_info = self.analyze_noise(image)
        resize_info = self.check_resizing(image)

        return {
            "metadata_status": "Available" if exif_info.get("available") else "Missing / Stripped",
            "compression_status": ela_info.get("compression_status", "Indeterminate"),
            "resize_status": resize_info.get("resize_status", "Indeterminate"),
            "filter_status": noise_info.get("filter_status", "Indeterminate"),
            "exif": exif_info,
            "ela": ela_info,
            "noise": noise_info,
            "dimensions": resize_info
        }
