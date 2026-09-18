"""
Report Service
==============
Generates publication-quality PDF analysis reports using ReportLab.
Branded strictly as 'Image Detection' with zero forensic terminology.
"""

import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ReportService:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4
        )
        self.subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12
        )
        self.section_heading = ParagraphStyle(
            "SectionHeading",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=10,
            spaceAfter=6
        )
        self.body_style = ParagraphStyle(
            "ReportBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        self.badge_style = ParagraphStyle(
            "BadgeStyle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            alignment=1  # Centered
        )
        self.disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748B")
        )

    def generate_pdf_report(self, analysis_data: Dict[str, Any]) -> bytes:
        """
        Builds and returns the binary PDF report for a given analysis record.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []

        # 1. Header
        story.append(Paragraph("IMAGE DETECTION REPORT", self.title_style))
        story.append(Paragraph("Real vs AI-Generated Image Detection Summary", self.subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=14))

        # Metadata Header Table
        meta_table_data = [
            [
                Paragraph(f"<b>Analysis ID:</b> {analysis_data.get('id', 'N/A')}", self.body_style),
                Paragraph(f"<b>Date:</b> {analysis_data.get('created_at', datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))}", self.body_style)
            ],
            [
                Paragraph(f"<b>Model:</b> {analysis_data.get('model_name', 'EfficientNet-B0')} ({analysis_data.get('model_version', 'v1.0')})", self.body_style),
                Paragraph(f"<b>Processing Time:</b> {analysis_data.get('processing_time_ms', 0)} ms", self.body_style)
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 14))

        # 2. Executive Result Card
        raw_classification = analysis_data.get("classification", "UNKNOWN").upper().replace("_", " ")
        banner_bg = colors.HexColor("#F8FAFC")
        banner_border = colors.HexColor("#CBD5E1")
        text_color = colors.HexColor("#0F172A")

        verdict_para = Paragraph(
            f"<font color='{text_color.hexval()}'><b>DETECTION RESULT: {raw_classification}</b></font>",
            self.badge_style
        )
        prob_summary = Paragraph(
            f"<b>AI Probability:</b> {analysis_data.get('ai_probability', 0.0):.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Real Probability:</b> {analysis_data.get('real_probability', 0.0):.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Model Confidence:</b> {analysis_data.get('confidence', 'N/A').upper()}",
            ParagraphStyle("CenterProb", parent=self.body_style, alignment=1)
        )

        verdict_table = Table([[verdict_para], [prob_summary]], colWidths=[540])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
            ("BOX", (0, 0), (-1, -1), 1, banner_border),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 14))

        # 3. Evaluation Reference (if provided)
        if analysis_data.get("is_evaluation") or analysis_data.get("ground_truth"):
            gt = analysis_data.get("ground_truth", "").upper()
            is_corr = analysis_data.get("is_correct")
            status_text = "CORRECT" if is_corr else "INCORRECT" if is_corr is False else "NOT EVALUATED"
            story.append(Paragraph("EVALUATION REFERENCE (CONTROLLED TESTING)", self.section_heading))
            eval_data = [
                [Paragraph("<b>Known Ground Truth</b>", self.body_style), Paragraph(gt or "N/A", self.body_style)],
                [Paragraph("<b>Model Independent Prediction</b>", self.body_style), Paragraph(raw_classification, self.body_style)],
                [Paragraph("<b>Evaluation Result</b>", self.body_style), Paragraph(f"<b>{status_text}</b>", self.body_style)]
            ]
            eval_table = Table(eval_data, colWidths=[200, 340])
            eval_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(eval_table)
            story.append(Spacer(1, 14))

        # 4. Image Information Section
        story.append(Paragraph("IMAGE INFORMATION", self.section_heading))
        img_info = analysis_data.get("image_info", {})
        img_props = [
            [Paragraph("<b>Filename</b>", self.body_style), Paragraph(str(img_info.get("filename", "N/A")), self.body_style)],
            [Paragraph("<b>Dimensions</b>", self.body_style), Paragraph(str(img_info.get("dimensions", "N/A")), self.body_style)],
            [Paragraph("<b>File Size</b>", self.body_style), Paragraph(str(img_info.get("file_size_human", "N/A")), self.body_style)],
            [Paragraph("<b>Image Format</b>", self.body_style), Paragraph(str(img_info.get("format", "N/A")), self.body_style)],
            [Paragraph("<b>Color Mode</b>", self.body_style), Paragraph(str(img_info.get("color_mode", "RGB")), self.body_style)],
        ]
        info_table = Table(img_props, colWidths=[160, 380])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 14))

        # 5. Supporting Image Analysis
        story.append(Paragraph("SUPPORTING IMAGE ANALYSIS", self.section_heading))
        man_info = analysis_data.get("manipulation", {})
        analysis_props = [
            [Paragraph("<b>Compression Characteristics (ELA)</b>", self.body_style), Paragraph(str(man_info.get("compression_status", "Analyzed")), self.body_style)],
            [Paragraph("<b>Resizing / Aspect Ratio</b>", self.body_style), Paragraph(str(man_info.get("resize_status", "Analyzed")), self.body_style)],
            [Paragraph("<b>Noise & Texture Distribution</b>", self.body_style), Paragraph(str(man_info.get("filter_status", "Analyzed")), self.body_style)],
            [Paragraph("<b>EXIF Hardware Metadata</b>", self.body_style), Paragraph(str(man_info.get("metadata_status", "Analyzed")), self.body_style)],
        ]
        analysis_table = Table(analysis_props, colWidths=[200, 340])
        analysis_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(analysis_table)
        story.append(Spacer(1, 14))

        # 6. Technical Basis & Responsible AI Disclaimer
        story.append(Paragraph("DETECTION BASIS & EXPLANATION", self.section_heading))
        explanation_obj = analysis_data.get("explanation", {})
        basis_text = ""
        if isinstance(explanation_obj, dict) and explanation_obj.get("model_basis"):
            basis_text = f"{explanation_obj.get('summary', '')} {explanation_obj.get('model_basis', '')}"
        else:
            basis_text = analysis_data.get(
                "interpretation",
                "The model processed the image tensor and mapped it to learned representations of real camera vs AI-synthesized distributions."
            )
        story.append(Paragraph(basis_text, self.body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("RESPONSIBLE AI DISCLAIMER", self.section_heading))
        disclaimer_text = (
            "Important: AI image detection is probabilistic and represents an algorithmic assessment based on learned visual representations. "
            "It is not absolute proof of origin. Results may be influenced by re-compression, social media filters, or novel generative models. "
            "This assessment should be evaluated alongside context and provenance."
        )
        story.append(Paragraph(disclaimer_text, self.disclaimer_style))

        # Build document
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

report_service = ReportService()
