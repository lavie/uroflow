#!/usr/bin/env python3
"""
PDF Report generation for uroflow analysis
Creates a professional medical report with chart, metrics, and interpretation
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class UroflowReportGenerator:
    """Generate professional PDF reports for uroflow analysis"""

    def __init__(self, session_path: Path):
        """
        Initialize the report generator

        Args:
            session_path: Path to the session directory
        """
        self.session_path = session_path
        self.output_path = session_path / "report.pdf"
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Create custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=10,
            alignment=TA_CENTER
        ))

        # Header info style
        self.styles.add(ParagraphStyle(
            name='HeaderInfo',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_CENTER
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2E86AB'),
            spaceBefore=10,
            spaceAfter=6
        ))

        # Clinical note style
        self.styles.add(ParagraphStyle(
            name='ClinicalNote',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceBefore=3
        ))

        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))

    def generate_report(self, metrics: Dict, patient_name: Optional[str] = None,
                       test_datetime: Optional[datetime] = None) -> str:
        """
        Generate the PDF report

        Args:
            metrics: Dictionary containing analysis metrics
            patient_name: Optional patient name
            test_datetime: Optional test date/time (defaults to now)

        Returns:
            Path to the generated PDF file
        """
        # Use current time if not provided
        if test_datetime is None:
            test_datetime = datetime.now()

        # Extract patient name from session if not provided
        if patient_name is None:
            session_name = self.session_path.name
            # Try to extract from session name (format: YYYY-MM-DD-HHMMSS-Patient_Name)
            parts = session_name.split('-', 4)
            if len(parts) > 4:
                patient_name = parts[4].replace('_', ' ')

        # Create PDF document
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            topMargin=2*cm,
            bottomMargin=2*cm,
            leftMargin=2*cm,
            rightMargin=2*cm
        )

        # Build content
        elements = []

        # 1. Title and header
        elements.append(Paragraph("UROFLOWMETRY ANALYSIS REPORT", self.styles['ReportTitle']))
        elements.append(Spacer(1, 8))

        # Patient and test info
        header_info = []
        if patient_name:
            header_info.append(f"<b>Patient:</b> {patient_name}")
        header_info.append(f"<b>Test Date:</b> {test_datetime.strftime('%Y-%m-%d %H:%M')}")
        header_info.append(f"<b>Session ID:</b> {self.session_path.name}")

        for info in header_info:
            elements.append(Paragraph(info, self.styles['HeaderInfo']))

        elements.append(Spacer(1, 12))

        # 2. Chart image (if exists)
        chart_path = self.session_path / "uroflow_chart.png"
        if chart_path.exists():
            # Calculate image size to fit nicely on single page while maintaining aspect ratio
            # Original chart is 11.7 x 8.3 inches (landscape A4), so aspect ratio is 11.7/8.3 = 1.41
            # A4 is 210mm × 297mm, with 20mm margins = 170mm × 257mm available
            # Scale to use available width (170mm) and calculate height to maintain aspect ratio
            img_width = 170 * mm
            img_height = img_width / 1.41  # Maintain original aspect ratio

            # If this is still too tall, scale down both proportionally
            max_height = 110 * mm  # Maximum height to ensure single page
            if img_height > max_height:
                img_height = max_height
                img_width = img_height * 1.41

            img = Image(str(chart_path), width=img_width, height=img_height)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 10))

        # 3. Metrics table
        elements.append(Paragraph("Key Metrics", self.styles['SectionHeader']))

        # Prepare table data (no status column)
        table_data = [
            ['Measurement', 'Value'],
            ['Voided Volume', f"{metrics.get('voided_volume', 0):.1f} ml"],
            ['Peak Flow Rate (Qmax)*', f"{metrics.get('peak_flow_rate', 0):.1f} ml/s"],
            ['Average Flow Rate (Qave)*', f"{metrics.get('average_flow_rate', 0):.1f} ml/s"],
            ['Time to Start', f"{metrics.get('time_to_start', 0):.1f} s"],
            ['Time to Peak', f"{metrics.get('time_to_peak', 0):.1f} s"],
            ['Flow Time', f"{metrics.get('flow_time', 0):.1f} s"],
            ['Total Time', f"{metrics.get('total_time', 0):.1f} s"]
        ]

        # Create table with styling
        table = Table(table_data, colWidths=[6*cm, 4*cm])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),

            # Measurement column (left align)
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('LEFTPADDING', (0, 1), (0, -1), 10),

            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')])
        ]))

        elements.append(table)
        elements.append(Spacer(1, 8))

        # Add note about smoothing methodology
        methodology_note = Paragraph(
            "*Flow rate measurements use 8-second moving average smoothing with 2-second sustained rule for Qmax calculation.",
            self.styles['Footer']
        )
        elements.append(methodology_note)
        elements.append(Spacer(1, 10))

        # 5. Footer - combine into one paragraph to save space
        footer_text = (
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | UroFlow Analysis System v1.0<br/>"
            f"This report is generated by automated analysis from home video recording."
        )
        elements.append(Paragraph(footer_text, self.styles['Footer']))

        # Build PDF
        doc.build(elements)

        return str(self.output_path)