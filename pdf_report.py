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
        elements.append(Spacer(1, 12))

        # Patient and test info
        header_info = []
        if patient_name:
            header_info.append(f"<b>Patient:</b> {patient_name}")
        header_info.append(f"<b>Test Date:</b> {test_datetime.strftime('%Y-%m-%d %H:%M')}")
        header_info.append(f"<b>Session ID:</b> {self.session_path.name}")

        for info in header_info:
            elements.append(Paragraph(info, self.styles['HeaderInfo']))

        elements.append(Spacer(1, 20))

        # 2. Chart image (if exists)
        chart_path = self.session_path / "uroflow_chart.png"
        if chart_path.exists():
            # Calculate image size to fit nicely on page
            # A4 is 210mm × 297mm, with 20mm margins = 170mm × 257mm available
            # Use about 60% of page height for chart
            img_width = 170 * mm
            img_height = 120 * mm

            img = Image(str(chart_path), width=img_width, height=img_height)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 15))

        # 3. Metrics table
        elements.append(Paragraph("Key Metrics", self.styles['SectionHeader']))

        # Prepare table data
        table_data = [
            ['Measurement', 'Value', 'Normal Range', 'Status'],
            ['Voided Volume', f"{metrics.get('voided_volume', 0):.1f} ml", '150-500 ml',
             self._get_volume_status(metrics.get('voided_volume', 0))],
            ['Peak Flow Rate (Qmax)*', f"{metrics.get('peak_flow_rate', 0):.1f} ml/s", '>15 ml/s',
             self._get_qmax_status(metrics.get('peak_flow_rate', 0))],
            ['Average Flow Rate (Qave)*', f"{metrics.get('average_flow_rate', 0):.1f} ml/s", '>10 ml/s',
             self._get_qave_status(metrics.get('average_flow_rate', 0))],
            ['Time to Peak', f"{metrics.get('time_to_peak', 0):.1f} s", '-', '-'],
            ['Flow Time', f"{metrics.get('flow_time', 0):.1f} s", '-', '-'],
            ['Total Time', f"{metrics.get('total_time', 0):.1f} s", '-', '-']
        ]

        # Create table with styling
        table = Table(table_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
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

        # Color code status column based on values
        for i, row in enumerate(table_data[1:], 1):  # Skip header
            status = row[3]
            if 'Normal' in status:
                table.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.green)]))
            elif 'Low' in status or 'Below' in status:
                table.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.red)]))
            elif 'Borderline' in status:
                table.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.orange)]))

        elements.append(table)
        elements.append(Spacer(1, 10))

        # Add note about smoothing methodology
        methodology_note = Paragraph(
            "*Flow rate measurements use 4-second moving average smoothing with 2-second sustained rule for Qmax calculation, following clinical standards.",
            self.styles['Footer']
        )
        elements.append(methodology_note)
        elements.append(Spacer(1, 15))

        # 4. Clinical interpretation
        elements.append(Paragraph("Clinical Interpretation", self.styles['SectionHeader']))

        interpretations = self._generate_interpretations(metrics)
        for interp in interpretations:
            elements.append(Paragraph(f"• {interp}", self.styles['ClinicalNote']))

        elements.append(Spacer(1, 30))

        # 5. Footer
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | UroFlow Analysis System v1.0"
        elements.append(Paragraph(footer_text, self.styles['Footer']))

        disclaimer = "This report is generated by automated analysis and should be reviewed by a qualified healthcare professional."
        elements.append(Paragraph(disclaimer, self.styles['Footer']))

        # Build PDF
        doc.build(elements)

        return str(self.output_path)

    def _get_volume_status(self, volume: float) -> str:
        """Determine status based on voided volume"""
        if volume < 150:
            return "Low"
        elif volume > 500:
            return "High"
        else:
            return "Normal"

    def _get_qmax_status(self, qmax: float) -> str:
        """Determine status based on peak flow rate"""
        if qmax < 10:
            return "Below Normal"
        elif qmax < 15:
            return "Borderline"
        else:
            return "Normal"

    def _get_qave_status(self, qave: float) -> str:
        """Determine status based on average flow rate"""
        if qave < 10:
            return "Below Normal"
        else:
            return "Normal"

    def _generate_interpretations(self, metrics: Dict) -> list:
        """Generate clinical interpretation notes based on metrics"""
        interpretations = []

        # Volume interpretation
        volume = metrics.get('voided_volume', 0)
        if volume < 150:
            interpretations.append("Low voided volume may affect the accuracy of flow measurements. Consider repeating the test with fuller bladder.")
        elif volume > 500:
            interpretations.append("High voided volume recorded.")

        # Peak flow interpretation
        qmax = metrics.get('peak_flow_rate', 0)
        if qmax < 10:
            interpretations.append("Significantly reduced peak flow rate suggests possible bladder outlet obstruction or detrusor weakness.")
        elif qmax < 15:
            interpretations.append("Borderline peak flow rate may indicate mild obstruction or early bladder dysfunction.")
        else:
            interpretations.append("Peak flow rate is within normal range.")

        # Average flow interpretation
        qave = metrics.get('average_flow_rate', 0)
        if qave < 10:
            interpretations.append("Below normal average flow rate suggests impaired voiding function.")

        # Flow time interpretation
        flow_time = metrics.get('flow_time', 0)
        if flow_time > 30:
            interpretations.append("Prolonged flow time may indicate obstruction or bladder dysfunction.")

        # Time to peak interpretation
        time_to_peak = metrics.get('time_to_peak', 0)
        if time_to_peak > 10:
            interpretations.append("Delayed time to peak flow may suggest hesitancy or obstruction.")

        if not interpretations:
            interpretations.append("All parameters are within normal ranges.")

        return interpretations