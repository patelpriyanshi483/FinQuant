# report_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

def generate_pdf_report(portfolio, analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []
    # Title
    flowables.append(Paragraph(f"Portfolio Report: {portfolio['name']}", styles['Title']))
    flowables.append(Spacer(1, 12))
    # Portfolio Information
    flowables.append(Paragraph(f"Description: {portfolio['description']}", styles['Normal']))
    flowables.append(Paragraph(f"Tags: {portfolio['tags']}", styles['Normal']))
    flowables.append(Paragraph(f"Risk Tolerance: {portfolio['risk_tolerance']}", styles['Normal']))
    flowables.append(Spacer(1, 12))
    # Metrics Table
    metrics = analysis['metrics']
    data = [
        ['Metric', 'Value'],
        ['Expected Return', f"{metrics['expected_return']:.2%}"],
        ['Volatility', f"{metrics['volatility']:.2%}"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"]
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ]))
    flowables.append(table)
    flowables.append(Spacer(1, 12))
    doc.build(flowables)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
