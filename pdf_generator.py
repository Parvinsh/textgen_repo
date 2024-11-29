from fpdf import FPDF
import os

def save_to_pdf(answer):
    """Save the answer to a PDF file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, answer)

    pdf_output = "/tmp/answer_output.pdf"
    pdf.output(pdf_output)
    return pdf_output

