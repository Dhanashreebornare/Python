# Helper function to generate PDF
def generate_certificate(name, crime):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    
    # Draw an elegant double border
    pdf.set_line_width(1)
    pdf.set_draw_color(244, 63, 94) # Rose color
    pdf.rect(10, 10, 277, 190)
    pdf.set_line_width(0.5)
    pdf.rect(13, 13, 271, 184)
    
    # Title Header
    pdf.set_font("Times", "B", 32)
    pdf.set_text_color(15, 23, 42) # Slate color
    pdf.cell(0, 30, "OFFICIAL DECREE OF ABSOLUTE FORGIVENESS", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 14)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Issued by the Sovereign High Court of Karma", ln=True, align="C")
    
    # Main Body Text
    pdf.ln(15)
    pdf.set_font("Times", "", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Let it be known to all mortal beings across the cosmos that", ln=True, align="C")
    
    # Royal Benefactor Name
    pdf.ln(5)
    pdf.set_font("Times", "B", 26)
    pdf.set_text_color(244, 63, 94)
    pdf.cell(0, 15, name.upper(), ln=True, align="C")
    
    # Forgiveness Text
    pdf.ln(5)
    pdf.set_font("Times", "", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "has graciously and with unparalleled benevolence extended full absolution for the crime of:", ln=True, align="C")
    
    # The Crime Box
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 14)
    pdf.set_text_color(59, 130, 246) # Blue color
    pdf.multi_cell(0, 10, f'"{crime}"', align="C")
    
    # Footer and Date Stamp
    pdf.ln(20)
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Split layout for Date and Signature
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 116, 139)
    
    # Date Line (Left side)
    pdf.set_xy(30, 160)
    pdf.cell(80, 10, f"Date: {current_date}", border="T", align="C")
    
    # Seal / Signature Line (Right side)
    pdf.set_xy(187, 160)
    pdf.cell(80, 10, "Signature of the Overlord", border="T", align="C")
    
    # FIX: Explicitly convert the output string/bytearray into clean byte data
    return bytes(pdf.output())
