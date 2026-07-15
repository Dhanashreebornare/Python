import streamlit as st
import random
import time
from datetime import datetime
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(
    page_title="The HumblePie Protocol", 
    page_icon="🥧", 
    layout="centered"
)

# 2. Styling (Dark Mode & Fun Fonts)
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1 { color: #f43f5e !important; text-align: center; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { background-color: #3b82f6; color: white; width: 100%; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

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
    
    return pdf.output()

# 3. Main Header
st.title("🥧 The HumblePie Protocol")
st.markdown("<h3 style='text-align: center; color: #94a3b8;'>The Ultimate Database of Global Accountability</h3>", unsafe_allow_html=True)
st.write("---")

# 4. User Input
user_name = st.text_input("👑 Identify yourself, Your Majesty (Enter Your Name):")

# 5. Persistent Session State to lock random choices per user session
if user_name:
    if "current_crime" not in st.session_state:
        crimes = [
            "breathing your oxygen without written permission",
            "existing in a 5-mile radius of your perfection without a license",
            "allowing the sun to shine too brightly directly into your magnificent eyes",
            "not preemptively apologizing for things they haven't even done yet"
        ]
        openings = [
            "Breaking News: The universe has ground to a halt.",
            "Alert: All planetary operations have been suspended.",
            "Hear ye, hear ye! A royal decree of absolute regret has been issued."
        ]
        punishments = [
            "sentenced to walk on sharp LEGO bricks for eternity",
            "ordered to write a 10,000-page essay on why you are always right",
            "banned from ever choosing the Netflix movie again",
            "required to perform a dramatic interpretive dance expressing their shame"
        ]
        st.session_state.current_opening = random.choice(openings)
        st.session_state.current_crime = random.choice(crimes)
        st.session_state.current_punishment = random.choice(punishments)

    # Display Loading Visuals
    with st.spinner("🔄 Checking the cosmic ledger for bad vibes..."):
        time.sleep(0.5)
        
    # Content display
    st.error(f"### {st.session_state.current_opening}")
    st.info(f"**Dearest {user_name},**\n\nAn unnamed offender hereby begs for your absolute mercy for the heinous crime of: \n\n👉 *{st.session_state.current_crime}*.")
    st.warning(f"⚖️ **The Sentence:** The High Court of Karma has officially {st.session_state.current_punishment}.")
    
    st.write("---")
    st.write(f"Do you, the flawless **{user_name}**, accept this groveling apology?")
    
    # Interactive response buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟩 YES - I am a benevolent deity"):
            st.session_state.choice = "yes"
            st.balloons()
    with col2:
        if st.button("🟥 NO - Let them suffer longer"):
            st.session_state.choice = "no"

    # Action output based on button clicks
    if "choice" in st.session_state:
        if st.session_state.choice == "yes":
            st.success("✨ Grovel accepted. Balance to the universe is restored. You can now download your official certificate below!")
            
            # Generate the PDF in system memory
            pdf_data = generate_certificate(user_name, st.session_state.current_crime)
            
            # Native Streamlit Download Button
            st.download_button(
                label="📥 Download Official Forgiveness Certificate (PDF)",
                data=pdf_data,
                file_name=f"Certificate_of_Forgiveness_{user_name}.pdf",
                mime="application/pdf"
            )
        elif st.session_state.choice == "no":
            st.error("🔥 Excellent choice. The LEGO bricks have been scattered. No certificate will be issued for this peasant.")
