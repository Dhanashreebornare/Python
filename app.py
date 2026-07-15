import streamlit as st
import random
import time
from datetime import datetime
from fpdf import FPDF

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="The HumblePie Protocol", 
    page_icon="🥧", 
    layout="centered"
)

# 2. Helper function to generate PDF bytes safely
def generate_certificate(name, crime):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    
    # Draw double border
    pdf.set_line_width(1)
    pdf.set_draw_color(244, 63, 94) 
    pdf.rect(10, 10, 277, 190)
    pdf.set_line_width(0.5)
    pdf.rect(13, 13, 271, 184)
    
    # Text Titles
    pdf.set_font("Times", "B", 32)
    pdf.set_text_color(15, 23, 42) 
    pdf.cell(0, 30, "OFFICIAL DECREE OF ABSOLUTE FORGIVENESS", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 14)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Issued by the Sovereign High Court of Karma", ln=True, align="C")
    
    pdf.ln(15)
    pdf.set_font("Times", "", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Let it be known to all mortal beings across the cosmos that", ln=True, align="C")
    
    pdf.ln(5)
    pdf.set_font("Times", "B", 26)
    pdf.set_text_color(244, 63, 94)
    pdf.cell(0, 15, name.upper(), ln=True, align="C")
    
    pdf.ln(5)
    pdf.set_font("Times", "", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "has graciously and with unparalleled benevolence extended full absolution for the crime of:", ln=True, align="C")
    
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 14)
    pdf.set_text_color(59, 130, 246) 
    pdf.multi_cell(0, 10, f'"{crime}"', align="C")
    
    pdf.ln(20)
    current_date = datetime.now().strftime("%B %d, %Y")
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 116, 139)
    
    pdf.set_xy(30, 160)
    pdf.cell(80, 10, f"Date: {current_date}", border="T", align="C")
    
    pdf.set_xy(187, 160)
    pdf.cell(80, 10, "Signature of the Overlord", border="T", align="C")
    
    # Output to binary format
    return bytes(pdf.output())

# 3. Main Web Application Header UI
st.title("🥧 The HumblePie Protocol")
st.markdown("<h4 style='text-align: center; color: #94a3b8;'>The Ultimate Database of Global Accountability</h4>", unsafe_allow_html=True)
st.write("---")

# 4. User Interaction Node
user_name = st.text_input("👑 Identify yourself, Your Majesty (Enter Your Name):")

# 5. Lock random selections into Session State memory 
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

    # Output text boxes
    st.error(f"### {st.session_state.current_opening}")
    st.info(f"**Dearest {user_name},**\n\nAn unnamed offender hereby begs for your absolute mercy for the heinous crime of: \n\n👉 *{st.session_state.current_crime}*.")
    st.warning(f"⚖️ **The Sentence:** The High Court of Karma has officially {st.session_state.current_punishment}.")
    
    st.write("---")
    st.write(f"Do you, the flawless **{user_name}**, accept this groveling apology?")
    
    # Interactive Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟩 YES - I am a benevolent deity"):
            st.session_state.choice = "yes"
            st.balloons()
    with col2:
        if st.button("🟥 NO - Let them suffer longer"):
            st.session_state.choice = "no"

    # Action layout handlers
    if "choice" in st.session_state:
        if st.session_state.choice == "yes":
            st.success("✨ Grovel accepted. Balance to the universe is restored. Download your certificate below!")
            
            try:
                pdf_data = generate_certificate(user_name, st.session_state.current_crime)
                st.download_button(
                    label="📥 Download Official Forgiveness Certificate (PDF)",
                    data=pdf_data,
                    file_name=f"Forgiveness_Certificate_{user_name}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Could not build PDF data stream: {e}")
                
        elif st.session_state.choice == "no":
            st.error("🔥 Excellent choice. The LEGO bricks have been scattered. No certificate will be issued for this peasant.")
