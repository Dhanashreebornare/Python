import datetime
import textwrap
import streamlit as st
import time

def generate_portal(your_name):
    offender_name = "Dhanashree"
    today_date = datetime.date.today().strftime("%B %d, %Y")
    
    # 1. Apology Message (From Dhanashree's POV)
    apology_message = textwrap.dedent(f"""
        Date: {today_date}
        From: {offender_name}
        To: {your_name}

        Hey {your_name},

        Alright, I am officially raising the white flag. 🏳️
        
        I am writing this to formally apologize for my absolute mischief and for being your 
        unofficial, highly persistent shadow. I know that continuously following you around 
        and being a general nuisance probably pushed your patience to the absolute limit. 
        My bad! I promise to give your shadow a break and respect your personal space bubble 
        moving forward. To make amends for my chaotic energy, I have issued you a special 
        document below. Please don't block me!

        Sincerely,
        {offender_name}
    """).strip()

    # 2. Forgiveness Certificate Template
    certificate = textwrap.dedent(f"""
        📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
                         OFFICIAL CERTIFICATE OF FORGIVENESS
                            (The "Stop Following Me" Edition)
        📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
        
        This document legally and unconditionally declares that:
        
                                🌟 {offender_name} 🌟
                                  
        Has been granted 100% total immunity and forgiveness by:
        
                                👑 {your_name} 👑
                                 
        For past crimes including, but not limited to:
        - Unwarranted mischief and chaotic energy.
        - Operating as a full-time, unpaid personal stalker.
        - Disrupting the peace by constantly following {your_name} around.
        
        TERMS & CONDITIONS:
        All past grudges are officially wiped clean. {offender_name} is free to roam 
        without guilt, provided she keeps a minimum distance of three steps and promises 
        to reduce the mischief by at least 15%.
        
        Signed and sealed on this day: {today_date}
        
        Chief Dispenser of Forgiveness:
        __________________________________
        {your_name}
        
        📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
    """).strip()

    # --- Render Streamlit UI ---
    st.markdown("### ✉️ Case File: The Apology Letter")
    st.text(apology_message)
    
    st.markdown("---") 
    
    st.markdown("### ⚖️ Verdict Pending...")
    st.write(f"Do you accept {offender_name}'s apology and wish to officially clear her record?")
    
    # Session state trick to keep the certificate visible after generation
    if "generated" not in st.session_state:
        st.session_state.generated = False

    # Button to dynamically generate the certificate
    if st.button("🌟 Grant Official Forgiveness & Generate Certificate 🌟"):
        st.session_state.generated = True
        
        # Playful loading animation
        with st.spinner("Processing official pardon paperwork..."):
            time.sleep(1.5)
            
        # Celebration animations
        st.balloons()
        st.snow()
        st.success(f"Absolution granted! {offender_name} is officially off the hook.")

    # Show certificate only if the button was pressed
    if st.session_state.generated:
        st.markdown("### 📜 Verdict: Official Decree of Cleared Grudges")
        st.code(certificate, language="text")

# --- Streamlit Layout Configuration ---
st.set_page_config(page_title="Pardon Portal", page_icon="🕊️", layout="centered")

# Main Header Area
st.title("🕊️ The Apology & Forgiveness Portal")
st.caption("Resolving extreme tracking cases and mischievous behavior with total legal absolution.")
st.markdown("---")

# User Input
user_input = st.text_input("Enter Your Name (The Person Granting Forgiveness):", value="Your Name Here")

# Initial Trigger Button
if user_input.strip() == "" or user_input == "Your Name Here":
    st.info("💡 Please type your actual name above to review Dhanashree's case file.")
else:
    generate_portal(your_name=user_input)
