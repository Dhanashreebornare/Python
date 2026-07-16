import datetime
import textwrap
import streamlit as st
import time

def get_chime_html():
    """Generates an HTML5 audio element with an upbeat success chime notification."""
    sound_url = "https://mixkit.co"
    return f"""
        <iframe src="{sound_url}" allow="autoplay" style="display:none" id="iframeAudio"></iframe>
        <audio autoplay style="display:none;">
            <source src="{sound_url}" type="audio/wav">
        </audio>
    """

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
    
    # 🕺 Dancing Bird GIF Section
    st.markdown("### ⚖️ Verdict Pending...")
    
    bird_gif_url = "https://githubusercontent.com"
    st.image(bird_gif_url, width=90)
    st.caption("Waiting for your absolute ruling...")

    st.write(f"Do you accept {offender_name}'s apology and wish to officially clear her record?")
    
    # Session state to track generation across clicks
    if "generated" not in st.session_state:
        st.session_state.generated = False

    # 🔵 Clean CSS Injection for Blue Primary Button
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #1E88E5 !important;
            color: white !important;
            border: 1px solid #1565C0 !important;
            box-shadow: 0px 4px 10px rgba(30, 136, 229, 0.3) !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #1565C0 !important;
            border-color: #0D47A1 !important;
            color: white !important;
        }
        div.stButton > button:first-child:active {
            background-color: #0D47A1 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Primary type activates the customized styles seamlessly
    if st.button("🌟 Grant Official Forgiveness & Generate Certificate 🌟", type="primary"):
        st.session_state.generated = True
        
        with st.spinner("Processing official pardon paperwork..."):
            time.sleep(1.0)
            
        # 🔊 Playful Audio Chime Notification
        st.components.v1.html(get_chime_html(), height=0, width=0)
        
        # 🎈 Double Celebration: Balloons AND Snow Spray Effects combined!
        st.balloons()
        st.snow()
        
        # 🔔 Styled Visual UI Popups
        st.toast(f"🔔 ALERT: {offender_name}'s apology has been APPROVED!", icon="✅")
        st.success(f"🎉 SUCCESS: Apology accepted! {offender_name} has been formally notified via sound and animations.")

    # Display certificate only after user approval
    if st.session_state.generated:
        st.markdown("### 📜 Verdict: Official Decree of Cleared Grudges")
        st.code(certificate, language="text")

# --- Streamlit Layout Configuration ---
st.set_page_config(page_title="Pardon Portal", page_icon="🕊️", layout="centered")

# Main Title Header
st.title("🕊️ The Apology & Forgiveness Portal")
st.caption("Resolving extreme tracking cases and mischievous behavior.")
st.markdown("---")

# User Input Box
user_input = st.text_input("Enter Your Name (The Person Granting Forgiveness):", value="Your Name Here")

# Initial Trigger Validation
if user_input.strip() == "" or user_input == "Your Name Here":
    st.info("💡 Please type your actual name above to review Dhanashree's case file.")
else:
    generate_portal(your_name=user_input)
