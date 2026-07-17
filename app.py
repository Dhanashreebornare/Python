import datetime
import textwrap
import time
import streamlit as st

# --- 1. Streamlit Layout Configuration MUST BE FIRST ---
st.set_page_config(page_title="Pardon Portal", page_icon="🕊️", layout="centered")

# --- 2. Session State Initialization ---
if "generated" not in st.session_state:
    st.session_state.generated = False

# --- 3. Core Helper Functions ---
def get_chime_html():
    """Generates an HTML5 audio element with an upbeat success chime notification."""
    sound_url = "https://google.com"
    return f"""
        <audio autoplay style="display:none;">
            <source src="{sound_url}" type="audio/ogg">
        </audio>
    """

def generate_portal(your_name, offender_name):
    today_date = datetime.date.today().strftime("%B %d, %Y")
    
    # Apology Message (From Sender's POV)
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

    # Forgiveness Certificate Template
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
        without guilt, provided she keeps a minimum distance of thousand miles and promises 
        to reduce the mischief by at least 50%.
        
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
    
    # Dancing Bird GIF Section
    st.markdown("### ⚖️ Verdict Pending...")
    bird_gif_url = "https://giphy.com"
    st.image(bird_gif_url, width=120)
    st.caption("Waiting for your absolute ruling...")

    st.write(f"Do you accept {offender_name}'s apology and wish to officially clear their record?")
    
    # Clean CSS Injection for Blue Primary Button
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

    if st.button("🌟 Grant Official Forgiveness & Generate Certificate 🌟", type="primary"):
        st.session_state.generated = True
        
        with st.spinner("Processing official pardon paperwork..."):
            time.sleep(1.0)
            
        st.components.v1.html(get_chime_html(), height=0, width=0)
        st.balloons()
        st.snow()
        
        st.toast(f"🔔 ALERT: {offender_name}'s apology has been APPROVED!", icon="✅")
        st.success(f"🎉 SUCCESS: Apology accepted! {offender_name} has been formally notified via sound and animations.")

    if st.session_state.generated:
        st.markdown("### 📜 Verdict: Official Decree of Cleared Grudges")
        st.code(certificate, language="text")

# --- 4. Main Application Routing Workflow ---

st.title("🕊️ The Apology & Forgiveness Portal")
st.caption("A private, secure communication channel.")
st.markdown("---")

# Read privacy parameters directly from secrets configuration engine
try:
    offender_name = st.secrets["portal_config"]["sender_name"]
    target_receiver = st.secrets["portal_config"]["target_receiver"]
    is_locked = st.secrets["portal_config"]["is_locked"]
except KeyError:
    # Safe defaults if file system configurations are missing
    offender_name = "Sender"
    target_receiver = "Receiver"
    is_locked = False

# Fallback Configuration helper screen
if not is_locked:
    st.subheader("⚙️ Portal Setup Required")
    st.info("💡 Repository Setup Needed: Please ensure you have created a `.streamlit/secrets.toml` file inside your GitHub repository folder layout structure.")
    st.code(textwrap.dedent("""
        [portal_config]
        sender_name = "Dhanashree"
        target_receiver = "The Target Recipient Name"
        is_locked = true
    """), language="toml")

# Active Privacy Protected Portal Mode
else:
    # Ask the visitor for their identity credentials
    user_input = st.text_input("Enter Your Name to Access Your Portal File:", value="Your Name Here")

    if user_input.strip() in ["", "Your Name Here"]:
        st.info("💡 Identity confirmation: Please enter your name to authenticate directory folder access clearances.")
    
    # 🔒 PRIVACY GATE: Check user's name against the secret target receiver value
    elif user_input.strip().lower() != target_receiver.strip().lower():
        st.error("⛔ ACCESS DENIED: This application portal link is strictly confidential and locked to a single specific matching recipient string name indicator.")
        st.warning("Ensure you typed the exact name configuration intended by the app creator initialization records.")
    
    # Successful authorization check
    else:
        st.success(f"🔓 Identity Confirmed: Welcome **{user_input.strip()}**. You have one pending case file open request from **{offender_name}**.")
        generate_portal(your_name=user_input.strip(), offender_name=offender_name)
