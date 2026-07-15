import datetime
import textwrap
import streamlit as st

def generate_playful_pardon(your_name):
    offender_name = "Dhanashree"
    today_date = datetime.date.today().strftime("%B %d, %Y")
    
    # textwrap.dedent removes leading whitespace/blank tabs from the deployment output
    apology_message = textwrap.dedent(f"""
        --- THE ACCUSED SPEAKS ---
        Date: {today_date}

        Hey {offender_name},

        Alright, I am officially raising the white flag. 🏳️
        
        I am writing this to formally apologize for my absolute mischief and for being your 
        unofficial, highly persistent shadow. I know that continuously following you around 
        and being a general nuisance probably pushed your patience to the absolute limit. 
        
        My bad! I promise to give your shadow a break and respect your personal space bubble 
        moving forward. To make amends for my chaotic energy, I have issued you a special 
        document below. 

        Please don't block me,
        {your_name}
    """).strip()

    certificate = textwrap.dedent(f"""
        📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
                         OFFICIAL CERTIFICATE OF FORGIVENESS
                            (The "Stop Following Me" Edition)
        📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
        
        This document legally and unconditionally declares that:
        
                                🌟 {offender_name} 🌟
                                  
        Is hereby granted 100% total immunity and forgiveness by:
        
                                👑 {your_name} 👑
                                 
        For crimes including, but not limited to:
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

    # Streamlit UI Components
    st.subheader("✉️ The Apology")
    st.text(apology_message)
    
    st.markdown("---") # Visual divider line
    
    st.subheader("📜 The Official Certificate")
    st.code(certificate, language="text") # st.code preserves the exact ASCII layout without scaling alignment bugs

# --- Streamlit App Entry Point ---
st.title("🕊️ The Apology & Forgiveness Portal")

# Sidebar or main input for your name
user_input = st.text_input("Enter Your Name:", value="Your Name Here")

if st.button("Generate & Deploy Documents"):
    generate_playful_pardon(your_name=user_input)
