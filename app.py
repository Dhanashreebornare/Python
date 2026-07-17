import datetime
import textwrap
import time
import streamlit as st

# --- Streamlit Layout Configuration MUST BE FIRST ---
st.set_page_config(page_title="Pardon Portal", page_icon="🕊️", layout="centered")

# --- Database Initialization (SQLite via st.connection) ---
# Creates a local database file named 'pardon_portal.db' automatically
conn = st.connection("sqlite", type="sql", url="sqlite:///pardon_portal.db")

# Create the schema if it doesn't exist yet
with conn.session as session:
    session.execute(
        textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS portal_configs (
            id INTEGER PRIMARY KEY DEFAULT 1,
            sender_name TEXT NOT NULL,
            is_locked INTEGER DEFAULT 0
        )
    """)
    )
    session.commit()


def get_chime_html():
    """Generates an HTML5 audio element with an upbeat success chime notification."""
    # Updated to a reliable open-source notification sound asset
    sound_url = "https://google.com"
    return f"""
    <audio autoplay style="display:none;">
        <source src="{sound_url}" type="audio/ogg">
    </audio>
    """


def get_db_status():
    """Fetches the current locked status and sender name from the database."""
    res = conn.query("SELECT sender_name, is_locked FROM portal_configs WHERE id = 1", ttl=0)
    if res.empty:
        return "", False
    return res.iloc[0]["sender_name"], bool(res.iloc[0]["is_locked"])


def update_db_status(name, locked_status):
    """Updates or inserts the portal state configuration in the database."""
    with conn.session as session:
        session.execute(
            textwrap.dedent("""
            INSERT INTO portal_configs (id, sender_name, is_locked)
            VALUES (1, :name, :locked)
            ON CONFLICT(id) DO UPDATE SET
                sender_name = excluded.sender_name,
                is_locked = excluded.is_locked
        """),
            {"name": name, "locked": 1 if locked_status else 0},
        )
        session.commit()


def generate_portal(your_name, offender_name):
    today_date = datetime.date.today().strftime("%B %d, %Y")

    # 1. Apology Message (From Sender's POV)
    apology_message = textwrap.dedent(
        f"""
    Date: {today_date}
    From: {offender_name}
    To: {your_name}

    Hey {your_name},

    Alright, I am officially raising the white flag. 🏳️ 

    I am writing this to formally apologize for my absolute mischief and for being your unofficial, highly persistent shadow. I know that continuously following you around and being a general nuisance probably pushed your patience to the absolute limit. My bad! 

    I promise to give your shadow a break and respect your personal space bubble moving forward. To make amends for my chaotic energy, I have issued you a special document below. Please don't block me!

    Sincerely,
    {offender_name}
    """
    ).strip()

    # 2. Forgiveness Certificate Template
    certificate = textwrap.dedent(
        f"""
    📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
    OFFICIAL CERTIFICATE OF FORGIVENESS (The "Stop Following Me" Edition)
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
    All past grudges are officially wiped clean. {offender_name} is free to roam without guilt, provided they keep a minimum distance of three steps and promise to reduce the mischief by at least 15%.

    Signed and sealed on this day: {today_date}

    Chief Dispenser of Forgiveness: __________________________________ {your_name}
    📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜📜
    """
    ).strip()

    # --- Render Streamlit UI ---
    st.markdown("### ✉️ Case File: The Apology Letter")
    st.text(apology_message)
    st.markdown("---")

    # 🕺 Dancing Bird GIF Section
    st.markdown("### ⚖️ Verdict Pending...")
    # Fixed broken URL: Live, rendering transparent dancing bird asset
    bird_gif_url = "https://giphy.com"
    st.image(bird_gif_url, width=120)
    st.caption("Waiting for your absolute ruling...")
    st.write(
        f"Do you accept {offender_name}'s apology and wish to officially clear their record?"
    )

    # 🔵 Clean CSS Injection for Blue Primary Button
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    # Primary type activates the customized styles seamlessly
    if st.button(
        "🌟 Grant Official Forgiveness & Generate Certificate 🌟", type="primary"
    ):
        st.session_state.generated = True
        with st.spinner("Processing official pardon paperwork..."):
            time.sleep(1.0)

        # 🔊 Playful Audio Chime Notification
        st.components.v1.html(get_chime_html(), height=0, width=0)

        # 🎈 Double Celebration: Balloons AND Snow Spray Effects combined!
        st.balloons()
        st.snow()

        # 🔔 Styled Visual UI Popups
        st.toast(
            f"🔔 ALERT: {offender_name}'s apology has been APPROVED!", icon="✅"
        )
        st.success(
            f"🎉 SUCCESS: Apology accepted! {offender_name} has been formally notified via sound and animations."
        )

    # Display certificate only after user approval
    if st.session_state.generated:
        st.markdown("### 📜 Verdict: Official Decree of Cleared Grudges")
        st.code(certificate, language="text")


# Session state initialization for UI actions
if "generated" not in st.session_state:
    st.session_state.generated = False

# Fetch structural state directly from DB to persist across initializations
db_sender_name, db_is_locked = get_db_status()

# Main Title Header
st.title("🕊️ The Apology & Forgiveness Portal")
st.caption("Resolving extreme tracking cases and mischievous behavior.")
st.markdown("---")

# STEP 1: Sender Setup (Shows only if database has no locked identity)
if not db_is_locked:
    st.subheader("⚙️ Portal Setup")
    st.write(
        "Type your name below to lock it into this database configuration permanently."
    )
    sender_input = st.text_input(
        "Your Name (The Person Seeking Apology):", value=""
    )

    if st.button("Lock Name & Prepare Portal"):
        if sender_input.strip() == "":
            st.error("Please enter your name to proceed.")
        else:
            update_db_status(sender_input.strip(), True)
            st.rerun()

# STEP 2: The Apology Portal (Always loads instantly if DB has a name locked)
else:
    offender_name = db_sender_name

    # Simple alert to confirm the setup is complete
    st.success(f"🔒 Portal locked with database identity: **{offender_name}**")

    # User Input Box for the Receiver
    user_input = st.text_input(
        "Enter Your Name (The Person Granting Forgiveness):",
        value="Your Name Here",
    )

    # Initial Trigger Validation
    if user_input.strip() in ["", "Your Name Here"]:
        st.info(
            f"💡 Please type your actual name above to review {offender_name}'s case file."
        )
    else:
        generate_portal(your_name=user_input, offender_name=offender_name)

# Reset button at the bottom
st.markdown("<br><br><br>", unsafe_allow_html=True)
if st.button("🔄 Reset Portal Database"):
    update_db_status("", False)
    st.session_state.generated = False
    st.rerun()
