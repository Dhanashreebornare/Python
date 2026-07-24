import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Vibe with Gaurav", page_icon="🌸", layout="centered")

# --- PASSCODE AUTHENTICATION LOCK ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <link rel="preconnect" href="https://googleapis.com">
        <link rel="preconnect" href="https://gstatic.com" crossorigin>
        <link href="https://googleapis.com" rel="stylesheet">
        <style>
        .stApp { 
            font-family: 'Inter', sans-serif !important;
            background-color: #fff0f5 !important;
            color: #4a1525 !important;
        }
        .lock-container {
            text-align: center;
            padding: 45px 35px;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            box-shadow: 0 15px 35px rgba(255, 182, 193, 0.3);
            border: 2px solid #ffb6c1;
            margin-top: 80px;
            margin-bottom: 20px;
        }
        .lock-title { color: #d84b80; font-weight: 800; font-size: 2.2rem; margin: 15px 0 0 0; }
        .lock-subtitle { color: #657b9e; font-size: 1rem; font-weight: 500; margin-top: 8px; margin-bottom: 25px; }
        .handshake-logo { width: 80px; filter: drop-shadow(0 0 12px rgba(255, 182, 193, 0.4)); }
        
        div[data-testid="stTextInput"] input {
            border-radius: 25px !important;
            border: 2px solid #add8e6 !important;
            background-color: #ffffff !important;
            padding: 12px 20px !important;
            font-size: 1.1rem !important;
            color: #4a1525 !important;
            text-align: center !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s ease;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #ffb6c1 !important;
            box-shadow: 0 0 15px rgba(255, 182, 193, 0.6) !important;
        }
        div[data-testid="stTextInput"] label { display: none !important; }
        footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Visible floral column pillars inside the login viewport layout boundary
    col_l, col_c, col_r = st.columns([1, 8, 1])
    with col_l:
        st.write("🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻")
    with col_r:
        st.write("🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻")
        
    with col_c:
        st.markdown(
            """
            <div class="lock-container">
                <img class="handshake-logo" src="https://icons8.com" alt="Handshake">
                <h2 class="lock-title">Vibe with Gaurav.</h2>
                <div class="lock-subtitle">Verify code to connect securely</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        passcode = st.text_input("Secret Code:", type="password", key="secret_gate", placeholder="Enter passcode here...")
        if passcode:
            if passcode.strip().lower() == "cutie pie":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid entry, buddy! Try again.")
    st.stop()

# 2. Main Lounge UI Core Styles
st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #fff0f5 !important;
        color: #4a1525 !important;
    }
    .lounge-header {
        text-align: center;
        padding: 24px 15px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid #ffb6c1;
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.2);
        margin-bottom: 35px;
    }
    .lounge-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #d84b80 0%, #4a77d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        font-size: 2.2rem;
    }
    .lounge-subtitle { color: #657b9e; font-size: 0.95rem; margin-top: 5px; font-weight: 500; }
    
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #edf7ff 100%) !important;
        color: #1a365d !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 4px 15px rgba(173, 216, 230, 0.1);
        padding: 14px 18px !important;
        border: 1px solid rgba(173, 216, 230, 0.3);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #ffe4e1 0%, #fff0f5 100%) !important;
        color: #4a1525 !important;
        border-radius: 20px 20px 20px 4px !important;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.1);
        padding: 14px 18px !important;
        border: 1px solid rgba(255, 182, 193, 0.3);
    }
    div[data-testid="stChatInput"] {
        border-radius: 35px !important;
        border: 2px solid #ffb6c1 !important;
        background-color: #ffffff !important;
        box-shadow: 0 10px 25px rgba(255, 182, 193, 0.12) !important;
    }
    div[data-testid="stChatInput"] textarea { color: #4a1525 !important; }
    footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

# Avatars Configuration (Using direct structural URL reference)
USER_AVATAR = "🌸"
BOT_AVATAR = "https://freepik.com"

def get_gemini_client():
    if "GEMINI_API_KEY" in st.secrets:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    elif os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    else:
        st.error("🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! What's up? Finally you remembered your best friend. Aur bata, what's cooking today? 🤔",
        }
    ]

# Layout distribution to frame flower tracks around the active chat modules
main_l, main_c, main_r = st.columns([1, 8, 1])

with main_l:
    st.write("🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻")
with main_r:
    st.write("🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻\n\n🪻")

with main_c:
    # 3. Main Header Card
    st.markdown(
        """
        <div class="lounge-header">
            <h1 class="lounge-title">Vibe with Gaurav.</h1>
            <div class="lounge-subtitle">Your Hinglish bestie • Available 24/7</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. History Feed Render
    for message in st.session_state.messages:
        avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# 5. Live Interaction Engine
if user_query := st.chat_input("Say something to Gaurav..."):
    with main_c:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with main_c:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""
            
            fallback_options = ["Bhai, thoda busy hoon! Mummy ne kaam saupa hai. 😂", "Arey yaar, internet bohot slow chal raha hai yahan... chill mar! ☕", "Bro, phone ki battery khatam hone wali hai! Late text karu? 😉", "Tension mat le bhai, main yahin hoon. Thoda breaks chahiye! 😂"]
            api_contents = [types.Content(role="user" if msg["role"] == "user" else "model", parts=[types.Part.from_text(text=msg["content"])]) for msg in st.session_state.messages]
            system_instruction = "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. CRUCIAL: Read the user's text carefully and answer their exact question contextually. Never use hardcoded greeting lists or switch topics randomly. Respond dynamically. Chat casually using informal internet slang and short sentences like a text message. You speak naturally in a mix of Hindi and English (Hinglish). Use casual terms like 'Bhai', 'Yaar', 'Bro', 'Chill mar', and 'tension mat le'. Do NOT use Gujarati phrases like 'Kem cho' or 'Majama' in every sentence. Only use them rarely if explicitly asked about Gujarati or if it fits a niche joke naturally. Crucially, you must use emojis effectively: include exactly ONE or a maximum of TWO highly relevant emojis per turn. Do not spam arrays of emojis under any circumstance."
            
            try:
                bot_response = get_gemini_client().models.generate_content(model="gemini-3.5-flash", contents=api_contents, config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.4)).text
            except:
                bot_response = random.choice(fallback_options)
                
            if bot_response:
                for chunk in bot_response.split():
                    full_response += chunk + " "
                    time.sleep(0.60)
                    message_placeholder.markdown(full_response)
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.rerun()
