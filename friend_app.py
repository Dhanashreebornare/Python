import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Vibe with Gaurav", page_icon="💐", layout="centered")

# Initialize global authentication tracking state safely
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- PASSCODE AUTHENTICATION LOCK ---
if not st.session_state.authenticated:
    st.markdown("""<style>.stApp {font-family: 'Inter', sans-serif !important; background: linear-gradient(135deg, #2d1124 0%, #4a1539 100%) !important; color: #000000 !important;} .lock-container {text-align: center; padding: 45px 35px; background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 24px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.5); margin-top: 40px; margin-bottom: 20px;} div[data-testid="stTextInput"] input {border-radius: 25px !important; border: 1px solid rgba(0, 0, 0, 0.2) !important; background-color: #ffffff !important; padding: 12px 20px !important; font-size: 1.1rem !important; color: #000000 !important; text-align: center !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important; transition: all 0.3s ease;} div[data-testid="stTextInput"] input:focus {border-color: #000000 !important; box-shadow: 0 0 12px rgba(0, 0, 0, 0.2) !important;} div[data-testid="stTextInput"] label {display: none !important;} footer {visibility: hidden !important;}</style>""", unsafe_allow_html=True)
    st.markdown("""<div class="lock-container"><img style="width: 100px; height: auto;" src="https://openclipart.org" alt="Bouquet"><h2 style="color: #000000; font-weight: 800; font-size: 2.2rem; margin: 15px 0 0 0;">Vibe with Gaurav.</h2><div style="color: #444444; font-size: 1rem; font-weight: 500; margin-top: 8px; margin-bottom: 12px;">Verify code to connect securely</div><div style="font-size: 1.2rem; margin-bottom: 20px; letter-spacing: 4px;">🌸 ✨ 🪻 ✨ 🌸</div></div>""", unsafe_allow_html=True)
    
    passcode_input = st.text_input("Secret Code:", type="password", key="secret_gate", placeholder="Enter passcode here...")
    
    if passcode_input:
        # Securely fetch the master passcode value from the Streamlit Secrets file environment
        if "SECRET_PASSCODE" in st.secrets:
            master_passcode = st.secrets["SECRET_PASSCODE"]
        elif os.environ.get("SECRET_PASSCODE"):
            master_passcode = os.environ.get("SECRET_PASSCODE")
        else:
            st.error("🔒 Security configuration missing! Please add 'SECRET_PASSCODE' to your Streamlit Secrets.")
            st.stop()
            
        # Cleanly validate the entry case-insensitively
        if passcode_input.strip().lower() == master_passcode.strip().lower():
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
        background: linear-gradient(135deg, #2d1124 0%, #4a1539 100%) !important;
        color: #000000 !important;
    }
    .lounge-header {
        text-align: center;
        padding: 24px 15px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 35px;
    }
    .lounge-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #000000 !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .lounge-subtitle { color: #444444; font-size: 0.95rem; margin-top: 5px; font-weight: 500; margin-bottom: 10px; }
    
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 14px 18px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background-color: #f5ecef !important;
        color: #000000 !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 14px 18px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    
    div[data-testid="stChatInput"] {
        border-radius: 35px !important;
        background-color: #ffffff !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
        border: none !important;
    }
    div[data-testid="stChatInput"] textarea { 
        color: #000000 !important; 
        font-size: 1.05rem !important;
    }
    footer { visibility: hidden !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Avatars Configuration
USER_AVATAR = "🌸"
BOT_AVATAR = (
    "data:image/svg+xml;utf8,<svg xmlns='http://w3.org' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='45' fill='%23f5ecef' stroke='%232d1124' stroke-width='3'/>"
    "<path d='M35,35 Q40,25 45,35' stroke='%23000000' stroke-width='4' fill='none' stroke-linecap='round'/>"
    "<path d='M65,35 Q60,25 55,35' stroke='%23000000' stroke-width='4' fill='none' stroke-linecap='round'/>"
    "<circle cx='40' cy='42' r='4' fill='%23000000'/>"
    "<circle cx='60' cy='42' r='4' fill='%23000000'/>"
    "<path d='M40,65 Q50,75 60,65' stroke='%23000000' stroke-width='4' fill='none' stroke-linecap='round'/>"
    "<path d='M25,25 L35,10 L45,22 L50,5 L58,22 L68,10 L75,25' stroke='%232d1124' stroke-width='4' fill='none' stroke-linejoin='round'/>"
    "</svg>"
)

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

# 3. Main Header Card
st.markdown(
    """
    <div class="lounge-header">
        <h1 class="lounge-title">Vibe with Gaurav.</h1>
        <div class="lounge-subtitle">Your Hinglish bestie • Available 24/7</div>
        <div style="font-size: 1.2rem; letter-spacing: 4px;">🌸 ✨ 🪻 ✨ 🌸</div>
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
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
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
            for char in bot_response:
                full_response += char
                time.sleep(0.008)
                message_placeholder.markdown(full_response)
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
    st.rerun()
