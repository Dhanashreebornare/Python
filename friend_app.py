import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="The Secret Lounge", page_icon="🤝", layout="centered")

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
            background-color: #0b0f19 !important;
            background-image: linear-gradient(135deg, #0b0f19 0%, #1a0f30 100%) !important;
            color: #f1f5f9 !important;
        }
        .floral-sidebar-left {
            position: fixed; left: 40px; top: 0; bottom: 0; width: 40px;
            display: flex; flex-direction: column; justify-content: space-around;
            align-items: center; font-size: 28px; z-index: 999; pointer-events: none;
        }
        .floral-sidebar-right {
            position: fixed; right: 40px; top: 0; bottom: 0; width: 40px;
            display: flex; flex-direction: column; justify-content: space-around;
            align-items: center; font-size: 28px; z-index: 999; pointer-events: none;
        }
        .lock-container {
            text-align: center;
            padding: 45px 35px;
            background: rgba(26, 21, 44, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(147, 51, 234, 0.3);
            margin-top: 80px;
            margin-bottom: 20px;
        }
        .lock-title { color: #a855f7; font-weight: 800; font-size: 2.2rem; margin: 15px 0 0 0; }
        .lock-subtitle { color: #94a3b8; font-size: 1rem; font-weight: 500; margin-top: 8px; margin-bottom: 25px; }
        .handshake-logo { width: 80px; filter: drop-shadow(0 0 12px rgba(168, 85, 247, 0.5)); }
        
        div[data-testid="stTextInput"] input {
            border-radius: 25px !important;
            border: 2px solid #3b82f6 !important;
            background-color: #0f172a !important;
            padding: 12px 20px !important;
            font-size: 1.1rem !important;
            color: #f8fafc !important;
            text-align: center !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.3s ease;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #a855f7 !important;
            box-shadow: 0 0 15px rgba(168, 85, 247, 0.6) !important;
        }
        div[data-testid="stTextInput"] label { display: none !important; }
        footer, header { visibility: hidden; }
        </style>
        
        <div class="floral-sidebar-left">🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻</div>
        <div class="floral-sidebar-right">🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻</div>
        
        <div class="lock-container">
            <img class="handshake-logo" src="https://icons8.com" alt="Handshake">
            <h2 class="lock-title">The Secret Lounge</h2>
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
        background-color: #0b0f19 !important;
        background-image: linear-gradient(135deg, #0b0f19 0%, #1a0f30 100%) !important;
        color: #f1f5f9 !important;
    }
    .floral-sidebar-left {
        position: fixed; left: 40px; top: 0; bottom: 0; width: 40px;
        display: flex; flex-direction: column; justify-content: space-around;
        align-items: center; font-size: 28px; z-index: 999; pointer-events: none;
    }
    .floral-sidebar-right {
        position: fixed; right: 40px; top: 0; bottom: 0; width: 40px;
        display: flex; flex-direction: column; justify-content: space-around;
        align-items: center; font-size: 28px; z-index: 999; pointer-events: none;
    }
    .lounge-header {
        text-align: center;
        padding: 24px 15px;
        background: rgba(26, 21, 44, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(147, 51, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 35px;
    }
    .lounge-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #3b82f6 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        font-size: 2.2rem;
    }
    .lounge-subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 5px; font-weight: 500; }
    
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        color: #f8fafc !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        padding: 14px 18px !important;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #2e1065 0%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
        border-radius: 20px 20px 20px 4px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        padding: 14px 18px !important;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }
    div[data-testid="stChatInput"] {
        border-radius: 35px !important;
        border: 2px solid #475569 !important;
        background-color: #0f172a !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stChatInput"] textarea { color: #f8fafc !important; }
    footer, header { visibility: hidden; }
    </style>
    
    <div class="floral-sidebar-left">🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻</div>
    <div class="floral-sidebar-right">🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻<br>🪻</div>
    """,
    unsafe_allow_html=True
)

# Avatars Configuration 
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

# 3. Main Header Card
st.markdown(
    """
    <div class="lounge-header">
        <h1 class="lounge-title">The Secret Lounge</h1>
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
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            client = get_gemini_client()
            api_contents = []
            for msg in st.session_state.messages:
                role_type = "user" if msg["role"] == "user" else "model"
                api_contents.append(
                    types.Content(
                        role=role_type,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )
            
            # Formatted nicely inside the box using string continuation
            system_instruction = (
                "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                "CRUCIAL: Read the user's text carefully and answer their exact question contextually. "
                "Never use hardcoded greeting lists or switch topics randomly. Respond dynamically. "
                "Chat casually using informal internet slang and short sentences like a text message. "
                "You speak naturally in a mix of Hindi and English (Hinglish). Use casual terms like "
                "'Bhai', 'Yaar', 'Bro', 'Chill mar', and 'tension mat le'. Do NOT use Gujarati phrases "
                "like 'Kem cho' or 'Majama' in every sentence. Only use them rarely if explicitly "
                "asked about Gujarati or if it fits a niche joke naturally. Crucially, you must use ""emojis effectively: include exactly ONE or a maximum of TWO highly relevant emojis ""per turn. Do not spam arrays of emojis under any circumstance.")response = client.models.generate_content(model="gemini-3.5-flash",contents=api_contents,config=types.GenerateContentConfig(system_instruction=system_instruction,temperature=0.4,),)bot_response = response.textexcept Exception as e:fallback_options = ["Bhai, thoda busy hoon! Mummy ne kaam saupa hai. 😂","Arey yaar, internet bohot slow chal raha hai yahan... chill mar! ☕","Bro, phone ki battery khatam hone wali hai! Late text karu? 😉","Tension mat le bhai, main yahin hoon. Thoda breaks chahiye! 😂"]bot_response = random.choice(fallback_options)if bot_response:for chunk in bot_response.split():full_response += chunk + " "time.sleep(0.60)message_placeholder.markdown(full_response)message_placeholder.markdown(full_response)st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.rerun()
