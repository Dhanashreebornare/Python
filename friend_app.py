import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Chat with Gaurav", page_icon="🌸", layout="centered")

# --- REDESIGNED: SECRET CODE LOCK ("Cutie Pie") ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <link rel="preconnect" href="https://googleapis.com">
        <link rel="preconnect" href="https://gstatic.com" crossorigin>
        <link href="https://googleapis.com" rel="stylesheet">
        <style>
        /* Immersive Login Workspace Page Background Setup */
        .stApp { 
            font-family: 'Inter', sans-serif !important;
            background-color: #fffdf0; 
            background-image: 
                radial-gradient(rgba(255, 182, 193, 0.4) 1.5px, transparent 1.5px), 
                radial-gradient(rgba(173, 216, 230, 0.4) 1.5px, transparent 1.5px);
            background-size: 30px 30px;
            background-position: 0 0, 15px 15px;
        }
        /* Aesthetic Frosted Glass Card Design for Passcode Field */
        .lock-container {
            text-align: center;
            padding: 45px 35px;
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 28px;
            box-shadow: 0 15px 35px rgba(255, 182, 193, 0.25);
            border: 3px dashed #ffb6c1;
            margin-top: 80px;
            margin-bottom: 20px;
        }
        .lock-title { 
            color: #d84b80; 
            font-weight: 800;
            font-size: 2.2rem;
            margin: 0;
        }
        .lock-subtitle { 
            color: #657b9e; 
            font-size: 1rem;
            font-weight: 500;
            margin-top: 8px;
            margin-bottom: 25px;
        }
        /* Overriding Native Streamlit Input Styling Rules */
        div[data-testid="stTextInput"] input {
            border-radius: 25px !important;
            border: 2px solid #add8e6 !important; /* Soft Blue Border */
            background-color: #ffffff !important;
            padding: 12px 20px !important;
            font-size: 1.1rem !important;
            color: #4a1525 !important;
            text-align: center !important;
            box-shadow: 0 4px 10px rgba(173, 216, 230, 0.1) !important;
            transition: all 0.3s ease;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #ffb6c1 !important; /* Changes to Pink on Focus */
            box-shadow: 0 0 12px rgba(255, 182, 193, 0.5) !important;
        }
        /* Hiding Default Streamlit Widget Label Elements */
        div[data-testid="stTextInput"] label {
            display: none !important;
        }
        </style>
        <div class="lock-container">
            <h2 class="lock-title">🌸 Locked Garden 🌸</h2>
            <div class="lock-subtitle">Enter the secret code to talk to Gaurav</div>
            <div style="font-size: 1.2rem; margin-bottom: 20px; letter-spacing: 4px;">✨ 💛 ✨</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Text input configuration automatically masks password entries with dots
    passcode = st.text_input("Secret Code:", type="password", key="secret_gate", placeholder="Type secret passcode here...")
    
    if passcode:
        if passcode.strip().lower() == "cutie pie":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Wrong code, yaar! Try again.")
    st.stop()

# 2. Inject CSS Styles (Pink, Blue, Yellow Floral Aesthetic Theme)
st.markdown(
    """
    <style>
    /* Global Pink, Blue & Soft Yellow Workspace with Floral Vector Dots */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #fffdf0; /* Soft Warm Yellow Hue */
        background-image: 
            radial-gradient(rgba(255, 182, 193, 0.4) 1.5px, transparent 1.5px), 
            radial-gradient(rgba(173, 216, 230, 0.4) 1.5px, transparent 1.5px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
    }

    /* TRIPLE FLORAL BORDER LAYOUT OUTER CONTAINER */
    .floral-outer-frame {
        border: 4px dashed #ffb6c1; /* Pink Outer Layer */
        padding: 6px;
        border-radius: 32px;
        background: transparent;
        margin-bottom: 25px;
    }
    .floral-mid-frame {
        border: 3px solid #fff3cd; /* Soft Yellow Middle Layer */
        padding: 6px;
        border-radius: 26px;
        background: transparent;
    }
    .floral-inner-frame {
        border: 4px dashed #add8e6; /* Blue Inner Layer */
        padding: 20px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(255, 182, 193, 0.1);
    }

    /* Elegant Floral Banner Display Container Card */
    .floral-header {
        text-align: center;
        padding: 24px 15px;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.15);
        border: 2px solid #fff3cd;
        margin-bottom: 30px;
    }
    .floral-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #d84b80 0%, #ffc107 50%, #4a77d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        font-size: 2.2rem;
    }
    .floral-subtitle {
        color: #657b9e;
        font-size: 0.95rem;
        margin-top: 5px;
        font-weight: 500;
    }
    .floral-deco {
        font-size: 1.1rem;
        margin-top: 8px;
        letter-spacing: 6px;
    }

    /* Pink & Blue Styled Message Bubbles Configuration Rules */
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #e3f2fd 0%, #edf7ff 100%) !important; /* Soft Sky Blue for User */
        color: #1a365d !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 4px 15px rgba(173, 216, 230, 0.1);
        padding: 14px 18px !important;
        border: 1px solid rgba(173, 216, 230, 0.3);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #ffe4e1 0%, #fff0f5 100%) !important; /* Pastel Peony Pink for Gaurav */
        color: #4a1525 !important;
        border-radius: 20px 20px 20px 4px !important;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.1);
        padding: 14px 18px !important;
        border: 1px solid rgba(255, 182, 193, 0.3);
    }

    /* Sleek Themed Bottom Chat Input Aesthetics */
    div[data-testid="stChatInput"] {
        border-radius: 35px !important;
        border: 2px solid #ffb6c1 !important;
        background-color: #ffffff !important;
        box-shadow: 0 10px 25px rgba(255, 182, 193, 0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

USER_AVATAR = "🌸"
BOT_AVATAR = "https://freepik.com"

def get_gemini_client():
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        st.error("🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Yo! What's up? Finally you remembered your best friend. Aur bata, what's cooking today? 🤔"}]

st.markdown("""<div class="floral-header"><h1 class="floral-title">Gaurav's Garden</h1><div class="floral-subtitle">Your Hinglish bestie • Available 24/7</div><div class="floral-deco">🌸 ✨ 💛 ✨ 🦋</div></div>""", unsafe_allow_html=True)
st.markdown('<div class="floral-outer-frame"><div class="floral-mid-frame"><div class="floral-inner-frame">', unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

st.markdown('</div></div></div>', unsafe_allow_html=True)

if user_query := st.chat_input("Say something to Gaurav..."):
    st.markdown('<div class="floral-outer-frame"><div class="floral-mid-frame"><div class="floral-inner-frame">', unsafe_allow_html=True)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        try:
            client = get_gemini_client()
            api_contents = []
            
            # --- FIXED FOR-LOOP INDENTATION PORTION ---
            for msg in st.session_state.messages:
                role_type = "user" if msg["role"] == "user" else "model"
                api_contents.append(
                    types.Content(
                        role=role_type, 
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )
            
            system_instruction = (
                "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                "CRUCIAL: Read the user's text carefully and answer their exact question contextually. "
                "Never use hardcoded greeting lists or switch topics randomly. Respond dynamically. "
                "Chat casually using informal internet slang and short sentences like a text message. "
                "You speak naturally in a mix of Hindi and English (Hinglish). Use casual terms like 'Bhai', "
                "'Yaar', 'Bro', 'Chill mar', and 'tension mat le'. "
                "Do NOT use Gujarati phrases like 'Kem cho' or 'Majama' in every sentence. Only use them rarely "
                "if explicitly asked about Gujarati or if it fits a niche joke naturally. "
                "Crucially, you must use emojis effectively: include exactly ONE or a maximum of TWO highly relevant emojis "
                "per turn. Do not spam arrays of emojis under any circumstance."
            )
            response = client.models.generate_content(
                model="gemini-3.5-flash", 
                contents=api_contents, 
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.4)
            )
            bot_response = response.text
        except Exception as e:
            fallback_options = [
                "Bhai, thoda busy hoon! Mummy ne kaam saupa hai. 😂", 
                "Arey yaar, internet bohot slow chal raha hai yahan... chill mar! ☕", 
                "Bro, phone ki battery khatam hone wali hai! Late text karu? 😉", 
                "Tension mat le bhai, main yahin hoon. Thoda breaks chahiye! 😂"
            ]
            bot_response = random.choice(fallback_options)
            
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.60)
                message_placeholder.markdown(full_response)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.markdown('</div></div></div>', unsafe_allow_html=True)
    st.rerun()
