import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Chat with Gaurav", page_icon="🪻", layout="wide")

# 2. Inject CSS Styles Privately
st.markdown(
    """
    <link rel="preconnect" href="https://googleapis.com">
    <link rel="preconnect" href="https://gstatic.com" crossorigin>
    <link href="https://googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Global Background Accent Settings */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #f6f8fc;
        background-image: radial-gradient(rgba(145, 157, 255, 0.15) 1.2px, transparent 1.2px);
        background-size: 32px 32px;
    }
    
    /* Left Profile Card Styling */
    .profile-card {
        background: #ffffff;
        border: 1px solid rgba(145, 157, 255, 0.25);
        border-radius: 24px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(145, 157, 255, 0.05);
        position: sticky;
        top: 20px;
    }
    
    .profile-avatar {
        font-size: 4.5rem;
        margin-bottom: 10px;
        filter: drop-shadow(0 8px 12px rgba(145, 157, 255, 0.2));
    }
    
    .profile-name {
        font-weight: 800;
        color: #21255e;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .profile-tagline {
        color: #787fb5;
        font-size: 0.9rem;
        margin-top: 4px;
        font-weight: 500;
    }
    
    .status-badge {
        display: inline-block;
        background-color: #e3fcef;
        color: #006644;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-box {
        margin-top: 25px;
        text-align: left;
        background: #f8fafc;
        padding: 15px;
        border-radius: 16px;
        border: 1px solid #edf2f7;
    }
    
    .info-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #4d55cc;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    
    .info-text {
        font-size: 0.85rem;
        color: #4a5568;
        line-height: 1.4;
    }

    /* Right Chat Message Bubble Formatting Rules */
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #e2e5ff 0%, #ebedff 100%) !important;
        color: #21255e !important;
        border-radius: 24px 24px 4px 24px !important;
        box-shadow: 0 4px 20px rgba(145, 157, 255, 0.08);
        padding: 16px 20px !important;
        border: 1px solid rgba(145, 157, 255, 0.2);
    }
    
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #f1e6ff 0%, #f9f2ff 100%) !important;
        color: #422a63 !important;
        border-radius: 24px 24px 24px 4px !important;
        box-shadow: 0 4px 20px rgba(220, 180, 255, 0.08);
        padding: 16px 20px !important;
        border: 1px solid rgba(220, 180, 255, 0.2);
    }
    
    /* Sleek User Chat Input Aesthetics */
    div[data-testid="stChatInput"] {
        border-radius: 35px !important;
        border: 1px solid rgba(145, 157, 255, 0.3) !important;
        background-color: #ffffff !important;
        box-shadow: 0 12px 30px rgba(145, 157, 255, 0.08) !important;
        padding-left: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Clean Native Avatars
USER_AVATAR = "🪻"
BOT_AVATAR = "👦"

# Initialize Google Gemini API securely
def get_gemini_client():
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        st.error("🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

# Setup initial structural session tracking state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! Kem cho? Finally you remembered your best friend. Aur bata, what's up today? 🤔",
        }
    ]

# 3. Explicit 1:3 structural column split mapping layout
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    # Render Left Layout Visual Profile Component Card
    st.markdown(
        """
        <div class="profile-card">
            <div class="profile-avatar">👦</div>
            <h1 class="profile-name">Gaurav</h1>
            <div class="profile-tagline">Hinglish & Gujlish Bestie</div>
            <div class="status-badge">● Active Now</div>
            <div class="info-box">
                <div class="info-title">Vibe Check</div>
                <div class="info-text">Funny, wildly sarcastic, fiercely loyal, and ready to listen. 💯</div>
            </div>
            <div class="info-box">
                <div class="info-title">Current Jam</div>
                <div class="info-text">🎵 Pasoori Nu (On Repeat)</div>
            </div>
            <div class="info-box">
                <div class="info-title">Favorite Words</div>
                <div class="info-text"><i>"Bhai", "Yaar", "Majama", "Tension mat le", "Jalsa kar"</i></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Construct right view column historical logs cleanly
    for message in st.session_state.messages:
        avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Wait for execution engine input signals
    if user_query := st.chat_input("Say something to Gaurav..."):
        # Mount incoming input query to layout interface view tracking
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_query)
            
        # Register values to data session storage matrix
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Open chatbot layout placeholder windows
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""
            bot_response = ""
            clean_input = user_query.lower().strip()
            
            # Layer A Engine Sorting Logic: Fast Greeting Interceptor Validation
            if any(word in clean_input for word in ["hey", "hello", "hi", "yo", "kem cho", "ram ram", "namaste"]):
                bot_response = random.choice([
                    "Yo! What's cracking, my friend? Kem cho? 🙌",
                    "Kevo che bhai? What's up today? 😎",
                    "Yo! Finally you remembered your best friend. Aur bata, shu khabar? 😉"
                ])
            elif any(word in clean_input for word in ["bye", "see ya", "aavjo", "chalo", "chal"]):
                bot_response = "Don't leave me alone, yaar! Just kidding, aavjo! Take care, bro. 👋"
                
            # Layer B Engine Sorting Logic: Live Global Gemini Target Content Evaluator
            if not bot_response:
                try:
                    client = get_gemini_client()
                    api_contents = []
                    for msg in st.session_state.messages:
                        role_type = "user" if msg["role"] == "user" else "model"
                        api_contents.append(
                            types.Content(role=role_type, parts=[types.Part(text=msg["content"])])
                        )
                    
                    system_instruction = (
                        "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                        "You must chat casually. Use informal internet slang and abbreviations. "
                        "Crucially, you MUST include exactly ONE highly relevant emoji at the end of your response "
                        "or inside your message (e.g., 😂 if making a joke, 💀 if reacting to something crazy, "
                        "🔥 for something cool, or 🤦‍♂️ for a facepalm moment) to sound like a natural human friend. "
                        "Do not leave the message as completely plain text, but do not spam multiple emojis either. "
                        "You speak naturally in a mix of English, Hinglish (Hindi + English), and Gujlish (Gujarati + English). "
                        "Frequently use local friendly slang terms like 'Bhai', 'Yaar', 'Bro', 'Kem cho', 'Majama', "
                        "'Shu vaat che', 'Chal ne', 'Jalsa kar', 'tension mat le'. "
                        "Keep your responses relatively punchy and short, exactly like a friend texting over WhatsApp. "
                        "Never sound like a formal corporate AI assistant or robot."
                    )
                    
                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=api_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=1.0,
                        ),
                    )
                    bot_response = response.text
                    
                except Exception as e:
                    bot_response = random.choice([
                        "Bhai, thoda busy hoon! Mummy ne kaam saupa hai, thodi der baad baat karte hain! 🏃‍♂️",
