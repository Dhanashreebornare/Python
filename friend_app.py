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
    
    /* Left Sidebar Profile Card Styling */
    .profile-card {
        background: #ffffff;
        border: 1px solid rgba(145, 157, 255, 0.25);
        border-radius: 20px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(145, 157, 255, 0.05);
        margin-bottom: 15px;
    }
    
    .profile-avatar {
        font-size: 3.5rem;
        margin-bottom: 5px;
    }
    
    .profile-name {
        font-weight: 800;
        color: #21255e;
        font-size: 1.6rem;
        margin: 0;
    }
    
    .profile-tagline {
        color: #787fb5;
        font-size: 0.85rem;
        margin-top: 2px;
        font-weight: 500;
    }
    
    .status-badge {
        display: inline-block;
        background-color: #e3fcef;
        color: #006644;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 8px;
        text-transform: uppercase;
    }
    
    .info-box {
        margin-top: 15px;
        text-align: left;
        background: #f8fafc;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid #edf2f7;
    }
    
    .info-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #4d55cc;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .info-text {
        font-size: 0.8rem;
        color: #4a5568;
        line-height: 1.3;
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Clean Native Avatars
USER_AVATAR = "🪻"
BOT_AVATAR = "👦"

# Initialize Google Gemini API client securely
def get_gemini_client():
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        st.error("🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

# Setup initial session tracking state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! What's up? Finally you remembered your best friend. Aur bata, what's cooking today? 🤔",
        }
    ]

# 3. Sidebar Anchor Layout Fix (Ensures correct placement of the input box layout)
with st.sidebar:
    st.markdown(
        """
        <div class="profile-card">
            <div class="profile-avatar">👦</div>
            <h1 class="profile-name">Gaurav</h1>
            <div class="profile-tagline">Your Hinglish Bestie</div>
            <div class="status-badge">● Active Now</div>
            
            <div class="info-box">
                <div class="info-title">Vibe Check</div>
                <div class="info-text">Funny, mildly sarcastic, fiercely loyal, and a solid listener. 💯</div>
            </div>
            
            <div class="info-box">
                <div class="info-title">Current Jam</div>
                <div class="info-text">🎵 Pasoori Nu (On Repeat)</div>
            </div>
            
            <div class="info-box">
                <div class="info-title">Catchphrases</div>
                <div class="info-text"><i>"Bhai", "Yaar", "Bro", "Chill mar", "Tension mat le"</i></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 4. Construct Main Feed Area Container
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
        if any(word in clean_input for word in ["hey", "hello", "hi", "yo", "ram ram", "namaste"]):
            bot_response = random.choice([
                "Yo! What's cracking, my friend? All good? 🙌",
                "Kya chal raha hai bhai? What's up today? 😎",
                "Yo! Finally you remembered your best friend. Aur bata, sab badhiya? 😉"
            ])
        elif any(word in clean_input for word in ["bye", "see ya", "chalo", "chal", "tata"]):
            bot_response = "Don't leave me alone, yaar! Just kidding, chal take care, bro. 👋"
            
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
                    "You must chat casually using informal internet slang and short sentences like a text message. "
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
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=1.0,
                    ),
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
                
        # Animate final layout variables downstream onto interface screens
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.06)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        # Permanently append values back into historical arrays
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
