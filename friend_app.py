import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Chat with Gaurav", page_icon="🌸", layout="centered")

# 2. Inject CSS Styles Privately (Pink & Blue Floral Aesthetic Theme)
st.markdown(
    """
    <link rel="preconnect" href="https://googleapis.com">
    <link rel="preconnect" href="https://gstatic.com" crossorigin>
    <link href="https://googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Global Pink & Blue Gradient App Workspace with Floral Vector Dots */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #fff0f5;
        background-image: 
            radial-gradient(rgba(255, 182, 193, 0.4) 1.5px, transparent 1.5px), 
            radial-gradient(rgba(173, 216, 230, 0.4) 1.5px, transparent 1.5px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
    }
    
    /* Elegant Floral Banner Display Container Card */
    .floral-header {
        text-align: center;
        padding: 24px 15px;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 30px;
    }
    
    .floral-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #d84b80 0%, #4a77d4 100%);
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
        border: 1px solid rgba(255, 182, 193, 0.4) !important;
        background-color: #ffffff !important;
        box-shadow: 0 10px 25px rgba(255, 182, 193, 0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Clean Native Avatars
USER_AVATAR = "🌸"
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

# 3. Render Minimalist Pink & Blue Floral Header Banner Widget
st.markdown(
    """
    <div class="floral-header">
        <h1 class="floral-title">Gaurav's Garden</h1>
        <div class="floral-subtitle">Your Hinglish bestie • Available 24/7</div>
        <div class="floral-deco">🌸 ✨ 🌸 ✨ 🌸</div>
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
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
        
    st.session_state.messages.append({"role": "user", "content": user_query})
    
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
                
                # RESTRUCTURED: Simplified history extraction format for better model tracking
                api_contents = []
                for msg in st.session_state.messages:
                    role_type = "user" if msg["role"] == "user" else "model"
                    api_contents.append({"role": role_type, "parts": [{"text": msg["content"]}]})
                
                system_instruction = (
                    "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                    "You must answer accurately and address the user's statements directly. Do not go off-topic. "
                    "Chat casually using informal internet slang and short sentences like a text message. "
                    "You speak naturally in a mix of Hindi and English (Hinglish). Use casual terms like 'Bhai', "
                    "'Yaar', 'Bro', 'Chill mar', and 'tension mat le'. "
                    "Do NOT use Gujarati phrases like 'Kem cho' or 'Majama' in every sentence. Only use them rarely "
                    "if explicitly asked about Gujarati or if it fits a niche joke naturally. "
                    "Crucially, use emojis effectively: include exactly ONE or a maximum of TWO highly relevant emojis "
                    "per turn. Do not spam arrays of emojis under any circumstance."
                )
                
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=api_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,  # LOWERED: Kept lower to prevent irrelevant branching logic
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
                
        # Animate final variables downstream (10x slower tracking rhythm)
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.60)  # Paced precisely 10 times slower than 0.06s
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
