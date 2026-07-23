import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Chat with Gaurav", page_icon="🪻", layout="centered")

# 2. Inject CSS Styles Privately (Frosty Cold-Tone Floral Theme)
st.markdown(
    """
    <link rel="preconnect" href="https://googleapis.com">
    <link rel="preconnect" href="https://gstatic.com" crossorigin>
    <link href="https://googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Frosty Cold-Tone Background with Floral Accents */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #f1f4f9;
        background-image: 
            radial-gradient(rgba(145, 175, 255, 0.25) 1px, transparent 1px), 
            radial-gradient(rgba(180, 160, 240, 0.2) 1.5px, transparent 1.5px);
        background-size: 40px 40px;
        background-position: 0 0, 20px 20px;
    }
    
    /* Elegant Minimalist Floral Header Widget */
    .floral-header {
        text-align: center;
        padding: 25px 15px;
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(145, 175, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 35px;
    }
    
    .floral-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #4552a1 0%, #6d78c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        font-size: 2.1rem;
    }
    
    .floral-subtitle {
        color: #7b84bf;
        font-size: 0.95rem;
        margin-top: 5px;
        font-weight: 500;
    }
    
    .floral-deco {
        font-size: 1.2rem;
        color: #9aa5e3;
        margin-top: 6px;
        letter-spacing: 4px;
    }

    /* Muted Cold-Tone Chat Bubble Structuring */
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #e3e9ff 0%, #edf1ff 100%) !important; /* Ice Powder Blue for User */
        color: #1c2554 !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 4px 15px rgba(145, 175, 255, 0.05);
        padding: 14px 18px !important;
        border: 1px solid rgba(145, 175, 255, 0.15);
    }
    
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #eae3ff 0%, #f3f0ff 100%) !important; /* Muted Frost Lavender for Gaurav */
        color: #2b1f47 !important;
        border-radius: 20px 20px 20px 4px !important;
        box-shadow: 0 4px 15px rgba(180, 160, 240, 0.05);
        padding: 14px 18px !important;
        border: 1px solid rgba(180, 160, 240, 0.15);
    }
    
    /* Sleek User Chat Input Aesthetics */
    div[data-testid="stChatInput"] {
        border-radius: 35px !important;
        border: 1px solid rgba(145, 175, 255, 0.25) !important;
        background-color: #ffffff !important;
        box-shadow: 0 10px 25px rgba(145, 175, 255, 0.06) !important;
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

# 3. Render Minimalist Cold Floral Header Banner
st.markdown(
    """
    <div class="floral-header">
        <h1 class="floral-title">Gaurav's Garden</h1>
        <div class="floral-subtitle">Your Hinglish bestie • Available 24/7</div>
        <div class="floral-deco">🪻 ✧ 🪻 ✧ 🪻</div>
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
                
        # Animate final variables downstream (10x slower tracking rhythm)
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.60)  # Paced precisely 10 times slower than 0.06s
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
