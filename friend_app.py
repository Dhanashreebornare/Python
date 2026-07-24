import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Vibe with Gaurav", page_icon="🌸", layout="centered")

# 2. Main Lounge UI Core Styles (Dark Purple-Pink Theme, Cloud Bubbles, Black Text)
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
    .lounge-subtitle { color: #444444; font-size: 0.95rem; margin-top: 5px; font-weight: 500; margin-bottom: 12px; }
    
    /* Smooth CSS Left-to-Right Slow Slide Animation Keyframe */
    @keyframes smoothCloudPop {
        0% { opacity: 0; transform: translateX(-30px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    /* Standard Instant Entry for User Chat Bubbles */
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 25px 25px 5px 25px !important; 
        padding: 14px 20px !important;
        border: 1px solid rgba(0, 0, 0, 0.04) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06) !important;
        animation: none !important;
    }
    
    /* Slower 12-second Left-to-Right Glide Effect EXCLUSIVELY for Gaurav's Chat Bubbles */
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background-color: #f5ecef !important; 
        color: #000000 !important;
        border-radius: 25px 25px 25px 5px !important; 
        padding: 14px 20px !important;
        border: 1px solid rgba(0, 0, 0, 0.04) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06) !important;
        animation: smoothCloudPop 12s ease-out forwards !important;
    }
    
    /* Minimalist Cloud Text Input Box */
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
BOT_AVATAR = "👦🏻"  

def get_gemini_client():
    if "gemini_client" not in st.session_state:
        if "GEMINI_API_KEY" in st.secrets:
            api_key_val = st.secrets["GEMINI_API_KEY"]
        elif os.environ.get("GEMINI_API_KEY"):
            api_key_val = os.environ.get("GEMINI_API_KEY")
        else:
            st.error("🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
            st.stop()
        st.session_state.gemini_client = genai.Client(api_key=api_key_val)
    return st.session_state.gemini_client

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
        <div style="font-size: 1.2rem; letter-spacing: 4px;">🌸  ✨  🪻  ✨  🌸</div>
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
        
        api_contents = [types.Content(role="user" if msg["role"] == "user" else "model", parts=[types.Part.from_text(text=msg["content"])]) for msg in st.session_state.messages]
        system_instruction = "You are Gaurav, a funny, witty, deeply loving, and loyal close best friend. CRUCIAL: Read the user's text carefully and answer their exact question contextually. Never use hardcoded greeting lists or switch topics randomly. Respond dynamically. Chat casually using informal internet slang and short sentences like a text message. You speak naturally in a mix of Hindi and English (Hinglish). Use casual terms like 'Bhai', 'Yaar', 'Bro', 'Chill mar', 'tension mat le', and 'Mast'. Crucially, you must use emojis in a highly optimistic, joyful, and supportive way to lift the user's spirits and spread positive vibes. Include exactly ONE or a maximum of TWO highly relevant, bright, happy emojis per turn. Do not spam arrays of emojis."
        
        start_time = time.time()
        
        # --- CLEAN UNIFIED SPINNER CONTEXT ---
        with st.spinner("Gaurav is typing..."):
            try:
                response_data = get_gemini_client().models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=api_contents, 
                    config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.5)
                )
                bot_response = response_data.text
            except Exception as e:
                # Witty, caring, and realistic backup lines when the system goes offline
                system_fallbacks = [
                    "Bhai, thoda system issue chhe yaar! Network haali gayo chhe dimaag mathi. 🥲",
                    "Arey bro, network j locha maari rahyu chhe! Tension mat le, thodi vaar ma vaat kariye. ☕",
                    "Gaurav no dimaag thakyo chhe... lag chhe pachhad thi server j down thadh gayo! 😂",
                    "Yaar, Mummy ne bolavyo kaam mate, etla ma server j bandh thai gayo! 🏃‍♂️",
                    "Tension shu kaam leve chhe bhai? Thodo technical issue chhe, haveli par aav vaat kariye! 😉"
                ]
                # Combine a funny line with the real hidden error trace for clean debugging
                bot_response = f"{random.choice(system_fallbacks)}\n\n*(Debug Trace: {str(e)})*"
            
            # --- CRUSH-PROOF COUNTDOWN DELAY ---
            elapsed_time = time.time() - start_time; remaining_time = max(0.0, 5.0 - elapsed_time)
            if remaining_time > 0: time.sleep(remaining_time)

        # --- WORD TYPEWRITER STREAMING ANIMATION ---
        if bot_response:
            word_list = bot_response.split(); word_delay = max(0.02, 5.0 / max(1, len(word_list)))
            for index, word in enumerate(word_list):
                full_response += word + " "; time.sleep(word_delay); message_placeholder.markdown(full_response.strip())
            
            message_placeholder.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()
