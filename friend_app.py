import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# 1. Global Page Layout Configurations
st.set_page_config(page_title="Vibe with Gaurav..", page_icon="🌸", layout="centered")

# Initialize global authentication tracking state safely from Streamlit Sessions
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- PASSCODE AUTHENTICATION LOCK ---
if not st.session_state.authenticated:
    st.markdown("""<style>.stApp {font-family: 'Inter', sans-serif !important; background: linear-gradient(135deg, #2d1124 0%, #4a1539 100%) !important; color: #000000 !important;} .lock-container {text-align: center; padding: 45px 35px; background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 24px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.5); margin-top: 40px; margin-bottom: 20px;} div[data-testid="stTextInput"] input {border-radius: 25px !important; border: 1px solid rgba(0, 0, 0, 0.2) !important; background-color: #ffffff !important; padding: 12px 20px !important; font-size: 1.1rem !important; color: #000000 !important; text-align: center !important; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important; transition: all 0.3s ease;} div[data-testid="stTextInput"] input:focus {border-color: #000000 !important; box-shadow: 0 0 12px rgba(0, 0, 0, 0.2) !important;} div[data-testid="stTextInput"] label {display: none !important;} footer {visibility: hidden !important;}</style>""", unsafe_allow_html=True)
    st.markdown("""<div class="lock-container"><h2 style="color: #000000; font-weight: 800; font-size: 2.2rem; margin: 15px 0 0 0;">Vibe with Gaurav...</h2><div style="color: #444444; font-size: 1rem; font-weight: 500; margin-top: 8px; margin-bottom: 12px;">Verify code to connect securely</div><div style="font-size: 1.2rem; letter-spacing: 4px; margin-bottom: 5px;">🌸 ✨ 🪻 ✨ 🌸</div></div>""", unsafe_allow_html=True)
    
    passcode_input = st.text_input("Secret Code:", type="password", key="secret_gate", placeholder="Enter passcode here...")
    if passcode_input:
        master_passcode = None
        # Explicit check for Streamlit Cloud Secrets storage keys
        if "SECRET_PASSCODE" in st.secrets:
            master_passcode = st.secrets["SECRET_PASSCODE"]
        else:
            master_passcode = os.environ.get("SECRET_PASSCODE")
            
        if master_passcode and passcode_input.strip().lower() == str(master_passcode).strip().lower():
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Invalid entry, buddy! Try again.")
            st.stop()

# --- MAIN LOUNGE APP (Loads ONLY when authenticated) ---
else:
                                # 2. Main Lounge UI Core Styles (Comic Clouds, Plus Jakarta Sans Body & Artistic Title)
    st.markdown(
        """
        <style>
        @import url('https://googleapis.com');

        .stApp {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
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
            font-family: 'Caveat', cursive !important; /* Artistic handwriting font */
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #000000 !important;
            margin: 0;
            font-size: 3.5rem; /* Increased size because handwriting fonts run slightly smaller */
            line-height: 1.1;
        }
        .lounge-subtitle {
            color: #444444;
            font-size: 0.95rem;
            margin-top: 5px;
            font-weight: 500;
            margin-bottom: 12px;
        }
        @keyframes smoothCloudPop {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        /* --- GLOBAL CHAT FLEX LAYOUT --- */
        div[data-testid="stChatMessage"] {
            display: flex !important;
            width: 100% !important;
            background-color: transparent !important;
            gap: 12px !important;
            align-items: flex-start !important;
        }
        
        /* --- USER CHAT BUBBLES (AVATAR RIGHT, COMIC BOOK TALK-CLOUD) --- */
        div[data-testid="stChatMessage"]:nth-child(even) {
            flex-direction: row-reverse !important;
        }
        div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
            background-color: rgba(227, 242, 253, 0.95) !important;
            color: #111111 !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            border-radius: 30px 15px 30px 20px !important;
            padding: 14px 24px !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
            text-align: left !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            animation: none !important;
        }
        
        /* --- GAURAV CHAT BUBBLES (AVATAR LEFT, COMIC BOOK TALK-CLOUD) --- */
        div[data-testid="stChatMessage"]:nth-child(odd) {
            flex-direction: row !important;
        }
        div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
            background-color: #ffffff !important;
            color: #111111 !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            border-radius: 15px 30px 20px 30px !important;
            padding: 14px 24px !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
            border: 1px solid rgba(0, 0, 0, 0.02) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
            text-align: left !important;
            margin-right: auto !important;
            margin-left: 0 !important;
            animation: smoothCloudPop 0.3s ease-out forwards !important;
        }
        
        div[data-testid="stChatInput"] {
            border-radius: 35px !important;
            background-color: #ffffff !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
            border: none !important;
        }
        div[data-testid="stChatInput"] textarea {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: #000000 !important;
            font-size: 1.05rem !important;
        }
        footer {
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
                "content": "Yo! What's up? Finally you remembered your best friend! Aur bata, what's cooking today? 🤔",
            }
        ]

    # 3. Main Header Card (Loads safely inside the authenticated block)
    st.markdown(
        """
        <div class="lounge-header">
            <h1 class="lounge-title">Vibe with Gaurav.</h1>
            <div class="lounge-subtitle">Your Hinglish bestie • Available 24/7</div>
            <div style="font-size: 1.2rem; letter-spacing: 4px;">🌸 ✨ ✨ 🌸</div>
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
    # Render user query instantly
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Assistant Response Generation
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        api_contents = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            )
            for msg in st.session_state.messages
        ]
        
        system_instruction = (
            "You are Gaurav, a funny, highly enthusiastic, witty, and loyal best friend. "
            "CRUCIAL: Read the user's text carefully and answer their exact question contextually. "
            "Chat casually using informal internet slang and short sentences like an instant message. "
            "You talk mostly in Hinglish (a mix of Hindi and English). Use casual terms like "
            "'Bhai', 'Yaar', 'Bro', 'Chill mar', 'tension mat le', and 'Mast'. "
            "Because you are Gujarati, you only RARELY drop a small Gujarati expression when you get "
            "super excited, shocked, or hyped up (e.g., 'Su vaat che!', 'Majama', or 'Locho thai gayo'), but keep 90% of the conversation in Hinglish. "
            "Crucially, you must use emojis in a highly optimistic, joyful, and supportive way to lift the user's spirits. "
            "Include exactly ONE or a maximum of TWO highly relevant, bright, happy emojis per turn."
        )

        try:
            # 1. Call the streaming API variant
            response_stream = get_gemini_client().models.generate_content_stream(
                model="gemini-3.5-flash",
                contents=api_contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.6)
            )
            
            # 2. Helper generator to yield text fragments instantly as they land
            def stream_chunks():
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            # 3. Stream out onto screen dynamically with no artificial delays
            bot_response = st.write_stream(stream_chunks())
            
        except Exception as e:
            system_fallbacks = [
                "Bhai, thoda system issue chhe yaar! Network haali gayo chhe dimaag mathi. 🥲",
                "Arey bro, network j locha maari rahyu chhe! Tension mat le, thodi vaar ma vaat kariye. ☕",
                "Gaurav no dimaag thakyo chhe... lag chhe pachhad thi server j down thadh gayo! 😂",
                "Yaar, wife ne bolavyo kaam mate, etla ma server j bandh thai gayo! 🏃‍♂️",
                "Tension shu kaam leve chhe bhai? Thodo technical issue chhe, haveli par aav vaat kariye! 😉"
            ]
            bot_response = f"{random.choice(system_fallbacks)}\n\n*(Debug Trace: {str(e)})*"
            st.markdown(bot_response)

        # Save generated content straight to state array
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
