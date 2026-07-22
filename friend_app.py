import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# Configure the web page
st.set_page_config(page_title="Chat with Gaurav", page_icon="🪻")

# 🌸 Premium Floral Aesthetic Design & Headings via CSS Injection
st.markdown(
    """
    <style>
    /* 1. Backdrop Wallpaper of soft, aesthetic periwinkle flowers */
    .stApp {
        background-color: #f6f7fb;
        background-image: 
            radial-gradient(rgba(174, 182, 255, 0.25) 1.5px, transparent 1.5px),
            radial-gradient(rgba(235, 186, 255, 0.2) 2px, transparent 2px);
        background-size: 45px 45px;
        background-position: 0 0, 22.5px 22.5px;
    }
    
    /* 2. Frosted Glass Modern Title Header Container */
    .aesthetic-header {
        text-align: center;
        padding: 24px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(162, 168, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.6);
        margin-bottom: 20px;
    }
    
    /* 3. Redesigned Heading Typography */
    .aesthetic-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #5058df 0%, #a46ae8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        font-size: 2.3rem;
    }
    
    .aesthetic-subtitle {
        color: #787ec2;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    
    /* 4. Elegant Bottom Chat-Input Customization */
    div[data-testid="stChatInput"] {
        border-radius: 30px !important;
        border: 1px solid rgba(162, 168, 255, 0.4) !important;
        box-shadow: 0 4px 15px rgba(162, 168, 255, 0.08) !important;
    }
    
    /* 5. Custom Themed Periwinkle Chat Bubbles */
    div[data-testid="stChatMessageContent"] {
        border-radius: 16px !important;
        padding: 12px 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Safe avatar loading
USER_AVATAR = "periwinkle.png" if os.path.exists("periwinkle.png") else "🪻"
BOT_AVATAR = "gaurav.jpg" if os.path.exists("gaurav.jpg") else "👦"


# 🔐 Initialize Google Gemini API client securely
def get_gemini_client():
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        st.error("🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)


# Render Redesigned Premium Heading
st.markdown(
    """
    <div class="aesthetic-header">
        <h1 class="aesthetic-title">🌸 Gaurav's Garden 🪻</h1>
        <div class="aesthetic-subtitle">Your Hinglish & Gujlish bestie • Available 24/7 ☕✨</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! Kem cho? Finally you remembered your best friend. Aur bata, what's up today?",
        }
    ]

# Display previous chat messages from history
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Say something to Gaurav..."):

    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    # Add user message to local history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Prepare response container
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        bot_response = ""

        clean_input = user_query.lower().strip()

        # 🛑 STEP 1: Local Python check for simple greetings to save quota
        if any(word in clean_input for word in ["hey", "hello", "hi", "yo", "kem cho", "ram ram", "namaste"]):
            bot_response = random.choice([
                "Yo! What's cracking, my friend? Kem cho?",
                "Kevo che bhai? What's up today?",
                "Yo! Finally you remembered your best friend. Aur bata, shu khabar?"
            ])
        elif any(word in clean_input for word in ["bye", "see ya", "aavjo", "chalo", "chal"]):
            bot_response = "Don't leave me alone, yaar! Just kidding, aavjo! Take care, bro."

        # 🌐 STEP 2: Call API if it's a complex message
        if not bot_response:
            try:
                client = get_gemini_client()

                api_contents = []
                for msg in st.session_state.messages:
                    role_type = "user" if msg["role"] == "user" else "model"
                    api_contents.append(
                        types.Content(role=role_type, parts=[types.Part(text=msg["content"])])
                    )

                # 🛠️ FIXED: Reduced emoji priority instructions
                system_instruction = (
                    "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                    "You must chat casually. Use informal internet slang and abbreviations. "
                    "Do NOT use unnecessary or spammy emojis. Only use a single emoji if it is absolutely necessary "
                    "to convey a specific emotion (like a laugh 😂 or joke), but keep most sentences plain text. "
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

            # 🛠️ STEP 3: Fallback mechanism if the API is exhausted (429 Error)
            except Exception as e:
                bot_response = random.choice([
                    "Bhai, thoda busy hoon! Mummy ne kaam saupa hai, thodi der baad baat karte hain!",
                    "Arey yaar, internet bohot slow chal raha hai yahan... Badhu saru thai jase, chill mar!",
                    "Bro, phone ki battery khatam hone wali hai! Tarat j jalsa kar ne yaar, late text karu!",
                    "Tension mat le bhai, main yahin hoon. Par abhi thoda dimaag thak gaya hai, breaks chahiye!"
                ])

        # ⏱️ STEP 4: Animate output (Slower word pacing)
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.50)  # Real-time slow typing pace
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Save to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
