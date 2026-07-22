import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# Configure the web page
st.set_page_config(page_title="Chat with Gaurav", page_icon="🪻")

# 🌸 Custom CSS to inject a floral backdrop and aesthetic theme
st.markdown(
    """
    <style>
    /* Add a subtle, beautiful floral background pattern to the whole app */
    .stApp {
        background-image: radial-gradient(rgba(230, 230, 250, 0.4) 1px, transparent 0),
                          radial-gradient(rgba(204, 204, 255, 0.3) 1px, transparent 0);
        background-size: 40px 40px;
        background-position: 0 0, 20px 20px;
        background-color: #fbfcff;
    }
    
    /* Title container styling */
    .title-container {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.75);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(204, 204, 255, 0.2);
        border: 1px solid rgba(204, 204, 255, 0.4);
        margin-bottom: 25px;
    }
    
    /* Floral divider aesthetic */
    .floral-divider {
        text-align: center;
        color: #9aa0e6;
        font-size: 20px;
        margin: 10px 0;
        letter-spacing: 5px;
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


# Styled App Title Header
st.markdown(
    """
    <div class="title-container">
        <h1 style='color: #5c62d6; margin: 0;'>🪻 Chat with Gaurav 🪻</h1>
        <p style='color: #7b81db; font-style: italic; margin: 5px 0 0 0;'>
            Your Hinglish & Gujlish best friend • Wrapped in Periwinkles 🌸
        </p>
    </div>
    <div class="floral-divider">🌸✨🪻✨🌸</div>
    """, 
    unsafe_allow_html=True
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! Kem cho? Finally you remembered your best friend. 👋 Aur bata, what's up today? 🤔🪻",
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
                "Yo! What's cracking, my friend? Kem cho? 😁👋🪻",
                "Kevo che bhai? What's up today? 😎🌸",
                "Yo! Finally you remembered your best friend. Aur bata, shu khabar? 😉🪻"
            ])
        elif any(word in clean_input for word in ["bye", "see ya", "aavjo", "chalo", "chal"]):
            bot_response = "Don't leave me alone, yaar! 😢 Just kidding, aavjo! Take care, bro! 👋⚡🌸"

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

                system_instruction = (
                    "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                    "You must chat casually. Use informal internet slang, abbreviations, and plenty of emojis. "
                    "Crucially, you must heavily sprinkle relevant, expressive emojis throughout your messages "
                    "(e.g., 😂, 💀, 🤣, 🤦‍♂️, 🤫, 👀, ☕, 🔥) wherever necessary to emphasize your jokes and emotions. "
                    "Since the chat has a periwinkle flower aesthetic, occasionally tease the user about flowers "
                    "or throw in flower emojis (🪻, 🌸, 🌼) when matching your sarcastic tone. "
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
                    "Bhai, thoda busy hoon! 🤫 Mummy ne kaam saupa hai (yaad aaya, ghar ke pouf saaf karne hain 🌸), thodi der baad baat karte hain! 😂🏃‍♂️",
                    "Arey yaar, internet bohot slow chal raha hai yahan... 💀 Badhu saru thai jase, chill mar aur thoda phool súngh! 🪻☕",
                    "Bro, phone ki battery khatam hone wali hai! 🔋 Tarat j jalsa kar ne yaar, late text karu! 😉🌸",
                    "Tension mat le bhai, main yahin hoon. Par abhi thoda dimaag thak gaya hai, fresh air aur flowers chahiye! 🤦‍♂️😂🪻"
                ])

        # ⏱️ STEP 4: Animate output (Half speed)
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.50)  # ⏱️ Increased delay to 0.50s per word for a slower typing experience
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Save to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
