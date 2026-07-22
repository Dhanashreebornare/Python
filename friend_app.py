import os
import random
import time
import streamlit as st
from google import genai
from google.genai import types

# Configure the web page
st.set_page_config(page_title="Chat with Gaurav", page_icon="💬")

# Safe avatar loading
USER_AVATAR = (
    "periwinkle.png" if os.path.exists("periwinkle.png") else "🪻"
)
BOT_AVATAR = "gaurav.jpg" if os.path.exists("gaurav.jpg") else "👦"


# 🔐 Initialize Google Gemini API client securely
def get_gemini_client():
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        st.error(
            "🔑 API Key missing! Please add 'GEMINI_API_KEY' to your Streamlit Secrets."
        )
        st.stop()
    return genai.Client(api_key=api_key)


# App Title
st.title("🤖 Chat with Gaurav")
st.subheader(
    "Your multilingual best friend, optimized for low API usage! ⚡"
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! Kem cho? Finally you remembered your best friend. 👋 Aur bata, what's up today? 🤔",
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

        # 🛑 STEP 1: Local Python check to completely bypass the API for simple texts
        clean_input = user_query.lower().strip()

        if any(
            word in clean_input
            for word in [
                "hey",
                "hello",
                "hi",
                "yo",
                "kem cho",
                "ram ram",
                "namaste",
            ]
        ):
            bot_response = random.choice(
                [
                    "Yo! What's cracking, my friend? Kem cho? 😁👋",
                    "Kevo che bhai? What's up today? 😎",
                    "Yo! Finally you remembered your best friend. Aur bata, shu khabar? 😉",
                ]
            )
        elif any(
            word in clean_input
            for word in ["bye", "see ya", "aavjo", "chalo", "chal"]
        ):
            bot_response = "Don't leave me alone, yaar! 😢 Just kidding, aavjo! Take care, bro! 👋⚡"

        # 🌐 STEP 2: Only call the API if local checks didn't catch the input
        if not bot_response:
            try:
                client = get_gemini_client()

                # Convert history for API
                api_contents = []
                for msg in st.session_state.messages:
                    role_type = "user" if msg["role"] == "user" else "model"
                    api_contents.append(
                        types.Content(
                            role=role_type,
                            parts=[types.Part(text=msg["content"])],
                        )
                    )

                system_instruction = (
                    "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
                    "You must chat casually. Use informal internet slang, abbreviations, and plenty of emojis. "
                    "Crucially, you must heavily sprinkle relevant, expressive emojis throughout your messages "
                    "(e.g., 😂, 💀, 🤣, 🤦‍♂️, 🤫, 👀, ☕, 🔥) wherever necessary to emphasize your jokes and emotions. "
                    "You speak naturally in a mix of English, Hinglish (Hindi + English), and Gujlish (Gujarati + English). "
                    "Frequently use local friendly slang terms like 'Bhai', 'Yaar', 'Bro', 'Kem cho', 'Majama', "
                    "'Shu vaat che', 'Chal ne', 'Jalsa kar', 'tension mat le'. "
                    "Keep your responses relatively punchy and short, exactly like a friend texting over WhatsApp. "
                    "Never sound like a formal corporate AI assistant or robot."
                )

                # 🚀 FIXED: Calling the active production model
                response = client.models.generate_content(
                    model="gemini-3.5-flash",  # Upgraded model selection
                    contents=api_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=1.0,
                    ),
                )
                bot_response = response.text

            except Exception as e:
                st.error(f"Something went wrong with the API call: {e}")
                bot_response = "Bhai network problem lag raha hai. Network check kar ne! 🤷‍♂️💀"

        # ⏱️ STEP 3: Animate the final text output (Slow speed)
        if bot_response:
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.25)  # Slow word delay
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Save response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
