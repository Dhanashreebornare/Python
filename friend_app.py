import os
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
    # Looks for GEMINI_API_KEY in Streamlit Cloud Secrets or local environment variables
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
    "Your multilingual best friend, powered by AI (Hinglish & Gujlish master)."
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

    # Initialize client and generate dynamic response
    client = get_gemini_client()

    # Formulate conversational history context for the API
    api_contents = []
    for msg in st.session_state.messages:
        role_type = "user" if msg["role"] == "user" else "model"
        api_contents.append(
            types.Content(
                role=role_type, parts=[types.Part.from_text(msg["content"])]
            )
        )

    # Define Gaurav's personality system instructions
    system_instruction = (
        "You are Gaurav, a funny, witty, sarcastic, and deeply loyal close best friend. "
        "You must chat casually. Use informal internet slang, abbreviations, and emojis. "
        "Crucially, you must speak naturally in a mix of English, Hinglish (Hindi + English), "
        "and Gujlish (Gujarati + English). Frequently use local friendly slang terms like "
        "'Bhai', 'Yaar', 'Bro', 'Kem cho', 'Majama', 'Shu vaat che', 'Chal ne', 'Jalsa kar', 'tension mat le'. "
        "Keep your responses relatively punchy and short, exactly like a friend texting over WhatsApp. "
        "Never sound like a formal corporate AI assistant or robot."
    )

    # Display Gaurav's response with a typing animation
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Fetch dynamic response using the recommended flash model
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=api_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=1.0,  # Higher temperature makes him more creative and funny
                ),
            )
            bot_response = response.text

            # Simulate typing effect
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.06)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Add assistant response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )

        except Exception as e:
            st.error(f"Something went wrong with the API call: {e}")
