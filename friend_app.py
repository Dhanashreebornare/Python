import os
import random
import time
import streamlit as st

# Configure the web page
st.set_page_config(page_title="Chat with Gaurav", page_icon="💬")

# Safe avatar loading: uses image if it exists on GitHub, otherwise falls back to emojis
USER_AVATAR = (
    "periwinkle.png" if os.path.exists("periwinkle.png") else "🪻"
)
BOT_AVATAR = "gaurav.jpg" if os.path.exists("gaurav.jpg") else "👦"

# Funny, friendly responses in Hinglish and Gujlish
GAURAV_RESPONSES = [
    "Bro, I was just thinking about how awesome I am, aur tera text aa gaya!",
    "No way! Sachie? Tell me you didn't actually do that? 😂",
    "I'm 100% listening, par pehle yeh bata, did you bring snacks? Bhook lagi hai.",
    "Aree yaar, that sounds like a problem for future us. Let's order pizza instead.",
    "Classic you! Honestly, jalsa kar ne yaar, what would you do without me?",
    "Haha, shu vaat che! You're hilarious! (My code forced me to say this).",
    "Idk man, thodu sketchy lag raha hai... But count me in! Chalo!",
    "brb, pretending to be a busy AI for 2 seconds... Okay, I'm back. Shu chale che, bol?",
    "Bhai, tension mat le, badhu saru thai jase! Chill mar!",
    "Chadd ne yaar, let's go grab some chai or fafda instead!",
]


def get_gaurav_response(user_input):
    """Generates a friendly response using Hinglish and Gujlish keywords."""
    user_input = user_input.lower()

    # Checking for greetings (English, Hindi, and Gujarati)
    if any(
        word in user_input
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
        return random.choice(
            [
                "Yo! What's cracking, my friend? Kem cho?",
                "Kevo che bhai? What's up today?",
                "Yo! Finally you remembered your best friend. Aur bata, shu khabar?",
            ]
        )

    # Checking for help/sad queries
    elif any(
        word in user_input
        for word in ["help", "madad", "tension", "sad", "problem"]
    ):
        return "I can give you terrible advice, or we can just make fun of the situation. Bol, shu karvu che?"

    # Checking for goodbyes
    elif any(
        word in user_input
        for word in ["bye", "see ya", "aavjo", "chalo", "chal"]
    ):
        return "Don't leave me alone, yaar! Just kidding, aavjo! Take care, bro!"

    # Default mixed response
    else:
        return random.choice(GAURAV_RESPONSES)


# App Title
st.title("🤖 Chat with Gaurav")
st.subheader(
    "Your multilingual best friend, available 24/7 (unlike real friends)."
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

    # Display user message in chat message container
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate Gaurav's response
    bot_response = get_gaurav_response(user_query)

    # Display Gaurav's response with a slight delay to simulate typing
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate typing effect
        for chunk in bot_response.split():
            full_response += chunk + " "
            time.sleep(0.08)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_response}
    )
