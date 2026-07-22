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

# Funny, friendly responses Gaurav might say
GAURAV_RESPONSES = [
    "Bro, I was just thinking about how awesome I am, and then you texted!",
    "No way! Tell me you didn't actually do that? 😂",
    "I'm 100% listening, but first, did you bring snacks?",
    "That sounds like a problem for future us. Let's order pizza instead.",
    "Classic you. Honestly, what would you do without me?",
    "Haha, you're hilarious! (My code forced me to say this).",
    "Idk man, sounds sketchy. I'm in.",
    "brb, pretending to be a busy AI for 2 seconds... Okay, I'm back. What's up?",
]


def get_gaurav_response(user_input):
    """Generates a friendly, funny response based on keywords or random selection."""
    user_input = user_input.lower()

    if "hey" in user_input or "hello" in user_input or "hi" in user_input:
        return "Yo! What's cracking, my friend?"
    elif "help" in user_input:
        return "I can give you terrible advice, or we can just make fun of the situation. Choose wisely."
    elif "bye" in user_input:
        return "Don't leave me alone with my thoughts! Just kidding, see ya bro!"
    else:
        return random.choice(GAURAV_RESPONSES)


# App Title
st.title("🤖 Chat with Gaurav")
st.subheader("Your best friend, available 24/7 (unlike real friends).")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Yo! Finally you remembered your best friend. What's up today?",
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
