import streamlit as st
from openai import OpenAI

# 1. Web Page Setup (Updated to Gaurav)
st.set_page_config(page_title="Gaurav - Best Friend Chat", page_icon="👊")
st.title("👊 Talk to Gaurav")
st.caption("Your AI best friend. Ask anything, vent, or just hang out.")

# 2. Setup the OpenAI Connection
try:
    client = OpenAI()
except Exception:
    st.error("Missing OpenAI API Key! Please set your environment variable.")
    st.stop()

# 3. Define the Best Friend Persona (Updated to Gaurav)
best_friend_persona = (
    "You are Gaurav, the user's absolute best friend. "
    "Speak natively, casually, and warmly. Use modern, relaxed human language. "
    "Do NOT sound like an AI, a corporate assistant, or a customer support agent. "
    "Match the user's energy. Use slang, casual grammar, and light humor. "
    "Be incredibly supportive, listen closely, ask thoughtful follow-up questions, "
    "and offer real, unfiltered friend advice."
)

# 4. Create Web Session Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": best_friend_persona}
    ]

# 5. Render Previous Chat Messages on Screen
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 6. Capture and Process User Input
if user_input := st.chat_input("Say something to Gaurav..."):
    
    # Instantly show what you typed in the UI
    with st.chat_message("user"):
        st.markdown(user_input)
        
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from the AI model
    with st.chat_message("assistant"):
        with st.spinner("Gaurav is typing..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    temperature=0.85
                )
                reply = response.choices.message.content
                st.markdown(reply)
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"Error communicating with Gaurav: {e}")
