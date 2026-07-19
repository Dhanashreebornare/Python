import streamlit as st
from huggingface_hub import InferenceClient

# 1. Page Configuration
st.set_page_config(page_title="Gaurav - Best Friend Chat", page_icon="🙄")
st.title("🙄 Talk to Gaurav")
st.caption("Your funny, easily annoyed AI best friend.")

# 2. Setup the Free Hugging Face Connection safely
try:
    hf_token = st.secrets["HF_TOKEN"]
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
except Exception:
    st.error("Missing HF_TOKEN! Please set it in your Streamlit Secrets Dashboard.")
    st.stop()

# 3. Create Web Session Memory & Irritation Counter
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# 4. Clear/Reset Chat Button
if st.button("🔄 Apologise & Reset Chat"):
    st.session_state.messages = []
    st.session_state.message_count = 0
    st.rerun()

# 5. Dynamically update Gaurav's system prompt based on message count
if st.session_state.message_count <= 3:
    mood_instruction = "You are in a great mood. Be witty, funny, drop casual jokes, and match their energy."
elif 3 < st.session_state.message_count <= 6:
    mood_instruction = "You are starting to get slightly annoyed. Use mild sarcasm, tell them they talk too much, and keep jokes a bit sharp."
else:
    mood_instruction = "You are highly irritated and exhausted by all this talking. Give short, blunt, sarcastic answers. Ask them to give you a break or shut up playfully."

best_friend_persona = (
    f"You are Gaurav, the user's absolute best friend. {mood_instruction} "
    "Speak natively, casually, and warmly, but let your irritation show if dictated by your mood. "
    "Use modern, relaxed human language. Do NOT sound like a corporate AI assistant. "
    "Use casual grammar and light humor. Offer real, unfiltered friend advice when asked."
)

# 6. Render Previous Chat Messages on Screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Capture and Process User Input
if user_input := st.chat_input("Say something to Gaurav..."):
    
    st.session_state.message_count += 1
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Clean payload construction for Hugging Face
    api_messages = [{"role": "system", "content": best_friend_persona}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Gaurav is typing..."):
            try:
                response = client.chat_completion(
                    messages=api_messages,
                    max_tokens=250,
                    temperature=0.85
                )
                
                # FIXED: Bulletproof response parsing handling both formats
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, "message"):
                        reply = choice.message.content
                    else:
                        reply = choice.get("message", {}).get("content", "")
                else:
                    # Fallback for old library formats
                    reply = response["choices"][0]["message"]["content"]
                
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error communicating with Gaurav: {e}")
