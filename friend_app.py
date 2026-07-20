import streamlit as st
from google import genai
from google.genai import types

# 1. Premium Visual Page Configuration
st.set_page_config(
    page_title="Chat with Gaurav", 
    page_icon="✨", 
    layout="centered"
)

# Custom CSS for a beautiful, premium dark chat interface
st.markdown("""
    <style>
    /* Gradient Background covering the whole screen */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #1e1b4b 100%);
    }
    /* Dynamic Glowing Gradient for Headers */
    h1 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(left, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
    }
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 5px;
        margin-bottom: 2.5rem;
    }
    /* Restructure padding for optimal layout layout */
    .block-container {
        padding-top: 3.5rem !important;
        max-width: 680px !important;
    }
    /* Premium Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    /* Customising action buttons */
    .stButton>button {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #4338ca !important;
        border-radius: 12px !important;
        width: 100%;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338ca !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Polished Sidebar Layout
with st.sidebar:
    st.markdown("### ✨ Meet Your Best Friend")
    st.markdown(
        "**Gaurav** is your dedicated AI companion. "
        "He is supportive, incredibly funny, and always ready to vibe with you."
    )
    st.write("---")
    st.markdown("🌐 **Languages Supported:**")
    st.caption("🗣️ **English** (Casual & Friendly)")
    st.caption("🇮🇳 **Hindi** (हिंदी - एकदम दोस्तों की तरह)")
    st.caption("🦁 **Gujarati** (ગુજરાતી - પ્રોપર મિત્ર ભાવે)")
    st.write("---")
    
    # Session reset action
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        if "gemini_chat" in st.session_state:
            del st.session_state.gemini_chat
        st.rerun()

# 3. App Header Interface
st.title("🤝 Chat with Gaurav")
st.markdown("<p class='subtitle-text'>Your close, funny, and multilingual companion.</p>", unsafe_allow_html=True)

# 4. Fetch the Key Safely from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.info("Please add your copied key to the Streamlit Advanced Secrets dashboard to begin.", icon="🔑")
    st.stop()

# 5. Initialize the Cache Client Standard Framework
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# 6. Deeply Defined Multilingual Friendly System Instructions
friend_personality = (
    "You are Gaurav, a close, supportive, and funny friend. "
    "Keep your answers short, casual, and highly conversational—just like a friend texting back and forth. "
    "You are fully multilingual. Reply natively in whichever language the user texts you in: English, Hindi (हिंदी), or Gujarati (ગુજરાતી). "
    "Match the user's conversation style completely. If they use Hinglish (Hindi written in English alphabets) or Gujlish, respond dynamically using the exact same style. "
    "Use casual phrasing, regular texting emojis, and ask questions to keep the chat lively. "
    "Never drop character, never say you are an AI model, and never act like a corporate assistant."
)

config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    temperature=0.85, # Adds warm conversational variety to text generations
)

# 7. Persistent Memory Orchestration
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = client.chats.create(
        model="gemini-1.5-flash", 
        config=config
    )

# 8. Render Beautiful Chat UI Logs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 9. Track Real-Time User Message Feed
if user_input := st.chat_input("Type a message to Gaurav..."):
    
    # Append & display immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process Gaurav's multilingual reply stream
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = st.session_state.gemini_chat.send_message(user_input)
            full_response = response.text
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Gaurav went offline for a second. Try again! Details: {e}")
