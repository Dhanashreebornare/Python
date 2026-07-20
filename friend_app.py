import streamlit as st
from google import genai
from google.genai import types

# 1. Premium Visual Page Configuration
st.set_page_config(
    page_title="Chat with Gaurav", 
    page_icon="✨", 
    layout="centered"
)

# Custom High-Contrast Aesthetic Light Theme Styling
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    
    /* 1. Light Dynamic Pastel Canvas Background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* 2. Bold Vibrant Header Styling */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #d90429, #6c5ce7, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0px !important;
    }
    
    .subtitle-text {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 2.5rem;
        font-weight: 600;
    }

    /* 3. Deep High-Contrast Chat Message Text Containers */
    div[data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
        color: #0f172a !important; /* Forces Text Dark */
    }
    
    /* Ensure markdown content within messages remains deeply readable */
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #0f172a !important;
        font-weight: 500;
    }
    
    /* User Message Frame: Light Pink with Deep Dark Text */
    div[data-testid="stChatMessageUser"] {
        background-color: #fff5f8 !important;
        border: 1px solid #ffe3ec !important;
        border-left: 6px solid #ff007f !important;
    }
    
    /* Gaurav Message Frame: Light Lavender with Deep Dark Text */
    div[data-testid="stChatMessageAssistant"] {
        background-color: #f7f4ff !important;
        border: 1px solid #ebdfff !important;
        border-left: 6px solid #7f00ff !important;
    }

    /* 4. Balanced Light Sidebar Layout formatting */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p {
        color: #0f172a !important;
    }
    
    /* 5. Custom Control Button Framework */
    .stButton>button {
        background: linear-gradient(90deg, #6c5ce7 0%, #ff007f 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.2);
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(255, 0, 127, 0.35) !important;
    }

    /* Standard container centering constraints */
    .block-container {
        padding-top: 4rem !important;
        max-width: 700px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation Layout Settings
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
    
    # Session data reset action
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        if "gemini_chat" in st.session_state:
            del st.session_state.gemini_chat
        st.rerun()

# 3. Main Header Typography
st.title("🤝 Chat with Gaurav")
st.markdown("<p class='subtitle-text'>Your close, funny, and multilingual companion.</p>", unsafe_allow_html=True)

# 4. Fetch the Active Authorization Key securely
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.info("Please add your copied key to the Streamlit Advanced Secrets dashboard to begin.", icon="🔑")
    st.stop()

# 5. Initialize the Cache Client Engine standard
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# 6. Deeply Configured Behavioral Model Context Instructions
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
    temperature=0.85, 
)

# 7. Core Thread Memory Persistence (Syncing onto Gemini 3.5 Engine Endpoint)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = client.chats.create(
        model="gemini-3.5-flash", 
        config=config
    )

# 8. Render High-Contrast Chat History Cards
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 9. Process Active Client Message Inputs
if user_input := st.chat_input("Type a message to Gaurav..."):
    
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response turn using active connection
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = st.session_state.gemini_chat.send_message(user_input)
            full_response = response.text
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Gaurav went offline for a second. Try again! Details: {e}")
