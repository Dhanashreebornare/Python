import streamlit as st
from google import genai
from google.genai import types

# 1. Premium Visual Page Configuration
st.set_page_config(
    page_title="Chat with Gaurav", 
    page_icon="✨", 
    layout="centered"
)

# Deeply Customized CSS Inject for a Beautiful, Colorful, and Aesthetic Layout
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    
    /* 1. Full Page Vibrant Mesh Gradient */
    .stApp {
        background: linear-gradient(135deg, #0d0b21 0%, #1a0b36 35%, #2a0845 70%, #0b1b36 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* 2. Top Banner Header Styling with Neon Glow Effects */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff007f, #7f00ff, #00f0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 25px rgba(127, 0, 255, 0.35);
        letter-spacing: -1px;
        margin-bottom: 0px !important;
    }
    
    .subtitle-text {
        color: #b4b8da;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* 3. Aesthetic Makeover for Chat Avatars & Message Blocks */
    div[data-testid="stChatMessage"] {
        border-radius: 18px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* User Message Style: Vibrant Magenta Hint */
    div[data-testid="stChatMessageUser"] {
        background-color: rgba(255, 0, 127, 0.08) !important;
        border-left: 5px solid #ff007f !important;
    }
    
    /* Assistant Message Style: Electric Purple/Blue Hint */
    div[data-testid="stChatMessageAssistant"] {
        background-color: rgba(127, 0, 255, 0.08) !important;
        border-left: 5px solid #7f00ff !important;
    }

    /* 4. Elegant Glassmorphism Sidebar Formatting */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 8, 26, 0.85) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* 5. Custom Control Button Designing */
    .stButton>button {
        background: linear-gradient(90deg, #7f00ff 0%, #ff007f 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(127, 0, 255, 0.3);
        transition: all 0.3s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 0, 127, 0.5) !important;
    }

    /* Fix layout width alignments */
    .block-container {
        padding-top: 4rem !important;
        max-width: 700px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation Layout
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
    
    # Session reset action button
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        if "gemini_chat" in st.session_state:
            del st.session_state.gemini_chat
        st.rerun()

# 3. App Header Interface Setup
st.title("🤝 Chat with Gaurav")
st.markdown("<p class='subtitle-text'>Your close, funny, and multilingual companion.</p>", unsafe_allow_html=True)

# 4. Fetch the Key Safely from Streamlit Secrets Management
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

# 7. Persistent Memory Orchestration (Updated to Current Gemini 3.5 Frontier Engine)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = client.chats.create(
        model="gemini-3.5-flash", 
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
