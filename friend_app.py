import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
        background: radial-gradient(circle at top right, #fdfbfb 0%, #ebedee 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* 2. Bold Vibrant Header Styling */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ff007f, #7f00ff, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
        margin-bottom: 0px !important;
    }
    
    .subtitle-text {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 6px;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }
    
    /* 3. Modern Floating Cards for Messages */
    div[data-testid="stChatMessage"] {
        border-radius: 20px !important;
        padding: 1.25rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #1e293b !important;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* User Message Frame: Vibrant Soft Pink Accent */
    div[data-testid="stChatMessageUser"] {
        background-color: rgba(255, 245, 248, 0.8) !important;
        border: 1px solid rgba(255, 227, 236, 0.7) !important;
        border-right: 6px solid #ff007f !important;
    }
    
    /* Gaurav Message Frame: Ultra-Clean Violet Soft Glass */
    div[data-testid="stChatMessageAssistant"] {
        background-color: rgba(247, 244, 255, 0.8) !important;
        border: 1px solid rgba(235, 223, 255, 0.7) !important;
        border-left: 6px solid #7f00ff !important;
    }
    
    /* 4. Elegant Sidebar Formatting */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* 5. Custom Control Button Framework */
    .stButton>button {
        background: linear-gradient(135deg, #7f00ff 0%, #ff007f 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(127, 0, 255, 0.2);
        transition: all 0.2s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 0, 127, 0.3) !important;
    }
    
    .block-container {
        padding-top: 3.5rem !important;
        max-width: 720px !important;
    }
    
    .token-footer {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 8px;
        display: block;
        text-align: right;
        font-family: monospace;
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
    "Match the user's conversation style completely. If they use Hinglish or Gujlish, respond dynamically using the exact same style. "
    "Use casual phrasing, regular texting emojis, and ask questions to keep the chat lively. "
    "Never drop character, never say you are an AI model, and never act like a corporate assistant."
)

config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    temperature=0.85,
)

# 7. Core Thread Memory Persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = client.chats.create(
        model="gemini-2.5-flash",  # Switched to production-stable gemini-2.5-flash to optimize quota usage
        config=config
    )

# 8. Render High-Contrast Chat History Cards
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "token_info" in message:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# --- Quota Minimizer: Context Limit Handler Function ---
def send_message_optimized(chat_session, user_message):
    """
    Trims structural history before sending requests to minimize token footprint 
    and completely eliminate exponential free tier usage explosion.
    """
    # Max history depth: Keep last 6 text turns (3 user, 3 assistant responses)
    MAX_HISTORY_TURNS = 6
    
    if len(chat_session._history) > MAX_HISTORY_TURNS:
        # Keep systemic settings but drop oldest historical conversations
        chat_session._history = chat_session._history[-MAX_HISTORY_TURNS:]
        
    return chat_session.send_message(user_message)

# --- Helper Function for Automatic Retries ---
@retry(
    stop=stop_after_attempt(3), # Reduced to 3 to prevent long UI lockups
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def send_message_with_retry(user_message):
    """Sends a message to the active chat session with exponential backoff safety."""
    return send_message_optimized(st.session_state.gemini_chat, user_message)

# 9. Process Active Client Message Inputs
if user_input := st.chat_input("Type a message to Gaurav..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response turn using active connection
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        token_placeholder = st.empty()
        
        with st.spinner("Gaurav is typing... 💬"):
            try:
                response = send_message_with_retry(user_input)
                full_response = response.text
                
                # Extract token metrics safely from response metadata
                input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                token_string = f"⚡ Spent: {input_tokens} input | {output_tokens} output tokens"
                
                # --- Optimized Fluid Typing Simulation Engine ---
                words = full_response.split(" ")
                for i in range(1, len(words) + 1):
                    # Join words progressively for cleaner memory execution
                    message_placeholder.markdown(" ".join(words[:i]) + " ▌")
                    time.sleep(0.04) # Smoother, slightly faster pacing
                    
                # Final clean layout pass
                message_placeholder.markdown(full_response)
                token_placeholder.markdown(f"<span class='token-footer'>{token_string}</span>", unsafe_allow_html=True)
                
                # Save chat payload with token metadata appended
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "token_info": token_string
                })
                
            except APIError as api_err:
                if api_err.code == 429:
                    st.error("🚨 **Gaurav is completely out of breath!** The free tier rate limit was fully exhausted. Please wait 15-20 seconds before typing your next message.")
