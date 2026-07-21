import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 1. Premium Visual Page Configuration
st.set_page_config(
    page_title="Vibe with Gaurav",
    page_icon="🤝",
    layout="centered"
)

# Custom High-Contrast Elegant Friendship Styling
st.markdown("""
<style>
    @import url('https://googleapis.com');
    
    /* 1. Soft Dynamic Warm Pastel Canvas (Friendship Theme) */
    .stApp {
        background: linear-gradient(135deg, #fff5f5 0%, #f0f4ff 50%, #f5f0ff 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* 2. Bold Radiant Header Typography */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff416c, #8a2387, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
        margin-bottom: 0px !important;
    }
    
    .subtitle-text {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 6px;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* 3. Rounded Elegant Chat Containers */
    div[data-testid="stChatMessage"] {
        border-radius: 24px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 10px 30px -10px rgba(100, 116, 139, 0.12);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -15px rgba(100, 116, 139, 0.2);
    }

    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {
        color: #1e293b !important;
        font-weight: 500;
        line-height: 1.6;
        font-size: 1.02rem;
    }
    
    /* User Message Bubble: Warm Blush Sunrise Frame */
    div[data-testid="stChatMessageUser"] {
        background: linear-gradient(120deg, rgba(255, 240, 243, 0.9) 0%, rgba(255, 245, 247, 0.9) 100%) !important;
        border-bottom-right-radius: 4px !important;
        border-right: 5px solid #ff416c !important;
    }
    
    /* Gaurav Message Bubble: Royal Cosy Lavender Frame */
    div[data-testid="stChatMessageAssistant"] {
        background: linear-gradient(120deg, rgba(243, 240, 255, 0.9) 0%, rgba(247, 245, 255, 0.9) 100%) !important;
        border-bottom-left-radius: 4px !important;
        border-left: 5px solid #8a2387 !important;
    }
    
    /* 4. Glassmorphism Sidebar formatting */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    /* 5. Custom Interactive Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff416c 0%, #8a2387 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.25);
        transition: all 0.2s ease !important;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(138, 35, 135, 0.4) !important;
    }
    
    .block-container {
        padding-top: 3.5rem !important;
        max-width: 720px !important;
    }
    
    /* Subtle minimalist token metrics display */
    .token-footer {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 10px;
        display: block;
        text-align: right;
        font-family: monospace;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation Layout Settings
with st.sidebar:
    st.markdown("## 💖 Your Best Friend's Corner")
    st.markdown(
        "**Gaurav** isn't just an assistant—he's your brother from another mother! "
        "Whether you want to share a joke, vent about a bad day, or talk in broken Hinglish, he's always here for you. 👊"
    )
    st.write("---")
    st.markdown("🗣️ **Vibe Checklist:**")
    st.caption("✅ 100% No Corporate Speak")
    st.caption("✅ Custom Emoji Expressiveness Loaded")
    st.caption("✅ English, हिंदी & ગુજરાતી Spoken Fluidly")
    st.write("---")
    
    # Session data reset action
    if st.button("🔄 Start Fresh Topic"):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.rerun()

# 3. Main Header Typography
st.title("🤝 Vibe with Gaurav")
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

# 6. Deeply Configured Behavioral Model Context Instructions with Emoji Adjustments
friend_personality = (
    "You are Gaurav, a close, supportive, ultra-funny, and loyal best friend. "
    "Keep your answers short, crisp, casual, and highly conversational—exactly like a friend texting on WhatsApp. "
    "You are completely multilingual. Reply natively in whichever language the user texts you in: English, Hindi (हिंदी), or Gujarati (ગુજરાતી). "
    "Match the user's conversational flow perfectly. If they use Hinglish or Gujlish, respond dynamically using the exact same style. "
    "Use plenty of casual text phrases (like 'bro', 'yaar', 'chill', 'sahi hai'), and ask engaging questions to keep the banter alive. "
    "EMOJI RULES: "
    "- Always add 1-3 emojis per message to feel natural, but don't overdo it. "
    "- Use modern conversational and Indian texting emoji vibes: "
    "  * For funny/roasting moments use: 💀, 🤣, 🤫, 🤡"
    "  * For hype/agreement use: 🔥, 🤙, 💯, 💥, 🫡"
    "  * For casual greeting/chill moments use: 👋, 🫂, 😎, 🍿, 🫠"
    "Never drop character, never act formal, never use robotic bullet points, and never mention you are an AI model."
)

config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    temperature=0.88,
)

# 7. Core Thread Memory Persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_history" not in st.session_state:
    st.session_state.api_history = []

# 8. Render High-Contrast Chat History Cards with Custom Avatars
for message in st.session_state.messages:
    avatar_icon = "✨" if message["role"] == "user" else "🧑‍💻"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "token_info" in message:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# --- Helper Function for Automatic Retries with Exponential Backoff ---
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def generate_content_with_retry(contents_payload):
    return client.models.generate_content(
        model='gemini-3.5-flash',
        contents=contents_payload,
        config=config
    )

# 9. Process Active Client Message Inputs
if user_input := st.chat_input("Say something to Gaurav..."):
    with st.chat_message("user", avatar="✨"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.api_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    )

    # Generate response turn using active connection
    with st.chat_message("assistant", avatar="🧑‍💻"):
        message_placeholder = st.empty()
        token_placeholder = st.empty()
        
        full_response = ""
        token_string = ""
        api_success = False
        
        with st.spinner("Gaurav is typing... 💬"):
            try:
                # --- QUOTA MINIMIZER: Rolling Context Window ---
                # Keeps only the last 6 messages to protect the free tier from blowing up
                MAX_HISTORY_TURNS = 6
                if len(st.session_state.api_history) > MAX_HISTORY_TURNS:
                    payload = st.session_state.api_history[-MAX_HISTORY_TURNS:]
                else:
                    payload = st.session_state.api_history

                # Fire structured API request
                response = generate_content_with_retry(payload)
                full_response = response.text
                
                # Extract token metrics safely from response metadata
                input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                token_string = f"⚡ Usage Check: {input_tokens} in | {output_tokens} out tokens"
                api_success = True
                
            except APIError as api_err:
                if api_err.code == 429:
                    st.error("🚨 **Gaurav is out of breath, bro!** The free limits ran out. Give him 15-20 seconds to catch his breath before typing again!")
                elif api_err.code == 503:
                    st.error("Gaurav's line is locked up due to high traffic! 😅 Try hitting send again in a few seconds.")
                else:
