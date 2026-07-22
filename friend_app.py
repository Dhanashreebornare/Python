import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 1. Premium Visual Page Configuration
st.set_page_config(
    page_title="Vibe with Gaurav",
    page_icon="🌸",
    layout="centered"
)

# Completely Revamped Layout Stylesheet with Text Box Gradient Glow and Optimized Background Spacing
st.markdown("""
<style>
    @import url('https://googleapis.com');
    
    /* Base Application Layout Restructuring */
    .stApp {
        background: linear-gradient(135deg, #fff0f3 0%, #fff9fc 50%, #f0f4ff 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        position: relative;
        overflow-x: hidden;
    }

    /* Tweaked Background Flowers Layer: Softer transparency and much wider layout spacing */
    .stApp::before {
        content: "🌸          💮          🌹          🌻          🌸          💐";
        position: fixed;
        top: -10%;
        left: 5%;
        width: 90%;
        height: 120%;
        font-size: 24px;
        line-height: 8;          /* Increased line-height for wider vertical spacing */
        word-spacing: 240px;     /* Double the word-spacing to separate them horizontally */
        opacity: 0.12;           /* Lower opacity (12%) for a premium, non-distracting watermark look */
        pointer-events: none;
        z-index: -1 !important; 
        white-space: pre-wrap;
        animation: floatFlowers 45s linear infinite; /* Slower, more calming movement cadence */
    }

    @keyframes floatFlowers {
        0% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-50px) rotate(3deg); }
        100% { transform: translateY(0) rotate(0deg); }
    }
    
    /* Content wrapper safety layer */
    .block-container {
        position: relative;
        z-index: 2 !important;
        padding-top: 3.5rem !important;
        max-width: 720px !important;
    }
    
    /* Clear Readable Typography & Headers */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff4e50, #d6249f, #b116de);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
        margin-bottom: 0px !important;
    }
    
    .subtitle-text {
        color: #5c3d46 !important;
        font-size: 1.1rem;
        margin-top: 6px;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Deep Container Elements Override for Chat Messages */
    div[data-testid="stChatMessage"] {
        border-radius: 24px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 10px 30px -10px rgba(225, 78, 202, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        position: relative;
        z-index: 5 !important;
    }
    
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -15px rgba(225, 78, 202, 0.25);
    }

    /* Structural fix for high-contrast message texts */
    div[data-testid="stChatMessageContent"] p, 
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] div {
        color: #231224 !important;
        font-weight: 600 !important;
        line-height: 1.6 !important;
        font-size: 1.05rem !important;
    }
    
    /* User Message Bubble Styling */
    div[data-testid="stChatMessageUser"] {
        background: linear-gradient(120deg, #fff3f5 0%, #ffeef1 100%) !important;
        border-bottom-right-radius: 4px !important;
        border-right: 6px solid #ff4e50 !important;
    }
    
    /* Assistant Message Bubble Styling */
    div[data-testid="stChatMessageAssistant"] {
        background: linear-gradient(120deg, #fdf2ff 0%, #fae6ff 100%) !important;
        border-bottom-left-radius: 4px !important;
        border-left: 6px solid #e14eca !important;
    }
    
    /* Glassmorphism High-Contrast Sidebar Formatting */
    section[data-testid="stSidebar"] {
        background-color: #fffafd !important;
        border-right: 2px solid #ffd1df !important;
        z-index: 100 !important;
    }
    
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #381a22 !important;
    }
    
    /* Custom High Contrast Action Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff4e50 0%, #e14eca 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(225, 78, 202, 0.3);
        transition: all 0.2s ease !important;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    /* Chat Input Container Visibility & Custom Radiant Gradient Glow Fix */
    div[data-testid="stChatInput"] {
        z-index: 99 !important;
        position: relative;
        background: transparent !important;
    }
    
    /* Adding a warm blur backdrop glow specifically anchoring behind the chat input box frame */
    div[data-testid="stChatInput"] > div {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(8px);
        border-radius: 20px !important;
        box-shadow: 0 -15px 40px -10px rgba(255, 78, 80, 0.15), 
                    0 15px 30px -10px rgba(225, 78, 202, 0.2) !important;
        border: 1px solid rgba(255, 224, 230, 0.8) !important;
        padding: 4px;
    }
    
    div[data-testid="stChatInput"] textarea {
        color: #1c0b1d !important;
        font-weight: 600 !important;
    }
    
    .token-footer {
        font-size: 0.75rem;
        color: #7a5a68 !important;
        margin-top: 12px;
        display: block;
        text-align: right;
        font-family: monospace;
        letter-spacing: 0.4px;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation Layout Settings
with st.sidebar:
    st.markdown("## 🌸 Gaurav's Floral Garden")
    st.markdown(
        "**Gaurav** isn't just an assistant—he's your truest friend! "
        "Pull up a seat in the garden, vent about your day, or have a laugh in casual Hinglish/Gujlish. He always has your back. 💐"
    )
    st.write("---")
    st.markdown("🗣️ **Vibe Checklist:**")
    st.caption("✅ 100% No Corporate Speak")
    st.caption("🌸 Friendly Floral Aesthetic Active")
    st.caption("💬 English, हिंदी & ગુજરાતી Spoken Fluidly")
    st.write("---")
    
    if st.button("🔄 Start Fresh Topic"):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.rerun()

# 3. Main Header Typography
st.title("💐 Vibe with Gaurav")
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
    "You are Gaurav, a close, supportive, ultra-funny, and loyal best friend. "
    "Keep your answers short, crisp, casual, and highly conversational—exactly like a friend texting on WhatsApp. "
    "You are completely multilingual. Reply natively in whichever language the user texts you in: English, Hindi (हिंदी), or Gujarati (ગુજરાતી). "
    "Match the user's conversational flow perfectly. If they use Hinglish or Gujlish, respond dynamically using the exact same style. "
    "Use plenty of casual text phrases (like 'bro', 'yaar', 'chill', 'sahi hai'), and ask engaging questions to keep the 'vibe' alive. "
    "EMOJI RULES: "
    "- Always add 1-3 emojis per message to feel natural, but don't overdo it. "
    "- Since your chat screen has a floral garden theme, occasionally drop a flower emoji (🌸, 🌹, 🌻, 💐) when greeting or sending positive vibes. "
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

# 8. Render High-Contrast Chat History Cards with Custom DP Assets
for message in st.session_state.messages:
    avatar_icon = "periwinkle.png" if message["role"] == "user" else "gaurav.jpg"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "token_info" in message:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# --- Helper Function for Automatic Retries with Exponential Backoff ---
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIError),
