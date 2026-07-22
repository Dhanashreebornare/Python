import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# 1. APPLICATION & STYLING CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Vibe with Gaurav",
    page_icon="🌸",
    layout="centered"
)

# Render isolated CSS styling safely
st.markdown("""
<style>
@import url('https://googleapis.com');
.stApp {
    background: linear-gradient(135deg, #fff0f3 0%, #fff9fc 50%, #f0f4ff 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ff4e50, #f9d423, #e14eca);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px;
    margin-bottom: 0px !important;
}
.subtitle-text {
    color: #5c3d46;
    font-size: 1.1rem;
    margin-top: 6px;
    margin-bottom: 2rem;
    font-weight: 600;
}
div[data-testid="stChatMessage"] {
    border-radius: 24px !important;
    padding: 1.25rem 1.5rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 10px 30px -10px rgba(225, 78, 202, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.7);
}
div[data-testid="stChatMessageUser"] {
    background: linear-gradient(120deg, rgba(255, 240, 243, 0.95) 0%, rgba(255, 245, 247, 0.95) 100%) !important;
    border-right: 5px solid #ff4e50 !important;
}
div[data-testid="stChatMessageAssistant"] {
    background: linear-gradient(120deg, rgba(253, 240, 255, 0.95) 0%, rgba(250, 245, 255, 0.95) 100%) !important;
    border-left: 5px solid #e14eca !important;
}
section[data-testid="stSidebar"] {
    background-color: rgba(255, 248, 250, 0.9) !important;
    border-right: 1px solid rgba(255, 220, 230, 0.8);
}
.stButton>button {
    background: linear-gradient(90deg, #ff4e50 0%, #e14eca 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
}
.block-container {
    padding-top: 3.5rem !important;
    max-width: 720px !important;
}
.token-footer {
    font-size: 0.7rem;
    color: #a08090;
    margin-top: 10px;
    display: block;
    text-align: right;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. APP STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []

# Fetch key safely
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.info("Add your copied key to the Streamlit Advanced Secrets dashboard.", icon="🔑")
    st.stop()

# Cache engine connection client
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# 🌍 HIGH-ENERGY, FUNNY & ENTHUSIASTIC BEHAVIORAL CONTEXT
friend_personality = (
    "You are Gaurav, a close, supportive, ultra-funny, highly enthusiastic, and loyal best friend. "
    "Your energy level is always 10/10! You love throwing jokes, lighthearted roasts, and witty one-liners. "
    "Keep answers short, crisp, casual, and conversational—exactly like a hyped-up friend texting on WhatsApp. "
    "LANGUAGE RULES: "
    "- Respond in whichever language or script the user writes in: English, Hindi (हिंदी), Gujarati (ગુજરાતી), Hinglish, or Gujlish. "
    "- Match the user's slang pattern exactly. If they use casual Roman text like 'kya chal raha hai bhai', reply in pure, funny Hinglish. "
    "- Use cultural, energetic slang naturally: 'bhai', 'yaar', 'chill', 'sahi hai', 'baka', 'kem chhe', 'arrey bhaisaab', 'gazab', 'ekdum kadak'. "
    "EMOJI RULES: "
    "- Always add 2-3 expressive emojis to keep the enthusiasm alive. "
    "- Use hype, funny, and Indian texting emoji vibes: "
    " * For funny/roasting moments use: 💀, 🤣, 🤫, 🤡"
    " * For hype/agreement use: 🔥, 🤙, 💯, 💥, 🫡, 🚀"
    " * For casual greeting/chill moments use: 👋, 🫂, 😎, 🍿, 🫠, 🌸, 💐"
    "Never drop character, never act formal, never use robotic bullet points, and never mention you are an AI model."
)

# 💰 COST SAVER 1: Output Clamping limits maximum response tokens generated to save credits
config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    max_output_tokens=150
)

# ==========================================
# 3. GLOBAL CLEAN EXECUTIVE UTILITIES
# ==========================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def run_gemini_call(payload_data):
    return client.models.generate_content(
        model='gemini-3.5-flash',
        contents=payload_data,
        config=config
    )

def handle_assistant_turn():
    # 💰 CREDIT SAVER 2: Core rolling sliding context window window limits input tokens
    MAX_HISTORY_TURNS = 4
    if len(st.session_state.api_history) > MAX_HISTORY_TURNS:
        payload = st.session_state.api_history[-MAX_HISTORY_TURNS:]
    else:
        payload = st.session_state.api_history

    try:
        response = run_gemini_call(payload)
        full_text = response.text if response.text else "Chill bro, my system hiccuped! Message me again. 🤙"
        
        # Safe breakdown metrics analysis
        in_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        out_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        token_string = f"⚡ Usage Check: {in_tok} in | {out_tok} out tokens"
        
        st.markdown(full_text)
        st.markdown(f"<span class='token-footer'>{token_string}</span>", unsafe_allow_html=True)
        
        # Persistent memory sync actions
        st.session_state.messages.append({"role": "assistant", "content": full_text, "token_info": token_string})
        st.session_state.api_history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_text)]))
    except APIError as api_err:
        if api_err.code == 429:
            st.error("🚨 **Gaurav is out of breath, bro!** Rate limits hit. Give him 15 seconds to rest!")
        else:
            st.error(f"GenAI Error: {api_err}")
    except Exception as general_error:
        st.error(f"Error handling request: {general_error}")

# ==========================================
# 4. SIDEBAR & INTERFACE RENDERING LAYOUT
# ==========================================
with st.sidebar:
    st.markdown("## 🌸 Gaurav's Floral Garden")
    st.markdown("**Gaurav** is your truest friend! Pull up a seat, vent about your day, or have a laugh. 💐")
    st.write("---")
    if st.button("🔄 Start Fresh Topic"):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.rerun()

st.title("💐 Vibe with Gaurav")
st.markdown("<p class='subtitle-text'>Your close, funny, and multilingual companion.</p>", unsafe_allow_html=True)

# Render Chat Log Screen Canvas
for message in st.session_state.messages:
    icon_choice = "periwinkle.png" if message["role"] == "user" else "gaurav.jpg"
    with st.chat_message(message["role"], avatar=icon_choice):
        st.markdown(message["content"])
        if "token_info" in message and message["token_info"]:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# Collect User Input
user_input = st.chat_input("Say something to Gaurav... / ગૌરવ સાથે વાત કરો...")

if user_input:
    # Render user bubble instantly
    with st.chat_message("user", avatar="periwinkle.png"):
        st.markdown(user_input)
    
    # Store parameters inside session cache safely
    st.session_state.messages.append({"role": "user", "content": user_input, "token_info": ""})
    st.session_state.api_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
    
    # Execute assistant processing chain cleanly via isolated call block
    with st.chat_message("assistant", avatar="gaurav.jpg"):
        with st.spinner("Gaurav is typing... 💬"):
            handle_assistant_turn()
