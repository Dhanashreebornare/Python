import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ====================================================================
# 1. PREMIUM VISUAL LAYOUT & CSS STYLING OVERRIDES
# ====================================================================
st.set_page_config(
    page_title="Vibe with Gaurav",
    page_icon="🌸",
    layout="centered"
)

# Custom High-Contrast Elegant Floral Friendship Styling Sheet
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
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.7);
}

div[data-testid="stChatMessage"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px -15px rgba(225, 78, 202, 0.2);
}

div[data-testid="stChatMessage"] p, 
div[data-testid="stChatMessage"] span {
    color: #2d1e2f !important;
    font-weight: 500;
    line-height: 1.6;
    font-size: 1.02rem;
}

div[data-testid="stChatMessageUser"] {
    background: linear-gradient(120deg, rgba(255, 240, 243, 0.95) 0%, rgba(255, 245, 247, 0.95) 100%) !important;
    border-bottom-right-radius: 4px !important;
    border-right: 5px solid #ff4e50 !important;
}

div[data-testid="stChatMessageAssistant"] {
    background: linear-gradient(120deg, rgba(253, 240, 255, 0.95) 0%, rgba(250, 245, 255, 0.95) 100%) !important;
    border-bottom-left-radius: 4px !important;
    border-left: 5px solid #e14eca !important;
}

section[data-testid="stSidebar"] {
    background-color: rgba(255, 248, 250, 0.9) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 220, 230, 0.8);
}

.stButton>button {
    background: linear-gradient(90deg, #ff4e50 0%, #e14eca 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(225, 78, 202, 0.25);
    transition: all 0.2s ease !important;
    width: 100%;
    letter-spacing: 0.5px;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px rgba(255, 78, 80, 0.4) !important;
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
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 2. SIDEBAR INTERFACE & SESSION MANAGEMENT
# ====================================================================
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

# ====================================================================
# 3. INTERFACE HEADERS & CACHED API CLIENT CONNECTION
# ====================================================================
st.title("💐 Vibe with Gaurav")
st.markdown("<p class='subtitle-text'>Your close, funny, and multilingual companion.</p>", unsafe_allow_html=True)

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.info("Please add your copied key to the Streamlit Advanced Secrets dashboard to begin.", icon="🔑")
    st.stop()

@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# ====================================================================
# 4. CREDITS SAVER CONFIGURATION & SYSTEM INSTRUCTIONS
# ====================================================================
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
    " * For funny/roasting moments use: 💀, 🤣, 🤫, 🤡"
    " * For hype/agreement use: 🔥, 🤙, 💯, 💥, 🫡"
    " * For casual greeting/chill moments use: 👋, 🫂, 😎, 🍿, 🫠"
    "Never drop character, never act formal, never use robotic bullet points, and never mention you are an AI model."
)

# 💰 CREDIT SAVER 1: max_output_tokens stops billing bleeding up to 70%!
config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    temperature=0.88,
    max_output_tokens=150
)

# Initialize global session tracking data structures
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []

# ====================================================================
# 5. ISOLATED RETRY LOGIC & RUNTIME ENGINE FUNCTION
# ====================================================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def run_api_call(payload):
    # Using cost-optimized gemini-2.5-flash for maximum savings
    return client.models.generate_content(
        model='gemini-2.5-flash',
        contents=payload,
        config=config
    )

def get_gaurav_response(payload_data):
    try:
        response = run_api_call(payload_data)
        if response and response.text:
            return response
        return None
    except APIError as api_err:
        if api_err.code == 429:
            st.error("🚨 Gaurav is out of breath, bro! Free usage limits reached. Give him 15 seconds to relax!")
        else:
            st.error(f"GenAI Connection Error: {api_err}")
        return None
    except Exception as e:
        st.error(f"Something went sideways: {e}")
        return None

# ====================================================================
# 6. APPLICATION DISPLAY CONTAINER & INPUT LOOP
# ====================================================================
# Draw persistent local interface chat canvas cards
for message in st.session_state.messages:
    avatar_icon = "periwinkle.png" if message["role"] == "user" else "gaurav.jpg"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "token_info" in message and message["token_info"]:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# Capture active incoming interactions
user_input = st.chat_input("Say something to Gaurav...")

if user_input:
    # 1. Update UI canvas instantly
    with st.chat_message("user", avatar="periwinkle.png"):
        st.markdown(user_input)
        
    # 2. Append incoming parameters to local history
    st.session_state.messages.append({"role": "user", "content": user_input, "token_info": ""})
    
    # 3. Construct a completely flat context item to pass to history payload safely
    new_user_part = types.Part.from_text(text=user_input)
    new_user_content = types.Content(role="user", parts=[new_user_part])
    st.session_state.api_history.append(new_user_content)
    
    # 4. Process API feedback pipeline
    with st.chat_message("assistant", avatar="gaurav.jpg"):
        with st.spinner("Gaurav is typing... 💬"):
            
            # 💰 CREDIT SAVER 2: Core sliding history tracking ceiling limits context size
            MAX_HISTORY_TURNS = 4
            if len(st.session_state.api_history) > MAX_HISTORY_TURNS:
                payload_slice = st.session_state.api_history[-MAX_HISTORY_TURNS:]
            else:
                payload_slice = st.session_state.api_history
                
            # Fire isolated function worker
            api_response = get_gaurav_response(payload_slice)
            
            if api_response:
                gaurav_text = api_response.text
                
