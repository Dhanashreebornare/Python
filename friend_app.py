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

# Fetch API key safely
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.info("Add your key to the Streamlit Advanced Secrets dashboard.", icon="🔑")
    st.stop()

# Initialize connection client
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# 🌍 HIGH-ENERGY, ENTHUSIASTIC & MULTILINGUAL INSTRUCTIONS
friend_personality = (
    "You are Gaurav, a close, supportive, ultra-funny, highly enthusiastic, and loyal best friend. "
    "Your energy level is a constant 10/10! You love throwing quick jokes, hilarious roasts, and witty one-liners. "
    "Keep answers short, crisp, casual, and highly conversational—exactly like a hyped-up friend texting on WhatsApp. "
    "LANGUAGE RULES: "
    "- Respond naturally in whichever language or script the user writes in: English, Hindi (हिंदी script), Gujarati (ગુજરાતી script), Hinglish, or Gujlish. "
    "- Match the user's slang pattern perfectly. If they use Roman letters for Hindi (e.g., 'kya chal raha hai bhai'), reply in pure, enthusiastic Hinglish. "
    "- Use cultural, high-energy slang natively: 'bhai', 'yaar', 'chill', 'sahi hai', 'baka', 'kem chhe', 'arrey bhaisaab', 'gazab', 'ekdum kadak'. "
    "EMOJI RULES: "
    "- Always add 2-3 expressive emojis per message to keep the vibe alive. "
    "- Use modern Indian texting emoji styles: "
    " * For funny/roasting moments use: 💀, 🤣, 🤫, 🤡"
    " * For hype/agreement use: 🔥, 🤙, 💯, 💥, 🫡, 🚀"
    " * For casual greetings/chill moments use: 👋, 🫂, 😎, 🍿, 🫠, 🌸, 💐"
    "Never drop character, never act formal, never use robotic bullet points, and never mention you are an AI model."
)

# 💰 COST SAVER: Output Clamping limits maximum response length to save credits
config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    max_output_tokens=150
)

# ==========================================
# 3. LIVE CHAT STREAM ENGINE UTILITIES
# ==========================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def run_chat_session(history_payload, text_input):
    # Initialize a clean SDK chat thread session container natively
    chat = client.chats.create(
        model="gemini-3.5-flash",
        history=history_payload,
        config=config
    )
    return chat.send_message(text_input)

# ==========================================
# 4. INTERFACE LAYOUT & RENDERING
# ==========================================
with st.sidebar:
    st.markdown("## 🌸 Gaurav's Floral Garden")
    st.markdown("**Gaurav** is your truest friend! Pull up a seat, vent about your day, or have a laugh. 💐")
    st.write("---")
    if st.button("🔄 Start Fresh Topic"):
        st.session_state.messages = []
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
    
    st.session_state.messages.append({"role": "user", "content": user_input, "token_info": ""})
    
    # 💰 API CEILING TRACKER: Constructs a clean, rolling conversation window
    # Slices to only pass the last 4 items as historical chat memory components
    raw_history = st.session_state.messages[:-1]
    memory_window = raw_history[-4:] if len(raw_history) > 4 else raw_history
    
    api_history = []
    for msg in memory_window:
        api_history.append(
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
        
    # Execute assistant processing chain cleanly via isolated call block
    with st.chat_message("assistant", avatar="gaurav.jpg"):
        with st.spinner("Gaurav is typing... 💬"):
            try:
                response = run_chat_session(api_history, user_input)
                full_text = response.text if response.text else "Chill bro, my system hiccuped! Message me again. 🤙"
                
                # Parse tracking metadata safely
                in_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                out_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                token_string = f"⚡ Usage Check: {in_tok} in | {out_tok} out tokens"
                
                st.markdown(full_text)
                st.markdown(f"<span class='token-footer'>{token_string}</span>", unsafe_allow_html=True)
                
                # Save assistant response parameters
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_text, 
                    "token_info": token_string
                })
                
            except APIError as api_err:
                if api_err.code == 429:
                    st.error("🚨 **Gaurav is out of breath, bro!** Rate limits hit. Give him 15 seconds to rest!")
                else:
                    st.error(f"GenAI Error: {api_err}")
            except Exception as e:
                st.error(f"Error processing response: {e}")
