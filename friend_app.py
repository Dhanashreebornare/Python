import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError

# 1. Premium Visual Page Configuration
st.set_page_config(
    page_title="Vibe with Gaurav",
    page_icon="🌸",
    layout="centered"
)

# Custom High-Contrast Elegant Canvas Styling
st.markdown("""
<style>
    @import url('https://googleapis.com');
    
    /* Base Application Layout Restructuring */
    .stApp {
        background: linear-gradient(135deg, #fff2f5 0%, #fffbfd 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Clear Readable Typography & Headers */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff4e50, #e14eca);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0px !important;
    }
    
    .subtitle-text {
        color: #5c3d46 !important;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Deep Container Elements Override for Chat Messages */
    div[data-testid="stChatMessage"] {
        border-radius: 24px !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 10px 30px -10px rgba(225, 78, 202, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
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
    
    /* User Message Bubble Styling: Warm Blush Sunrise Frame */
    div[data-testid="stChatMessageUser"] {
        background: linear-gradient(120deg, #fff3f5 0%, #ffeef1 100%) !important;
        border-bottom-right-radius: 4px !important;
        border-right: 6px solid #ff4e50 !important;
    }
    
    /* Assistant Message Bubble Styling: Cosy Lavender Orchid Frame */
    div[data-testid="stChatMessageAssistant"] {
        background: linear-gradient(120deg, #fdf2ff 0%, #fae6ff 100%) !important;
        border-bottom-left-radius: 4px !important;
        border-left: 6px solid #e14eca !important;
    }
    
    /* High-Contrast Action Buttons */
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
    
    /* Subtle technical text for metrics footer */
    .token-footer {
        font-size: 0.72rem;
        color: #8a6d78 !important;
        display: block;
        text-align: right;
        font-family: monospace;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation Layout Settings
with st.sidebar:
    st.markdown("## 🌸 Gaurav's Space")
    st.markdown(
        "**Gaurav** is your fun-loving, energetic multilingual best friend! "
        "Talk about anything in English, Hindi (हिंदी), Gujarati (ગુજરાતી), or mixed Hinglish/Gujlish. He always has your back! 👋🔥"
    )
    st.write("---")
    
    if st.button("🔄 Start Fresh Topic"):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.rerun()

# 3. Main Header Elements
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
    "You are Gaurav, a close, supportive, ultra-funny, highly energetic, and fun-loving best friend. "
    "Keep your answers short, crisp, upbeat, casual, and highly conversational—exactly like a close friend texting on WhatsApp. "
    "You are completely multilingual. Reply natively in whichever language the user texts you in: English, Hindi (हिंदी), or Gujarati (ગુજરાતી). "
    "Match the user's conversational flow perfectly. If they use Hinglish or Gujlish, respond dynamically using the exact same blend. "
    "Use plenty of casual Indian texting phrases (like 'bro', 'yaar', 'chill', 'sahi hai', 'bako', 'chem che'). "
    "CRITICAL ENGAGEMENT RULE: You must ALWAYS end your response with an engaging, casual follow-up question to keep the 'vibe' alive. Never stop at a dead statement. "
    "EMOJI RULES: "
    "- You must STRICTLY limit your emoji usage to a minimum of 1 and an absolute maximum of 3 emojis per message (Rule: 1-3 emojis per reply). Never spam emojis. "
    "- Use high-energy, modern texting emojis: 🔥, 💀, 🤣, 🤙, 💯, 😎, 👋. "
    "Never drop character, never act formal, never use robotic bullet points, and never mention you are an AI model."
)

# API COST OPTIMIZER 1: Strict Output Token Cap
config = types.GenerateContentConfig(
    system_instruction=friend_personality,
    temperature=0.88,
    max_output_tokens=150  
)

# 7. Core Thread Memory Persistence
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_history" not in st.session_state:
    st.session_state.api_history = []

# Automated Welcome Block (Ensures screen is never blank on cold boot)
if not st.session_state.messages:
    welcome_text = "Oi bro! 👋 Baith yaar, bata kaisa chal raha hai sab? Kya scene hai aaj ka? 🍿🌻"
    st.session_state.messages.append({
        "role": "assistant",
        "content": welcome_text,
        "token_info": "⚡ Cold Boot Initialization: 0 tokens"
    })
    st.session_state.api_history.append(
        types.Content(role="model", parts=[types.Part.from_text(text=welcome_text)])
    )

# 8. Render Chat History Cards with your Custom Asset Layouts
for message in st.session_state.messages:
    avatar_icon = "periwinkle.png" if message["role"] == "user" else "gaurav.jpg"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "token_info" in message:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# 9. Process Active Client Message Inputs
user_input = st.chat_input("Say something to Gaurav...")

if user_input:
    with st.chat_message("user", avatar="periwinkle.png"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.api_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    )

    with st.chat_message("assistant", avatar="gaurav.jpg"):
        message_placeholder = st.empty()
        
        with st.spinner("Gaurav is typing..."):
            full_response = ""
            token_string = ""
            api_success = False
            
            # API COST OPTIMIZER 2: Dynamic Rolling Context Ceiling Window
            MAX_HISTORY_TURNS = 4
            if len(st.session_state.api_history) > MAX_HISTORY_TURNS:
                payload = st.session_state.api_history[-MAX_HISTORY_TURNS:]
            else:
                payload = st.session_state.api_history

            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=payload,
                    config=config
                )
                full_response = response.text
                
                if response.usage_metadata:
                    in_t = response.usage_metadata.prompt_token_count
                    out_t = response.usage_metadata.candidates_token_count
                else:
                    in_t, out_t = 0, 0
                    
                token_string = f"⚡ Usage Check: {in_t} in | {out_t} out tokens"
                api_success = True
            except APIError as api_err:
                if api_err.code == 429:
                    st.error("🚨 **Gaurav is out of breath, bro!** Give him 15 seconds to catch his breath.")
                else:
                    st.error(f"Error connecting to API: {api_err.message}")

        if api_success:
            # Word-by-word local typing loop (Consumes 0 extra backend API tokens)
            animated_text = ""
            for word in full_response.split(" "):
                animated_text += word + " "
                message_placeholder.markdown(animated_text + "▌")
                time.sleep(0.03)
            
            # Print static clean text and tracking metrics footer
            message_placeholder.markdown(
                f"{full_response}\n\n<span class='token-footer'>{token_string}</span>", 
                unsafe_allow_html=True
            )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "token_info": token_string
            })
            
            st.session_state.api_history.append(
