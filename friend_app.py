import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError

# 1. Standard Page Configuration
st.set_page_config(
    page_title="Vibe with Gaurav",
    page_icon="🌸",
    layout="centered"
)

# Minimalist, Clean Pastel Friendship Canvas Style
st.markdown("""
<style>
    @import url('https://googleapis.com');
    
    /* Soft, high-contrast stable pastel canvas background */
    .stApp {
        background: linear-gradient(135deg, #fff2f5 0%, #fffbfd 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Clean Radiant Header Typography */
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #ff4e50, #e14eca);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .subtitle-text {
        color: #5c3d46 !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }

    /* Clear technical text for usage tracking metrics */
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
    st.markdown("## 🌸 Gaurav's Garden")
    st.markdown("Your truest friend! Vent about your day, or have a laugh in casual Hinglish, Hindi, or Gujarati. 💐")
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

# 6. UPDATED BEHAVIORAL INSTRUCTIONS: Added strict question-asking loop rule
friend_personality = (
    "You are Gaurav, a close, supportive, ultra-funny, and loyal best friend. "
    "Keep your answers short, crisp, casual, and highly conversational—exactly like a friend texting on WhatsApp. "
    "You are completely multilingual. Reply natively in whichever language the user texts you in: English, Hindi (हिंदी), or Gujarati (ગુજરાતી). "
    "Match the user's conversational flow perfectly. If they use Hinglish or Gujlish, respond dynamically using the exact same style. "
    "Use plenty of casual text phrases (like 'bro', 'yaar', 'chill', 'sahi hai'). "
    "CRITICAL ENGAGEMENT RULE: You must ALWAYS end your response with an engaging, casual follow-up question to keep the 'vibe' alive and continue the chat. Never just answer a statement and stop. "
    "EMOJI RULES: "
    "- Always add 1-3 emojis per message to feel natural, but don't overdo it. "
    "- Since your chat screen has a floral garden theme, occasionally drop a flower emoji (🌸, 🌹, 🌻, 💐) when greeting or sending positive vibes. "
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

# 8. Render Standard Chat History Cards
for message in st.session_state.messages:
    avatar_icon = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "token_info" in message:
            st.markdown(f"<span class='token-footer'>{message['token_info']}</span>", unsafe_allow_html=True)

# 9. Process Active Client Message Inputs
if user_input := st.chat_input("Say something to Gaurav..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.api_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    )

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Gaurav is typing..."):
            full_response = ""
            token_string = ""
            api_success = False
            
            # API COST OPTIMIZER 2: Rolling Context Ceiling Window
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
                types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            )
