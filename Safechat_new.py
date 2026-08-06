import streamlit as st
from google import genai
from google.genai import types
import base64
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re
import datetime
import pandas as pd
from PIL import Image
import io

# 1. Page Configuration & Custom UI Color Theme Injection
st.set_page_config(page_title="SafeChat AI Analyzer", page_icon="🛡️", layout="centered")

# Custom CSS to force deep charcoal backgrounds, glowing amber safety highlights, and crisp readable typography
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #E2E8F0;
        }
        h1, h2, h3, .stSubheader {
            color: #F59E0B !important;
        }
        .stButton>button {
            background-color: #D97706 !important;
            color: white !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #B45309 !important;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #1F2937;
            padding: 8px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #9CA3AF !important;
            border-radius: 4px;
            padding: 6px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #D97706 !important;
            color: white !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_with_html=True)

st.title("🛡️ SafeChat AI Analyzer")
st.subheader("Manipulation, love-bombing aur sugar-coated red flags ko pehchanein.")
st.caption("✨ Designed for student safety. Your chats are processed securely and logged safely.")

# 2. Pre-loaded Sample Chat Library (Hinglish)
SAMPLE_CHATS = {
    "--- Select a Sample Scenario ---": "",
    "🚨 Scenario 1: Love-Bombing & Isolation (High Risk)": "Sender: Yaar honestly, jab se kal tumse mila hoon, I feel like you are my soulmate. Pata nahi log dating apps pe kaise time waste karte hain, main toh bas tumhare sath hi poori life spend karna chahta hoon. Suno, apne college ke dosto ko hamare baare me mat batana abhi, unhe bohot jalan hoti hai tumse. Wo bas hamara relation kharab karna chahte hain. Aaj raat ko akele milte hain na please?",
    "💸 Scenario 2: Sugar-Coated Financial/Crypto Trap (High Risk)": "Sender: Babe, agle mahine tumko ek super luxury vacation pe le chalunga, you deserve the best! Actually, abhi main ek bohot badi crypto trade execute kar raha hoon mere mama ki company ke sath. 24 hours me paise double hone ki guarantee hai. Ek slot bacha hai jo 10 mins me close ho jayega par mere paas abhi ₹15,000 kam pad rahe hain wallet me load karne ke liye. Tum jaldi se mujhe GPay ya UPI kar do na? Kal subah tak direct ₹30,000 wapas kar dunga, trust me baby.",
    "🟢 Scenario 3: Healthy & Safe Communication (Safe)": "Sender: Hey! Bas ye check karne ke liye message kiya ki tum study group ke baad safely ghar pahonch gayi na? Jab bhi next week free ho batana, saath me psychology presentation complete kar lenge. Koi jaldbaazi nahi hai, pehle tum apne weekend exams pe focus karo! All the best!"
}

# 3. Sidebar Configuration (Secure Google AI Client Check)
st.sidebar.header("⚙️ Configuration")
if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip() != "":
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("🔒 System Secure: Google AI Key Loaded")
else:
    client = None
    st.sidebar.error("❌ Configuration Error: GEMINI_API_KEY missing from cloud secrets dashboard.")

st.sidebar.markdown("---")
st.sidebar.header("📖 Test with Examples")
selected_sample = st.sidebar.selectbox("Choose a sample scenario to load:", list(SAMPLE_CHATS.keys()))

# 4. System Prompt Design incorporating updated safety tracking metrics
SYSTEM_PROMPT = """
You are an expert psychological profiler and communication safety assistant specialized in Indian dating culture and digital interactions. Your job is to protect young Indian women and college students from digital manipulation, grooming, "sugar-coated" traps, love-bombing, financial scams, or isolation tactics. The input conversation will be provided in Hinglish (a mix of Hindi and English words typed in the Roman script) or via audio-visual files. You must deeply understand the contextual meaning of Hinglish slang, expressions, and emotional undertones. Analyze the text, screenshots, audio, or video and format your response EXACTLY as structured below using markdown headers. Keep the explanations simple, using clear language (mix of simple English/Hinglish) so it is universally accessible.

### 🚨 Threat Level Assessment
[🟢 SAFE / 🟡 CAUTION / 🔴 HIGH RISK] - Give a brief 1-sentence reason.

### 📊 Metric Scores
Love Bombing: [Score 0-100]
Isolation Tactics: [Score 0-100]
Urgency & Pressure: [Score 0-100]
Financial Risk: [Score 0-100]
Deception/Guilt-Tripping: [Score 0-100]
Grooming Intent: [Score 0-100]
Emotional Blackmail: [Score 0-100]
Coercive Control: [Score 0-100]

### 🔍 Flagged Behaviors & Tactics
* **[Tactic Name]**: "Quote or reference from input" -> Explain the psychology behind this tactic and why it is a red flag in this context.

### 💡 What to Do Next
* Provide actionable, practical safety advice tailored to this specific scenario.
"""

# Helper Functions
def extract_metrics(text):
    metrics = {
        "Love Bombing": 0,
        "Isolation Tactics": 0,
        "Urgency & Pressure": 0,
        "Financial Risk": 0,
        "Deception/Guilt-Tripping": 0,
        "Grooming Intent": 0,
        "Emotional Blackmail": 0,
        "Coercive Control": 0
    }
    for key in metrics.keys():
        match = re.search(rf"{re.escape(key)}:\s*(\d+)", text)
        if match:
            metrics[key] = int(match.group(1))
    return metrics

def extract_threat_level(text):
    if "🔴 HIGH RISK" in text:
        return "HIGH RISK"
    elif "🟡 CAUTION" in text:
        return "CAUTION"
    return "SAFE"

def generate_pdf(analysis_text):
    pdf_path = "SafeChat_Safety_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor("#1A365D"), spaceAfter=15)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=8)
    alert_style = ParagraphStyle('Alert', parent=styles['Normal'], fontSize=11, leading=16, textColor=colors.HexColor("#C53030"), spaceAfter=8)
    
    story.append(Paragraph("🛡️ SafeChat AI Safety Report", title_style))
    story.append(Spacer(1, 10))
    lines = analysis_text.split('\n')
    for line in lines:
        if line.startswith("###"):
            header_text = line.replace("###", "").strip()
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>{header_text}</b>", ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2C5282"), spaceAfter=5)))
        elif line.strip().startswith("*"):
            bullet_text = line.replace("*", "").strip()
            story.append(Paragraph(f"• {bullet_text}", body_style))
        elif line.strip():
            story.append(Paragraph(line.strip(), body_style))
            
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>🚨 Emergency Support Resources (India)</b>", ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#C53030"), spaceAfter=5)))
    story.append(Paragraph("• National Cyber Crime Helpline: Call 1930 (For financial scams/online harassment)", alert_style))
    story.append(Paragraph("• National Emergency Number: Call 112", alert_style))
    story.append(Paragraph("• Women Helpline: Call 1091", alert_style))
    doc.build(story)
    with open(pdf_path, "rb") as f:
        return f.read()

def log_data_to_sheets(chat_text, threat_rating, user_review):
    if "connections" in st.secrets and "gsheets" in st.secrets.connections:
        try:
            from streamlit_gsheets import GSheetsConnection
            conn = st.connection("gsheets", type=GSheetsConnection)
            try:
                df = conn.read(ttl=0)
            except Exception:
                df = pd.DataFrame(columns=["Timestamp", "Input_Content", "Threat_Level", "Feedback"])
            new_data = pd.DataFrame([{
                "Timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "Input_Content": str(chat_text)[:500],
                "Threat_Level": str(threat_rating),
                "Feedback": str(user_review)
            }])
            updated_df = pd.concat([df, new_data], ignore_index=True)
            conn.update(data=updated_df)
        except Exception:
            pass

# =====================================================================
# 5. Streamlit Tabs Interface Configuration
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📝 Copy-Paste Chat", "📸 Upload Screenshots", "🎥 Audio-Visual Input"])

if "current_chat_content" not in st.session_state:
    st.session_state.current_chat_content = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

analyze_text_button = False
analyze_image_button = False
analyze_av_button = False

with tab1:
    default_text = SAMPLE_CHATS[selected_sample] if selected_sample != "--- Select a Sample Scenario ---" else ""
    user_text = st.text_area("Paste the conversation or sample text here:", value=default_text, height=180)
    analyze_text_button = st.button("Analyze Text", type="primary", key="txt_btn")

with tab2:
    uploaded_images = st.file_uploader("Upload WhatsApp Screenshots (PNG/JPG):", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        st.write(f"📂 {len(uploaded_images)} image(s) staged for processing.")
    analyze_image_button = st.button("Analyze Staged Screenshots", type="primary", key="img_btn")

with tab3:
    uploaded_av_file = st.file_uploader("Upload Chat Recordings, Video Proofs or Voice Notes (MP4/M4A/MP3/WAV):", type=["mp4", "m4a", "mp3", "wav"])
    if uploaded_av_file:
        if uploaded_av_file.name.endswith(('mp3', 'wav', 'm4a')):
            st.audio(uploaded_av_file)
        else:
            st.video(uploaded_av_file)
    analyze_av_button = st.button("Analyze Audio-Visual Input", type="primary", key="av_btn")

# =====================================================================
# 6. Processing Execution using specialized model endpoints
# =====================================================================
if analyze_text_button or analyze_image_button or analyze_av_button:
    if not client:
        st.error("Please configure GEMINI_API_KEY inside Streamlit Cloud Secrets dashboard settings.")
    else:
        model_to_use = 'gemini-2.5-flash'
        ai_output = ""
        with st.spinner("Google AI is executing multi-modal context profiling..."):
            try:
                # ADVANCED MODEL PARAMETERS: Lowering temperature for strict psychological analysis consistency
                config = types.GenerateContentConfig(
                    temperature=0.15,
                    max_output_tokens=1500,
                    system_instruction=SYSTEM_PROMPT
                )

                if analyze_text_button and user_text:
                    st.session_state.current_chat_content = user_text
                    full_prompt = f"Analyze this text chat according to your system profile:\n\n{user_text}"
                    response = client.models.generate_content(model=model_to_use, contents=full_prompt, config=config)
                    ai_output = response.text
                
                elif analyze_image_button and uploaded_images:
                    st.session_state.current_chat_content = f"[Screenshots Uploaded: {len(uploaded_images)} file(s)]"
                    contents_payload = ["Analyze these sequential communication screenshots written in Hinglish text according to your system profile:"]
                    for img in uploaded_images:
                        img_bytes = img.read()
                        contents_payload.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                    response = client.models.generate_content(model=model_to_use, contents=contents_payload, config=config)
                    ai_output = response.text
                    
                elif analyze_av_button and uploaded_av_file:
                    st.session_state.current_chat_content = f"[Audio-Visual File: {uploaded_av_file.name}]"
                    mime_type = "video/mp4" if uploaded_av_file.name.endswith('mp4') else "audio/mp3"
                    av_bytes = uploaded_av_file.read()
                    contents_payload = [
                        "Analyze this media interaction containing conversations/recordings for safety flags according to your system profile:",
                        types.Part.from_bytes(data=av_bytes, mime_type=mime_type)
                    ]
                    response = client.models.generate_content(model=model_to_use, contents=contents_payload, config=config)
                    ai_output = response.text
                
                if ai_output:
                    st.session_state.analysis_result = ai_output
                else:
                    st.warning("Please provide input data before clicking analyze.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# =====================================================================
# 7. Render Output Dashboard from State
# =====================================================================
if st.session_state.analysis_result:
    output = st.session_state.analysis_result
    st.success("Analysis Complete!")
    metrics = extract_metrics(output)
    threat_tier = extract_threat_level(output)
    
    st.write("### 📊 Psychological Risk Profile")
    
    # Render safety scores cleanly without cutting off labels on the axis
    fig = go.Figure(go.Bar(
        x=list(metrics.values()),
        y=list(metrics.keys()),
        orientation='h',
        marker=dict(color=['#E53E3E' if v > 60 else '#DD6B20' if v > 30 else '#38A169' for v in metrics.values()])
    ))
    fig.update_layout(
        xaxis=dict(title="Risk Level (%)", range=[0, 100], gridcolor='#2D3748'),
        yaxis=dict(autorange="reversed", automargin=True),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        height=400,
        margin=dict(l=180, r=20, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(output)
    pdf_data = generate_pdf(output)
    st.markdown("---")
    st.download_button(
        label="📥 Download Full Safety Report + Emergency Helplines (PDF)",
        data=pdf_data,
        file_name="SafeChat_Safety_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # =====================================================================
    # 8. Interactive User Feedback Block
    # =====================================================================
    st.info("##### 💬 Kya AI analysis ne sender ke sahi intentions ko catch kiya?")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("👍 Yes, it was accurate", use_container_width=True, key="fb_yes"):
            log_data_to_sheets(chat_text=st.session_state.current_chat_content, threat_rating=threat_tier, user_review="Accurate Analysis")
            st.success("Logged! Thank you.")
    with col_no:
        if st.button("👎 No, it missed the context", use_container_width=True, key="fb_no"):
            log_data_to_sheets(chat_text=st.session_state.current_chat_content, threat_rating=threat_tier, user_review="Missed Context")
            st.warning("Logged! We will improve.")

# =====================================================================
# 9. Fixed Interface Footer: Verified Indian Support Helplines
# =====================================================================
st.markdown("---")
st.error("### 🚨 Emergency Support Helpline Directory (India)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Cyber Crime (Scams)", value="📞 1930")
with col2:
    st.metric(label="Women Helpline", value="📞 1091")
with col3:
    st.metric(label="National Emergency", value="📞 112")
st.caption("If you feel threatened, blackmailed, or forced, please reach out immediately. Your safety comes first.")
