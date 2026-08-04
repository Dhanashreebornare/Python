import streamlit as st
import openai
import base64
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re
import datetime
import pandas as pd

# 1. Page Configuration & Cloud Engine Integration
st.set_page_config(page_title="SafeChat AI Analyzer", page_icon="🛡️", layout="centered")

st.title("🛡️ SafeChat AI Analyzer")
st.subheader("Manipulation, love-bombing aur sugar-coated red flags ko pehchanein.")
st.caption("✨ Designed for student safety. Your chats are processed securely and logged safely.")

# 2. Pre-loaded Sample Chat Library (Hinglish)
SAMPLE_CHATS = {
    "--- Select a Sample Scenario ---": "",
    "🚨 Scenario 1: Love-Bombing & Isolation (High Risk)": 
        "Sender: Yaar honestly, jab se kal tumse mila hoon, I feel like you are my soulmate. Pata nahi log dating apps pe kaise time waste karte hain, main toh bas tumhare sath hi poori life spend karna chahta hoon. Suno, apne college ke dosto ko hamare baare me mat batana abhi, unhe bohot jalan hoti hai tumse. Wo bas hamara relation kharab karna chahte hain. Aaj raat ko akele milte hain na please?",
    
    "💸 Scenario 2: Sugar-Coated Financial/Crypto Trap (High Risk)": 
        "Sender: Babe, agle mahine tumko ek super luxury vacation pe le chalunga, you deserve the best! Actually, abhi main ek bohot badi crypto trade execute kar raha hoon mere mama ki company ke sath. 24 hours me paise double hone ki guarantee hai. Ek slot bacha hai jo 10 mins me close ho jayega par mere paas abhi ₹15,000 kam pad rahe hain wallet me load karne ke liye. Tum jaldi se mujhe GPay ya UPI kar do na? Kal subah tak direct ₹30,000 wapas kar dunga, trust me baby.",
    
    "🟢 Scenario 3: Healthy & Safe Communication (Safe)": 
        "Sender: Hey! Bas ye check karne ke liye message kiya ki tum study group ke baad safely ghar pahonch gayi na? Jab bhi next week free ho batana, saath me psychology presentation complete kar lenge. Koi jaldbaazi nahi hai, pehle tum apne weekend exams pe focus karo! All the best!"
}

# 3. Sidebar Configuration (Secure API Secrets Engine Execution)
st.sidebar.header("⚙️ Configuration")
if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"].strip() != "":
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("🔒 System Secure: Key Loaded")
else:
    api_key = None
    st.sidebar.error("❌ Configuration Error: OPENAI_API_KEY missing from cloud secrets dashboard.")

st.sidebar.markdown("---")
st.sidebar.header("📖 Test with Examples")
selected_sample = st.sidebar.selectbox("Choose a sample scenario to load:", list(SAMPLE_CHATS.keys()))

# 4. System Prompt Design
SYSTEM_PROMPT = """
You are an expert psychological profiler and communication safety assistant specialized in Indian dating culture and digital interactions. Your job is to protect young Indian women and college students from digital manipulation, grooming, "sugar-coated" traps, love-bombing, financial scams, or isolation tactics.

The input conversation will be provided in Hinglish (a mix of Hindi and English words typed in the Roman script). You must deeply understand the contextual meaning of Hinglish slang, expressions, and emotional undertones.

Analyze the text or screenshot and format your response EXACTLY as structured below using markdown headers. Keep the explanations simple, using clear language (mix of simple English/Hinglish) so it is universally accessible.

### 🚨 Threat Level Assessment
[🟢 SAFE / 🟡 CAUTION / 🔴 HIGH RISK] - Give a brief 1-sentence reason.

### 📊 Metric Scores
Love Bombing: [Score 0-100]
Isolation Tactics: [Score 0-100]
Urgency & Pressure: [Score 0-100]
Financial Risk: [Score 0-100]
Deception/Guilt-Tripping: [Score 0-100]

### 🔍 Flagged Behaviors & Tactics
*   **[Tactic Name]**: "Quote from chat" -> Explain the psychology behind this tactic and why it is a red flag in this context.

### 💡 What to Do Next
*   Provide actionable, practical safety advice tailored to this specific scenario.
"""

# Helper Functions
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

def extract_metrics(text):
    metrics = {
        "Love Bombing": 0, "Isolation Tactics": 0, 
        "Urgency & Pressure": 0, "Financial Risk": 0, 
        "Deception/Guilt-Tripping": 0
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

# 4.5 Persistent Database Data Logging Function
def log_data_to_sheets(chat_text, threat_rating, user_review):
    if "connections" in st.secrets and "gsheets" in st.secrets.connections:
        try:
            from streamlit_gsheets import GSheetsConnection
            conn = st.connection("gsheets", type=GSheetsConnection)
            
            # Read current sheet matrix
            try:
                df = conn.read(ttl=0)
            except Exception:
                df = pd.DataFrame(columns=["Timestamp", "Input_Content", "Threat_Level", "Feedback"])
            
            new_data = pd.DataFrame([{
                "Timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "Input_Content": str(chat_text)[:500], # Keep strings light for faster parsing
                "Threat_Level": str(threat_rating),
                "Feedback": str(user_review)
            }])
            
            updated_df = pd.concat([df, new_data], ignore_index=True)
            conn.update(data=updated_df)
        except Exception:
            pass # Silent execution prevents UI blockage during server spikes

# 5. Streamlit Tabs Interface
tab1, tab2 = st.tabs(["📝 Copy-Paste Chat", "📸 Upload Screenshot"])

# Track current active content globally for analytics processing layers
if "current_chat_content" not in st.session_state:
    st.session_state.current_chat_content = ""

with tab1:
    default_text = SAMPLE_CHATS[selected_sample] if selected_sample != "--- Select a Sample Scenario ---" else ""
    user_text = st.text_area("Paste the conversation or sample text here:", value=default_text, height=180)
    analyze_text_button = st.button("Analyze Text", type="primary", key="txt_btn")

with tab2:
    uploaded_image = st.file_uploader("Upload a WhatsApp Screenshot (PNG/JPG):", type=["png", "jpg", "jpeg"])
    analyze_image_button = st.button("Analyze Screenshot", type="primary", key="img_btn")

# Initialize session state variables
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# 6. Processing Execution
if analyze_text_button or analyze_image_button:
    if not api_key:
        st.error("Please configure your OpenAI API Key inside Streamlit Cloud Secrets dashboard settings to run live analysis.")
    else:
        st.session_state.feedback_submitted = False
        client = openai.OpenAI(api_key=api_key)
        
        with st.spinner("Analyzing communication patterns..."):
            try:
                ai_output = ""
                if analyze_text_button and user_text:
                    st.session_state.current_chat_content = user_text
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Analyze this text chat:\n\n{user_text}"}
                        ]
                    )
                    ai_output = response.choices.message.content
                    
git add "Threat detector.py"
git commit -m "Cleaned image handling layout and set complete chart coordinate arrays"
git push origin main
