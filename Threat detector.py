import streamlit as st
import openai
import base64
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re

# 1. Page Configuration & Custom CSS for Modern College Aesthetic
st.set_page_config(page_title="SafeChat AI Analyzer", page_icon="🛡️", layout="centered")

# Custom UI Styling: Pastel accents, rounded cards, clean look
st.markdown("""
    <style>
    .stApp {
        background-color: #F7FAFC;
    }
    h1 {
        color: #4A5568 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #718096;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #3182CE !important;
        border-bottom-color: #3182CE !important;
    }
    div.stButton > button:first-child {
        background-color: #3182CE;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.15s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2B6CB0;
        transform: translateY(-1px);
    }
    .feedback-box {
        background-color: #EDF2F7;
        padding: 15px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allowed_html=True)

st.title("🛡️ SafeChat AI Analyzer")
st.subheader("Manipulation, love-bombing aur sugar-coated red flags ko pehchanein.")
st.caption("✨ Designed for student safety. Your chats are processed securely and never saved.")

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

# 3. Sidebar Configuration (Secure API Secrets Check)
st.sidebar.header("⚙️ Configuration")
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("🔑 API Key securely loaded from Secrets!")
else:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📖 Test with Examples")
selected_sample = st.sidebar.selectbox("Choose a sample scenario to load:", list(SAMPLE_CHATS.keys()))

# 4. Advanced System Prompt for Hinglish Extraction
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

# Helper functions
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

# 5. Streamlit Tabs Interface
tab1, tab2 = st.tabs(["📝 Copy-Paste Chat", "📸 Upload Screenshot"])

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
                    elif analyze_image_button and uploaded_image:
                    base64_image = encode_image(uploaded_image)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Analyze this WhatsApp screenshot written in Hinglish text:"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ]
                    )
                    ai_output = response.choices.message.content
                
                if ai_output:
                    st.session_state.analysis_result = ai_output
                else:
                    st.warning("Please provide input data before clicking analyze.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# 7. Render Output Dashboard from State
if st.session_state.analysis_result:
    output = st.session_state.analysis_result
    st.success("Analysis Complete!")
    
    metrics = extract_metrics(output)
    
    st.write("### 📊 Psychological Risk Profile")
    fig = go.Figure(go.Bar(
        x=list(metrics.values()),
        y=list(metrics.keys()),
        orientation='h',
        marker=dict(color=['#E53E3E' if v > 60 else '#DD6B20' if v > 30 else '#38A169' for v in metrics.values()])
    ))
    fig.update_layout(
        xaxis=dict(title="Risk Level (%)", range=[0, 100]), 
        yaxis=dict(autorange="reversed"), 
        height=280, 
        margin=dict(l=5, r=5, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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

    # 8. Interactive User Feedback Block
    st.markdown('<div class="feedback-box">', unsafe_allowed_html=True)
    st.write("##### 💬 Kya AI analysis ne sender ke sahi intentions ko catch kiya?")
    
    if not st.session_state.feedback_submitted:
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("👍 Yes, it was accurate", use_container_width=True, key="fb_yes"):
                st.session_state.feedback_submitted = True
                st.rerun()
        with col_no:
            if st.button("👎 No, it missed the context", use_container_width=True, key="fb_no"):
                st.session_state.feedback_submitted = True
                st.rerun()
    else:
        st.info("Thank you for your feedback! It helps us train a safer model.")
    st.markdown('</div>', unsafe_allowed_html=True)

# 9. Fixed Interface Footer: Verified Indian Support Helplines
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
