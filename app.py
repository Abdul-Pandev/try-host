import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from model_utils import EfficientNetPreprocessing

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CocoaGuard GH",
    page_icon="🌿",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App Background */
.stApp {
    background: linear-gradient(180deg, #f4fff4 0%, #eef7ee 100%);
    font-family: 'Segoe UI', sans-serif;
}

/* Main spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Hero Section */
.hero {
    background:
        linear-gradient(rgba(7, 47, 24, 0.82),
        rgba(7, 47, 24, 0.82)),
        url("https://images.unsplash.com/photo-1598515213692-d9c8b8f6dd12?q=80&w=1600&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    border-radius: 28px;
    padding: 4rem 3rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 35px rgba(0,0,0,0.15);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    line-height: 1.8;
    max-width: 750px;
}

/* Glass cards */
.card {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 1.5rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.07);
    border: 1px solid rgba(255,255,255,0.4);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #166534;
    border-radius: 18px;
    padding: 1rem;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: none;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14532d 0%, #166534 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Result boxes */
.success-box {
    background: #dcfce7;
    border-left: 6px solid #16a34a;
    padding: 1rem;
    border-radius: 16px;
    color: #14532d;
    font-weight: 600;
}

.error-box {
    background: #fee2e2;
    border-left: 6px solid #dc2626;
    padding: 1rem;
    border-radius: 16px;
    color: #7f1d1d;
    font-weight: 600;
}

.warning-box {
    background: #fef3c7;
    border-left: 6px solid #f59e0b;
    padding: 1rem;
    border-radius: 16px;
    color: #78350f;
    font-weight: 600;
}

/* Audio player */
audio {
    width: 100%;
    margin-top: 10px;
}

/* Progress bar */
.stProgress > div > div {
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_cocoa_checker():
    return tf.keras.models.load_model(
        'cocoa_checker.keras',
        compile=False
    )

@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model(
        'cssvd_model.keras',
        compile=False,
        custom_objects={
            'EfficientNetPreprocessing': EfficientNetPreprocessing
        }
    )

disease_model = load_disease_model()
cocoa_model = load_cocoa_checker()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

DISEASE_THRESHOLD = 0.65
COCOA_THRESHOLD = 0.65

LANGUAGES = {
    "English": "eng",
    "Twi": "twi",
    "Dagbani": "dag",
    "Ewe": "ewe"
}

RESULTS = {
    'healthy': {
        'icon': '✅',
        'title': 'Healthy Cocoa Plant',
        'advice': 'No signs of Cocoa Swollen Shoot Virus Disease detected.',
        'type': 'success'
    },

    'cssvd': {
        'icon': '🚨',
        'title': 'CSSVD Detected',
        'advice': 'Possible infection detected. Consult your extension officer immediately.',
        'type': 'error'
    }
}

# ─────────────────────────────────────────────────────────────
# AUDIO FUNCTION
# ─────────────────────────────────────────────────────────────

def play_audio(lang_folder, result_class):
    url = f"https://raw.githubusercontent.com/Abdul-Pandev/try-host/main/Audio/{lang_folder}/{result_class}.mp3"
    st.audio(url, format="audio/mp3")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.title("🌿 CocoaGuard GH")

    st.markdown("""
    AI-powered cocoa disease detection platform helping farmers identify
    Cocoa Swollen Shoot Virus Disease early and protect crop yields.
    """)

    st.divider()

    st.subheader("🌍 Language")

    lang_label = st.selectbox(
        "Choose Language",
        options=list(LANGUAGES.keys())
    )

    lang_folder = LANGUAGES[lang_label]

    st.divider()

    st.subheader("🛡 Features")

    st.markdown("""
    ✅ Cocoa Verification  
    ✅ CSSVD Detection  
    ✅ AI-powered Analysis  
    ✅ Audio Feedback  
    ✅ Multi-language Support  
    """)

# ─────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">

    <div class="hero-title">
        CocoaGuard GH 🌿
    </div>

    <div class="hero-subtitle">
        Smart AI-powered early detection system for Cocoa Swollen Shoot Virus Disease (CSSVD),
        helping cocoa farmers across Ghana identify infections early, reduce crop losses,
        and improve sustainable cocoa production.
    </div>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────

left, right = st.columns([1.1, 0.9], gap="large")

# ─────────────────────────────────────────────────────────────
# LEFT COLUMN
# ─────────────────────────────────────────────────────────────

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📸 Upload Cocoa Plant Image")

    uploaded = st.file_uploader(
        "Upload a clear image of a cocoa leaf, pod, or stem",
        type=['jpg', 'jpeg', 'png']
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:

        image = Image.open(uploaded).convert('RGB')

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🖼 Uploaded Sample")

        st.image(image, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RIGHT COLUMN
# ─────────────────────────────────────────────────────────────

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("ℹ️ About CocoaGuard")

    st.markdown("""
    CocoaGuard GH uses Artificial Intelligence to assist cocoa farmers and
    agricultural extension officers in identifying Cocoa Swollen Shoot Virus Disease (CSSVD) early.

    Upload a cocoa plant image and the system will:

    • Verify if it is a cocoa plant  
    • Detect signs of CSSVD  
    • Provide confidence scores  
    • Deliver multilingual audio feedback  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREDICTION FLOW
# ─────────────────────────────────────────────────────────────

if uploaded:

    img = image.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Cocoa Verification
    with st.spinner("🌿 Verifying cocoa plant..."):

        cocoa_prob = cocoa_model.predict(img_array)[0][0]
        is_cocoa = cocoa_prob <= COCOA_THRESHOLD

    # Non-cocoa
    if not is_cocoa:

        st.markdown("""
        <div class="warning-box">
        🍃 This image does not appear to be a cocoa plant.
        Please upload a clear cocoa leaf, pod, or stem image.
        </div>
        """, unsafe_allow_html=True)

        play_audio(lang_folder, "non_cocoa")

        st.stop()

    # Disease Detection
    with st.spinner("🔬 Running AI disease analysis..."):

        probability = disease_model.predict(img_array)[0][0]

        predicted = (
            'healthy'
            if probability > DISEASE_THRESHOLD
            else 'cssvd'
        )

        confidence = (
            probability
            if probability > DISEASE_THRESHOLD
            else 1 - probability
        )

        result = RESULTS[predicted]

    st.markdown("<br>", unsafe_allow_html=True)

    # Result Box
    if result['type'] == 'success':

        st.markdown(f"""
        <div class="success-box">
        {result['icon']} {result['title']}
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="error-box">
        {result['icon']} {result['title']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            "Confidence Score",
            f"{confidence * 100:.1f}%"
        )

    with m2:
        st.metric(
            "Selected Language",
            lang_label
        )

    st.info(result['advice'])

    # Audio
    st.subheader("🔊 Audio Feedback")

    play_audio(lang_folder, predicted)

    # Breakdown
    with st.expander("📊 Detailed Prediction Breakdown"):

        st.progress(
            float(1 - probability),
            text=f'CSSVD Probability: {(1 - probability) * 100:.1f}%'
        )

        st.progress(
            float(probability),
            text=f'Healthy Probability: {probability * 100:.1f}%'
        )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("<br><br>", unsafe_allow_html=True)

st.caption(
    "CocoaGuard GH • AI for Sustainable Cocoa Farming in Ghana 🇬🇭"
)

