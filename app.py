import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from model_utils import EfficientNetPreprocessing

st.set_page_config(
    page_title='CocoaGuard GH',
    page_icon='🌿',
    layout='centered'
)

# ── Custom Styling ────────────────────────────────────────────────
st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App background */
.stApp {
    background: linear-gradient(135deg, #f4fff4 0%, #eef8ee 100%);
    font-family: 'Segoe UI', sans-serif;
}

/* Hero section */
.hero-box {
    background: linear-gradient(135deg, #14532d, #166534);
    padding: 2.5rem;
    border-radius: 24px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.hero-sub {
    font-size: 1.05rem;
    opacity: 0.92;
    line-height: 1.6;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #166534;
    border-radius: 18px;
    padding: 1rem;
    background: white;
}

/* Metric container */
[data-testid="metric-container"] {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid #ecf0ec;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Image card */
.result-card {
    padding: 1rem;
    border-radius: 18px;
    background: white;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

/* Audio player */
audio {
    width: 100%;
    margin-top: 12px;
}

/* Progress */
.stProgress > div > div {
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────

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

# ── Constants ─────────────────────────────────────────────────────

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
        'title': 'Healthy Plant',
        'advice': 'No signs of CSSVD detected. Monitor your farm regularly.',
        'type': 'success'
    },

    'cssvd': {
        'icon': '🚨',
        'title': 'CSSVD Detected',
        'advice': 'Infection confirmed. Consult your extension officer immediately.',
        'type': 'error'
    }
}

# ── Audio ─────────────────────────────────────────────────────────

def play_audio(lang_folder, result_class):
    url = f"https://raw.githubusercontent.com/Abdul-Pandev/try-host/main/Audio/{lang_folder}/{result_class}.mp3"
    st.audio(url, format="audio/mp3")

# ── Hero Section ──────────────────────────────────────────────────

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🌿 CocoaGuard GH</div>
    <div class="hero-sub">
        AI-powered early detection system for Cocoa Swollen Shoot Virus Disease (CSSVD),
        helping cocoa farmers identify infections early and protect crop yields across Ghana.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Language Selection ────────────────────────────────────────────

st.markdown("### 🌍 Select Preferred Language")

lang_label = st.selectbox(
    "",
    options=list(LANGUAGES.keys())
)

lang_folder = LANGUAGES[lang_label]

st.info(f"🌐 Selected Language: {lang_label}")

st.divider()

# ── Upload Section ────────────────────────────────────────────────

st.markdown("### 📸 Upload Cocoa Plant Image")

uploaded = st.file_uploader(
    "Upload a clear photo of a cocoa leaf, stem, or pod",
    type=['jpg', 'jpeg', 'png'],
    help='Use a well-lit and focused image for better detection accuracy.'
)

# ── Prediction Flow ───────────────────────────────────────────────

if uploaded:

    image = Image.open(uploaded).convert('RGB')

    st.markdown("""
    <div class="result-card">
        <h4>Uploaded Sample</h4>
    </div>
    """, unsafe_allow_html=True)

    st.image(image, use_column_width=True)

    # ── Step 1: Cocoa Check ───────────────────────────────────────

    with st.spinner('🌿 Verifying cocoa plant...'):

        img = image.resize((224, 224))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        try:
            cocoa_prob = cocoa_model.predict(img_array)[0][0]
            is_cocoa = cocoa_prob <= COCOA_THRESHOLD

        except Exception as e:
            st.error(f"Cocoa verification failed: {e}")
            st.stop()

    # ── Non Cocoa ─────────────────────────────────────────────────

    if not is_cocoa:

        st.divider()

        st.warning(
            "🍃 This does not appear to be a cocoa plant. Please upload a clear cocoa leaf, stem, or pod image."
        )

        play_audio(lang_folder, "non_cocoa")

        st.stop()

    # ── Disease Detection ─────────────────────────────────────────

    with st.spinner('🔬 Analyzing for CSSVD...'):

        try:
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

        except Exception as e:
            st.error(f"Disease detection failed: {e}")
            st.stop()

    st.divider()

    # ── Result Display ────────────────────────────────────────────

    if result['type'] == 'success':
        st.success(f"{result['icon']} {result['title']}")
    else:
        st.error(f"{result['icon']} {result['title']}")

    st.metric(
        "Detection Confidence",
        f"{confidence * 100:.1f}%"
    )

    st.info(result['advice'])

    # ── Audio Feedback ────────────────────────────────────────────

    play_audio(lang_folder, predicted)

    # ── Detailed Breakdown ────────────────────────────────────────

    with st.expander('📊 View Detailed Prediction Breakdown'):

        st.progress(
            float(1 - probability),
            text=f'CSSVD Probability: {(1 - probability) * 100:.1f}%'
        )

        st.progress(
            float(probability),
            text=f'Healthy Probability: {probability * 100:.1f}%'
        )

# ── Footer ────────────────────────────────────────────────────────

st.divider()

st.caption(
    "Built with AI for sustainable cocoa farming and early disease detection in Ghana 🇬🇭"
)
