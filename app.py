import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from model_utils import EfficientNetPreprocessing
import base64
import os

st.set_page_config(
    page_title='CocoaGuard GH',
    page_icon='🌿',
    layout='centered'
)

# ── Load models ───────────────────────────────────────────────────

@st.cache_resource
def load_cocoa_checker():
    return tf.keras.models.load_model('cocoa_checker.keras',
                                      compile=False)

@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model('cssvd_model.keras',
                                      compile=False,
                                      custom_objects={
                                          'EfficientNetPreprocessing': EfficientNetPreprocessing
                                      })

disease_model = load_disease_model()
cocoa_model   = load_cocoa_checker()


# ── Constants ─────────────────────────────────────────────────────
DISEASE_THRESHOLD = 0.65
COCOA_THRESHOLD   = 0.65

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
        'advice': 'Infection confirmed. Consult your extension officer.',
        'type': 'error'
    }
}

# ── Audio ─────────────────────────────────────────────────────────
def play_audio(lang_folder, result_class):
    url = f"https://raw.githubusercontent.com/Abdul-Pandev/try-host/main/Audio/{lang_folder}/{result_class}.mp3"
    st.audio(url, format="audio/mp3")

# ── UI ────────────────────────────────────────────────────────────
st.title('🌿 CocoaGuard GH')
st.caption('CSSVD Early Detection - Powered by Sankofa Intelligence')
st.divider()

lang_label = st.selectbox(
    "🌍 Select your language before you continue",
    options=list(LANGUAGES.keys())
)
lang_folder = LANGUAGES[lang_label]
st.success(f"✓ Language set to: {lang_label}")
st.divider()

uploaded = st.file_uploader(
    'Upload a photo of a cocoa leaf, stem or pod',
    type=['jpg', 'jpeg', 'png'],
    help='Take a clear, close-up photo of the leaf you want to check'
)

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded photo', use_column_width=True)

    # ── Step 1: Check if image is cocoa ───────────────────────────
    with st.spinner('Checking image...'):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        try:
            cocoa_prob = cocoa_model.predict(img_array)[0][0]
            is_cocoa = cocoa_prob <= COCOA_THRESHOLD
        except Exception as e:
            st.error(f"Cocoa check failed: {e}")
            st.stop()

if not is_cocoa:
    st.divider()

    st.warning(
        "🍃 This does not appear to be a cocoa leaf or stem. Please upload a clear photo of a cocoa plant."
    )

    play_audio(lang_folder, "non_cocoa")

    st.stop()
    

    # ── Step 2: Run disease detection ─────────────────────────────
    with st.spinner('Analyzing for CSSVD...'):
        try:
            probability = disease_model.predict(img_array)[0][0]
            predicted  = 'healthy' if probability > DISEASE_THRESHOLD else 'cssvd'
            confidence = probability if probability > DISEASE_THRESHOLD else 1 - probability
            result     = RESULTS[predicted]
        except Exception as e:
            st.error(f"Disease detection failed: {e}")
            st.stop()

    st.divider()

    if result['type'] == 'success':
        st.success(f"{result['icon']} {result['title']}")
    else:
        st.error(f"{result['icon']} {result['title']}")

    st.metric('Confidence', f'{confidence*100:.1f}%')
    st.info(result['advice'])

    play_audio(lang_folder, predicted)

    with st.expander('See full breakdown'):
        st.progress(float(1 - probability),
                    text=f'CSSVD: {(1-probability)*100:.1f}%')
        st.progress(float(probability),
                    text=f'Healthy: {probability*100:.1f}%')
