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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cssvd_model.keras',
                                      compile=False,
                                      custom_objects={
                                          'EfficientNetPreprocessing': EfficientNetPreprocessing
                                      })

model = load_model()
OPTIMAL_THRESHOLD = 0.65
CLASS_NAMES = ['cssvd', 'healthy']

LANGUAGES = {
    "English": "eng",
    "Twi": "twi",
    "Dagbani": "dagbani",
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

def play_audio(lang_folder, result_class):
    url = f"https://raw.githubusercontent.com/Abdul-pandev/try-host/main/audio/{lang_folder}/{result_class}.mp3"
    st.markdown(
        f'<audio autoplay controls style="width:100%;">'
        f'<source src="{url}" type="audio/mp3">'
        f'</audio>',
        unsafe_allow_html=True
    )
    else:
        st.warning("Audio file not found.")

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

    with st.spinner('Analyzing...'):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        probability = model.predict(img_array)[0][0]
        predicted = 'healthy' if probability > OPTIMAL_THRESHOLD else 'cssvd'
        confidence = probability if probability > OPTIMAL_THRESHOLD else 1 - probability
        result = RESULTS[predicted]

    st.divider()

    if result['type'] == 'success':
        st.success(f"{result['icon']} {result['title']}")
    else:
        st.error(f"{result['icon']} {result['title']}")

    st.metric('Confidence', f'{confidence*100:.1f}%')
    st.info(result['advice'])

    play_audio(lang_folder, predicted)

    with st.expander('See full breakdown'):
        st.progress(float(1 - probability), text=f'CSSVD: {(1-probability)*100:.1f}%')
        st.progress(float(probability), text=f'Healthy: {probability*100:.1f}%')
