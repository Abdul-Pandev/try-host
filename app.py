# app.py — CocoaGuard GH CSSVD Detector
# Sankofa Intelligence | Ghana | 2026

# app.py — CocoaGuard GH CSSVD Detector
# Sankofa Intelligence | Ghana | 2026

import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import streamlit as st
import numpy as np
from PIL import Image



# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="CocoaGuard GH",
    # page_icon="🌿",
    layout="centered"
)

# ── Load model (cached — only loads once per session) ─────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'cssvd_model.keras',
        compile=False,
        custom_objects={
            "EfficientNetPreprocessing": EfficientNetPreprocessing
        }
    )

model = load_model()

# ── Class names — must match alphabetical order from your labels.txt ──────
# Verify this order by opening labels.txt from your Colab export.
# The order here MUST match exactly, otherwise predictions will be wrong.
CLASS_NAMES = ['cssvd', 'healthy']

# ── Result messages for each class ────────────────────────────────────────
RESULTS = {
    'healthy': {
        'icon': '✅',
        'title': 'Healthy Plant',
        'advice': (
            'No signs of CSSVD detected in this image. '
            'Continue monitoring your farm regularly and keep records of any changes.'
        ),
        'type': 'success'
    },
    'cssvd': {
        'icon': '🚨',
        'title': 'CSSVD Detected',
        'advice': (
            'Signs of Cocoa Swollen Shoot Virus Disease detected. '
            'Contact your extension officer immediately. Do not move plant material '
            'from this area to other parts of your farm.'
        ),
        'type': 'error'
    }
}

# ── UI Header ─────────────────────────────────────────────────────────────
st.title("CocoaGuard GH")
st.caption("CSSVD Early Detection — Powered by Sankofa Intelligence")
st.divider()

st.markdown(
    """
    Upload a clear, close-up photo of a **cocoa leaf or stem**.  
    The AI model will analyse it and tell you whether signs of  
    **Cocoa Swollen Shoot Virus Disease (CSSVD)** are present.
    """
)

# ── File uploader ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a photo of a cocoa leaf or stem",
    type=['jpg', 'jpeg', 'png'],
    help="Take a clear, close-up photo of the leaf you want to check"
)

# ── Inference block ───────────────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded photo', use_column_width=True)

    with st.spinner('Analysing image...'):
        # Resize to model input size
        img = image.resize((224, 224))

        # Convert to float array and normalise to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
        predictions = model.predict(img_array, verbose=0)

        # For binary sigmoid output (single value)
        # predictions[0][0] is probability of class index 1 (healthy)
        if predictions.shape[-1] == 1:
            # Binary sigmoid: output is probability of being 'healthy' (index 1)
            prob_healthy = float(predictions[0][0])
            prob_cssvd = 1.0 - prob_healthy
            probs = [prob_cssvd, prob_healthy]
            idx = int(prob_healthy > 0.65)   # optimal threshold from notebook
        else:
            # Softmax / multi-class output
            probs = predictions[0].tolist()
            idx = int(np.argmax(probs))

        predicted = CLASS_NAMES[idx]
        confidence = probs[idx] * 100

    result = RESULTS[predicted]

    st.divider()

    # ── Result display ────────────────────────────────────────────────────
    if result['type'] == 'success':
        st.success(f"{result['icon']}  **{result['title']}**")
    else:
        st.error(f"{result['icon']}  **{result['title']}**")

    st.metric(label="Confidence", value=f"{confidence:.1f}%")
    st.info(result['advice'])

    # ── Full probability breakdown ────────────────────────────────────────
    with st.expander("See full score breakdown"):
        for name, score in zip(CLASS_NAMES, probs):
            st.progress(float(score), text=f"{name}:  {score * 100:.1f}%")

    # ── Low confidence warning ────────────────────────────────────────────
    if confidence < 60:
        st.warning(
            "⚠️  **Low confidence result.** The model is uncertain about this image. "
            "Try a clearer, closer photo of the leaf in good lighting."
        )

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CocoaGuard GH is a research tool and should not replace advice from a "
    "qualified agricultural extension officer. Sankofa Intelligence | Ghana | 2026"
)
