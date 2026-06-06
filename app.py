import streamlit as st
import anthropic, base64, json, os
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="CSSVD Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #F0F4F0;
}
[data-testid="stSidebar"] {
    background: #0D2B12;
}
[data-testid="stSidebar"] * {
    color: #C8E6C9 !important;
}
.stButton > button {
    background: #1B5E20;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    width: 100%;
    padding: 0.6rem;
}
.stButton > button:hover {
    background: #2E7D32;
    color: white;
}
.result-box {
    padding: 1rem;
    border-radius: 10px;
    margin-top: 1rem;
}
.detected-box {
    background: #FFEBEE;
    border-left: 4px solid #E53935;
}
.healthy-box {
    background: #E8F5E9;
    border-left: 4px solid #43A047;
}
.hist-card {
    background: white;
    border-radius: 10px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    border: 1px solid #E0E0E0;
}
</style>
""", unsafe_allow_html=True)

# ── History helpers ────────────────────────────
HISTORY_FILE = Path("scan_history.json")
IMAGES_DIR   = Path("history_images")
IMAGES_DIR.mkdir(exist_ok=True)

def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []

def save_to_history(img_bytes, filename, result, lang):
    img_path = IMAGES_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    img_path.write_bytes(img_bytes)
    history = load_history()
    history.insert(0, {
        "timestamp":  datetime.now().strftime("%d %b %Y, %H:%M"),
        "image_path": str(img_path),
        "filename":   filename,
        "detected":   result["detected"],
        "confidence": result["confidence"],
        "title":      result["title"],
        "diagnosis":  result["diagnosis"],
        "action":     result["action"],
        "language":   lang,
    })
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

# ── Language config ────────────────────────────
LANGUAGES = {
    "English": {
        "prompt": "Respond in English.",
        "scan_btn": "🔬 Scan for CSSVD",
        "upload_label": "Upload cocoa leaf or stem photo",
    },
    "Twi": {
        "prompt": "Respond entirely in Twi (Akan).",
        "scan_btn": "🔬 Hwehwɛ CSSVD",
        "upload_label": "De kookoo foto ba",
    },
    "Ewe": {
        "prompt": "Respond entirely in Ewe.",
        "scan_btn": "🔬 Biam CSSVD",
        "upload_label": "Tsɔ koko foto va",
    },
}

SYSTEM_PROMPT = """You are an expert plant pathologist specialising
in Cocoa Swollen Shoot Virus Disease (CSSVD) in Ghana.
Analyse the uploaded cocoa image and respond ONLY with
valid JSON (no markdown fences):
{{
  "detected": true | false,
  "confidence": 0-100,
  "title": "short result title",
  "diagnosis": "2-3 sentence explanation",
  "action": "1-2 sentence recommended action"
}}
{lang_instruction}"""

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Theobroma_cacao_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-030.jpg/200px-Theobroma_cacao_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-030.jpg",
               use_column_width=True)
    st.markdown("## 🌱 CocoaGuard GH")
    st.markdown("AI-powered CSSVD early detection")
    st.divider()

    lang = st.radio("Language / Kasa / Gbegbɔgblɔ",
                      list(LANGUAGES.keys()))

    st.divider()
    history = load_history()
    st.markdown(f"**Total scans:** {len(history)}")
    detected_count = sum(1 for h in history if h["detected"])
    st.markdown(f"**CSSVD detected:** {detected_count}")
    st.markdown(f"**Healthy scans:** {len(history) - detected_count}")

# ── Main tabs ──────────────────────────────────
tab_scan, tab_history = st.tabs(["🔬 Scan", "📋 History"])

with tab_scan:
    st.markdown("### Upload a cocoa image")
    cfg = LANGUAGES[lang]
    uploaded = st.file_uploader(
        cfg["upload_label"],
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption=uploaded.name,
                       use_column_width=True)
        with col2:
            if st.button(cfg["scan_btn"]):
                img_bytes = uploaded.read()
                b64 = base64.b64encode(img_bytes).decode()
                with st.spinner("Analysing image..."):
                    client = anthropic.Anthropic()
                    msg = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=600,
                        system=SYSTEM_PROMPT.format(
                            lang_instruction=cfg["prompt"]
                        ),
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image",
                                 "source": {
                                     "type": "base64",
                                     "media_type": uploaded.type,
                                     "data": b64
                                 }},
                                {"type": "text",
                                 "text": "Analyse this image for CSSVD."}
                            ]
                        }]
                    )
                raw = msg.content[0].text.strip()
                result = json.loads(raw)
                save_to_history(img_bytes, uploaded.name,
                                  result, lang)

                box_cls = ("detected-box" if result["detected"]
                           else "healthy-box")
                icon = "⚠️" if result["detected"] else "✅"
                st.markdown(f"""
<div class="result-box {box_cls}">
  <h4>{icon} {result['title']}</h4>
  <p>{result['diagnosis']}</p>
  <p><strong>Action:</strong> {result['action']}</p>
  <p><small>Confidence: {result['confidence']}%</small></p>
</div>""", unsafe_allow_html=True)

with tab_history:
    history = load_history()
    if not history:
        st.info("No scans yet. Upload a cocoa image to get started.")
    else:
        col_a, col_b = st.columns(2)
        for i, h in enumerate(history):
            col = col_a if i % 2 == 0 else col_b
            with col:
                with st.expander(
                    f"{'⚠️' if h['detected'] else '✅'} "
                    f"{h['title']} · {h['timestamp']}"
                ):
                    img_path = Path(h["image_path"])
                    if img_path.exists():
                        st.image(str(img_path),
                                  use_column_width=True)
                    st.markdown(h["diagnosis"])
                    st.markdown(f"**Action:** {h['action']}")
                    st.markdown(
                        f"Confidence: **{h['confidence']}%** · "
                        f"Language: {h['language']}"
                    )
