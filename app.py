import streamlit as st

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="FarmEye",
    page_icon="🌱",
    layout="wide"
)

# =============================
# THEME STATE
# =============================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

st.session_state.theme = "dark" if dark_mode else "light"


# =============================
# CSS THEMES
# =============================
def load_css(theme):
    if theme == "light":
        css = """
        <style>
        body {
            background-color: #F8F5EE;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #1B5E20;
        }

        .subtitle {
            font-size: 18px;
            color: #444;
        }

        .card {
            background: white;
            padding: 20px;
            border-radius: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            transition: 0.3s;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .metric-box {
            background: #ffffff;
            padding: 15px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        }
        </style>
        """
    else:
        css = """
        <style>
        body {
            background-color: #121212;
            color: white;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #43A047;
        }

        .subtitle {
            font-size: 18px;
            color: #ccc;
        }

        .card {
            background: #1E1E1E;
            padding: 20px;
            border-radius: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }

        .metric-box {
            background: #1E1E1E;
            padding: 15px;
            border-radius: 20px;
            text-align: center;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


load_css(st.session_state.theme)


# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.markdown("## 🌱 FarmEye")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🔬 Detection",
        "🗺 Disease Intelligence",
        "📊 Analytics",
        "🌍 Impact",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")


# =============================
# DASHBOARD
# =============================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">FarmEye</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Cocoa Disease Intelligence Platform</div>', unsafe_allow_html=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    for col, title, value in zip(
        [col1, col2, col3, col4],
        ["Farmers Reached", "Total Scans", "Accuracy", "Hotspots"],
        ["800K+", "5,360", "83.8%", "32"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{value}</h3>
                <p>{title}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">
        <h3>👨‍🌾 Farmers</h3>
        <p>Detect cocoa disease early using AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🗺 Disease Intelligence</h3>
        <p>Real-time national disease monitoring system.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🏛 Institutions</h3>
        <p>Data-driven decisions for cocoa sustainability.</p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# DETECTION PAGE
# =============================
elif page == "🔬 Detection":
    st.title("🔬 Cocoa Disease Detection")

    uploaded = st.file_uploader("Upload cocoa leaf image", type=["jpg", "png", "jpeg"])

    if uploaded:
        st.image(uploaded, use_container_width=True)

        st.success("🟢 Healthy Plant (Demo Output)")
        st.info("Confidence: 94%")

        st.markdown("""
        <div class="card">
            <h3>✓ Scan Recorded</h3>
            <p>Your scan contributes to Ghana’s cocoa disease intelligence network.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================
# DISEASE INTELLIGENCE
# =============================
elif page == "🗺 Disease Intelligence":
    st.title("🗺 National Disease Intelligence Platform")

    st.markdown("""
    <div class="card">
        <h3>Real-time CSSVD Surveillance</h3>
        <p>Farmer-generated data powering national cocoa disease insights.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Scans", "4,521")
    col2.metric("CSSVD Cases", "387")
    col3.metric("Hotspots", "32")

    st.info("📍 Map placeholder (you will plug Folium or Plotly here)")

    st.markdown("""
    <div class="card">
        <h3>Who Uses This Intelligence?</h3>
        <p>🏛 COCOBOD • 🏭 Cocoa Buyers • 🌍 NGOs & Donors</p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# ANALYTICS
# =============================
elif page == "📊 Analytics":
    st.title("📊 Platform Analytics")

    st.metric("Total Scans", "5,360")
    st.metric("Healthy Plants", "4,900")
    st.metric("CSSVD Detections", "387")
    st.metric("Active Districts", "14")


# =============================
# IMPACT
# =============================
elif page == "🌍 Impact":
    st.title("🌍 FarmEye Impact")

    st.markdown("""
    <div class="card">
        <h3>800,000+ Farmers Reached</h3>
        <h3>32 Disease Hotspots Tracked</h3>
        <h3>387 Early Detections</h3>
        <p>Every scan strengthens Ghana’s cocoa intelligence system.</p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# ABOUT
# =============================
elif page == "ℹ About":
    st.title("ℹ About FarmEye")

    st.write("""
    FarmEye is an AI-powered cocoa disease detection and intelligence platform 
    designed to improve Ghana’s cocoa productivity through real-time insights.
    """)
