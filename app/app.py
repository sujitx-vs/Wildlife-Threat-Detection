import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model # type:ignore

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="EcoGuard AI",
    page_icon="🌿",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS — Liquid Glass Morphism
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Body ── */
:root {
    --blue: #60a5fa;
    --blue-dim: rgba(96,165,250,0.15);
    --blue-glow: rgba(96,165,250,0.35);
    --glass: rgba(255,255,255,0.07);
    --glass-border: rgba(255,255,255,0.14);
    --blur: blur(18px) saturate(180%);
    --text-muted: rgba(255,255,255,0.5);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #050b14 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #fff !important;
}

/* Animated gradient background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(59,130,246,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(29,78,216,0.1) 0%, transparent 55%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: var(--blur) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 20px !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #fff, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
[data-testid="stSidebar"] h3 {
    font-size: 10px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--blue) !important; -webkit-text-fill-color: var(--blue) !important;
    margin-top: 24px !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    font-size: 13px !important; color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    line-height: 1.7 !important;
}

/* ── Main content column ── */
[data-testid="stMain"] {
    background: transparent !important;
}
.main .block-container {
    padding: 3rem 2.5rem 2rem !important;
    max-width: 720px !important;
}

/* ── Typography ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 42px !important; font-weight: 800 !important;
    letter-spacing: -1px !important; line-height: 1.1 !important;
    background: linear-gradient(135deg, #fff 30%, #60a5fa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}
[data-testid="stCaptionContainer"] p {
    font-size: 14px !important; color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
}
hr { border-color: var(--glass-border) !important; margin: 1.2rem 0 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--glass) !important;
    border: 1.5px dashed rgba(96,165,250,0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: var(--blur) !important;
    padding: 24px !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue) !important;
    background: rgba(96,165,250,0.06) !important;
    box-shadow: 0 0 30px rgba(96,165,250,0.1) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderDropInstructions"] {
    color: rgba(255,255,255,0.7) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.7) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploaderDropInstructions"] {
    font-size: 15px !important; font-weight: 500 !important;
}
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
    background: rgba(96,165,250,0.12) !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    color: var(--blue) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Audio player ── */
audio {
    width: 100% !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.05) !important;
    filter: invert(1) hue-rotate(210deg) brightness(0.85) !important;
}

/* ── Analyze Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 60%, #1d4ed8 100%) !important;
    color: #050b14 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important; font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    box-shadow: 0 4px 30px rgba(96,165,250,0.35) !important;
    transition: all 0.25s !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(96,165,250,0.45) !important;
}

/* ── Alert boxes (error / success / warning / info) ── */
[data-testid="stAlert"] {
    border-radius: 16px !important;
    backdrop-filter: var(--blur) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 17px !important; font-weight: 700 !important;
}
/* Error → threat */
[data-testid="stAlert"][data-baseweb="notification"]:has(svg[data-icon="error"]),
div[data-testid="stAlert"].st-emotion-cache-x3xypo {
    background: rgba(248,113,113,0.1) !important;
    border: 1px solid rgba(248,113,113,0.35) !important;
    color: #f87171 !important;
}
/* Success → safe */
div[data-testid="stAlert"]:has([data-baseweb*="success"]) {
    background: rgba(96,165,250,0.1) !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    color: #60a5fa !important;
}
/* Warning */
div[data-testid="stAlert"]:has([data-baseweb*="warning"]) {
    background: rgba(251,191,36,0.08) !important;
    border: 1px solid rgba(251,191,36,0.3) !important;
}
/* Info */
div[data-testid="stAlert"]:has([data-baseweb*="info"]) {
    background: rgba(96,165,250,0.08) !important;
    border: 1px solid rgba(96,165,250,0.25) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    backdrop-filter: var(--blur) !important;
}
[data-testid="stMetric"] label {
    font-size: 10px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 36px !important; font-weight: 800 !important;
    color: var(--blue) !important;
    -webkit-text-fill-color: var(--blue) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--blue) !important;
    font-size: 14px !important;
}
[data-testid="stSpinner"] > div {
    border-top-color: var(--blue) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "../models/threat_detector.keras"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌿 EcoGuard AI")

st.sidebar.markdown("""
EcoGuard AI is a deep learning system that analyzes environmental audio recordings and identifies potential threats to wildlife habitats.

### Threat Sounds
* 🪚 Chainsaw
* 🚗 Vehicle Noise
* 🏭 Industrial Activity

### Natural Sounds
* 🐦 Birds
* 🌧 Rain
* 💨 Wind
* 🦋 Wildlife
""")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌿 EcoGuard AI")
st.caption("Wildlife Threat Detection · Environmental Audio Analysis")
st.markdown("---")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_audio = st.file_uploader(
    "Drop a .wav recording or click to browse",
    type=["wav"]
)

# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------

if uploaded_audio is not None:
    st.audio(uploaded_audio)

st.markdown("")

if st.button("🔍 Analyze Recording", use_container_width=True):

    with st.spinner("Processing audio through CNN pipeline..."):

        y, sr = librosa.load(uploaded_audio, sr=22050)
        y = librosa.util.fix_length(y, size=int(5 * 22050))

        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram = librosa.power_to_db(spect, ref=np.max)

        sample_spectro = np.array(spectrogram)
        sample_spectro = np.expand_dims(sample_spectro, axis=0)
        sample_spectro = np.expand_dims(sample_spectro, axis=-1)

        prediction = model.predict(sample_spectro, verbose=0)
        probability = float(prediction[0][0])

    st.markdown("---")

    if probability >= 0.5:
        confidence = probability * 100
        st.error("🚨 Threat Detected — Human or Industrial Activity")
        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.warning(
            "Potential human or industrial activity was detected in this recording. "
            "Consider alerting field rangers for on-site verification."
        )
    else:
        confidence = (1 - probability) * 100
        st.success("🌿 No Threat — Natural Environment")
        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.info(
            "The recording appears to contain natural environmental sounds "
            "consistent with an undisturbed habitat zone."
        )

else:
    if uploaded_audio is None:
        st.info("Upload a .wav audio recording above to begin analysis.")