# 🌿 CocoaGuard GH

**CSSVD Early Detection for Ghanaian Cocoa Farmers**  
Powered by Sankofa Intelligence | Ghana | 2026

---

## What It Does

CocoaGuard GH is an AI-powered web app that detects **Cocoa Swollen Shoot Virus Disease (CSSVD)** from photos of cocoa leaves and stems.

A farmer or extension officer uploads a photo → the model analyses it → a result is returned in seconds: **Healthy** or **CSSVD Detected**, with a confidence score.

---

## Live App

> 🔗 **[Click here to use CocoaGuard GH](#)**  
> *(Replace this link with your Streamlit Cloud URL after deployment)*

---

## Model

- **Architecture:** EfficientNetB0 (transfer learning, ImageNet weights)
- **Training:** Two-phase fine-tuning on a labelled Ghanaian cocoa dataset
- **Classes:** `cssvd` | `healthy`
- **Test accuracy:** 83.80% (threshold 0.5) → 85% (threshold 0.65)
- **Export:** TensorFlow Lite (.tflite) + Keras (.keras) for web deployment
- **Framework:** TensorFlow / Keras

---

## Project Structure

```
cocoaguard-mvp/
├── app.py                  # Streamlit web application
├── cssvd_model.keras       # Trained CSSVD classifier (add this file)
├── requirements.txt        # Python dependencies for Streamlit Cloud
├── README.md               # This file
└── .streamlit/
    └── config.toml         # App theme and colour settings
```

---

## Local Setup

```bash
# 1. Clone this repo
git clone https://github.com/YOUR-USERNAME/cocoaguard-mvp.git
cd cocoaguard-mvp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained model file to the folder
#    (download cssvd_model.keras from Google Drive)

# 4. Run the app
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push all files (including `cssvd_model.keras`) to this GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, branch `main`, main file `app.py`
5. Click **Deploy**

See the full Deployment Guide (PDF) for step-by-step instructions including Git LFS for large model files and the Google Drive model hosting option.

---

## Important Notes

- The classification threshold is set to **0.65** (optimised from notebook evaluation)
- For results below 60% confidence, the app warns the user to retake the photo
- This tool is a research aid and does not replace professional agricultural advice

---

## About

**Sankofa Intelligence** — building AI for African agriculture.  
Developed as part of the Capstone Project, Code4Security Fellowship — Blossom Academy, supported by WFP & KOICA.

---

*Confidential — Sankofa Intelligence*
