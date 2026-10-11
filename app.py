import os
import io
import hashlib
import secrets
import sqlite3
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from fpdf import FPDF

st.set_page_config(page_title="OPTI-HEMO Anemia Screening", page_icon="🩸", layout="centered")

# ---- Custom website colours ----
st.markdown("""
<style>
.stApp {
    background-color:F3EFFF;
}
</style>
""", unsafe_allow_html=True)

DB_PATH = "anemia.db"
MODEL_PATH = "model.pkl"

# ---- Quality thresholds (adjust here if good photos get rejected) ----
BLUR_MIN = 30.0        # Laplacian variance
BRIGHT_MIN = 60.0      # mean gray
BRIGHT_MAX = 220.0
REFLECT_MAX = 0.03     # fraction of near-white pixels
ROI_MIN = 0.03         # fraction of image that must be conjunctiva

FEATURE_ORDER = ["mean_r", "mean_g", "mean_b", "lab_a", "lab_b",
                 "redness", "rg_ratio", "texture", "vessel"]


# =====================================================================
# DATABASE
# =====================================================================
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, age INTEGER, gender TEXT,
        phone TEXT UNIQUE, salt TEXT, pw_hash TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, ts TEXT, hb REAL, risk TEXT, conf REAL)""")
    con.commit()
    con.close()


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 100000).hex()


def register_user(name, age, gender, phone, pw):
    salt = secrets.token_hex(16)
    try:
        con = db()
        con.execute("INSERT INTO users(name,age,gender,phone,salt,pw_hash) VALUES(?,?,?,?,?,?)",
                    (name, age, gender, phone, salt, hash_pw(pw, salt)))
        con.commit()
        con.close()
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(phone, pw):
    con = db()
    row = con.execute("SELECT id,name,age,gender,phone,salt,pw_hash FROM users WHERE phone=?",
                      (phone,)).fetchone()
    con.close()
    if row and hash_pw(pw, row[5]) == row[6]:
        return {"id": row[0], "name": row[1], "age": row[2], "gender": row[3], "phone": row[4]}
    return None


def save_record(user_id, hb, risk, conf):
    con = db()
    con.execute("INSERT INTO records(user_id,ts,hb,risk,conf) VALUES(?,?,?,?,?)",
                (user_id, datetime.now().strftime("%Y-%m-%d %H:%M"), hb, risk, conf))
    con.commit()
    con.close()


def get_records(user_id):
    con = db()
    df = pd.read_sql_query("SELECT ts AS Date, hb AS 'Hb (g/dL)', risk AS Risk, conf AS Confidence "
                           "FROM records WHERE user_id=? ORDER BY id", con, params=(user_id,))
    con.close()
    return df


# =====================================================================
# IMAGE PIPELINE  (blocks 3 and 4 of the diagram)
# =====================================================================
def load_image(file):
    im = Image.open(file)
    im = ImageOps.exif_transpose(im).convert("RGB")
    rgb = np.array(im)
    h, w = rgb.shape[:2]
    if w > 640:  # Resize
        rgb = cv2.resize(rgb, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
    return rgb


def preprocess(rgb):
    """Denoise + luminance normalisation (keeps colour ratios, which carry the Hb signal)."""
    out = cv2.bilateralFilter(rgb, 7, 40, 40)
    gray_mean = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).mean()
    gain = float(np.clip(130.0 / max(gray_mean, 1.0), 0.7, 1.5))
    out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def segment_roi(rgb):
    """Conjunctiva segmentation: red/pink region (high a* in LAB), largest blob."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    a = lab[:, :, 1]
    thr = max(np.percentile(a, 70), 135)
    mask = (a >= thr) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 245)
    mask = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros(mask.shape, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def assess_quality(raw_rgb, mask):
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    reflect = float((raw_rgb.min(axis=2) > 245).mean())
    roi = float(mask.mean())
    checks = [
        ("Sharpness (blur)", f"{blur:.0f}", blur >= BLUR_MIN, "Image blurry. Hold steady and focus."),
        ("Brightness", f"{bright:.0f}", BRIGHT_MIN <= bright <= BRIGHT_MAX,
         "Too dark or too bright. Use even light, no flash."),
        ("Reflection / glare", f"{reflect*100:.1f}%", reflect <= REFLECT_MAX,
         "Glare spots found. Avoid direct light reflection."),
        ("ROI (conjunctiva) found", f"{roi*100:.1f}%", roi >= ROI_MIN,
         "Conjunctiva not clearly visible. Pull lower eyelid down and come closer."),
    ]
    metrics = {"blur": blur, "bright": bright, "reflect": reflect, "roi": roi}
    return checks, metrics


def extract_features(rgb, mask):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    px = rgb[mask].astype(np.float32)
    r, g, b = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    edges = cv2.Canny(gray, 50, 150)
    return {
        "mean_r": float(r), "mean_g": float(g), "mean_b": float(b),
        "lab_a": float(lab[:, :, 1][mask].mean()),
        "lab_b": float(lab[:, :, 2][mask].mean()),
        "redness": float(r / (r + g + b + 1e-6)),
        "rg_ratio": float(r / (g + 1e-6)),
        "texture": float(gray[mask].std()),
        "vessel": float((edges[mask] > 0).mean()),
    }


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def estimate_hb(f):
    """Uses trained model.pkl if present, else an UNCALIBRATED prototype formula."""
    model = load_model()
    if model is not None:
        x = np.array([[f[k] for k in FEATURE_ORDER]])
        return float(np.clip(model.predict(x)[0], 3, 18)), "Trained ML model"
    hb = 5.0 + (f["lab_a"] - 135.0) * 0.35
    return float(np.clip(hb, 4, 17)), "Prototype formula (not clinically calibrated)"


def confidence_score(m, source):
    s_blur = min(m["blur"] / 150.0, 1.0)
    s_bright = float(np.clip(1 - abs(m["bright"] - 135) / 100.0, 0, 1))
    s_ref = 1 - min(m["reflect"] / REFLECT_MAX, 1.0)
    s_roi = min(m["roi"] / 0.10, 1.0)
    conf = (s_blur + s_bright + s_ref + s_roi) / 4.0
    if source.startswith("Prototype"):
        conf *= 0.6
    return round(float(conf), 2)


def classify(hb, gender):
    normal_cut = 13.0 if gender == "Male" else 12.0
    if hb >= normal_cut:
        return "Normal", "🟢"
    if hb >= 11.0:
        return "Mild", "🟡"
    if hb >= 8.0:
        return "Moderate", "🟠"
    return "Severe", "🔴"


RECOMMENDATIONS = {
    "Normal": ["Maintain a balanced diet with iron and vitamin C rich foods.",
               "Repeat screening every few months to monitor.",
               "No immediate medical action needed."],
    "Mild": ["Add iron rich foods: spinach, beetroot, dates, jaggery, pulses, eggs.",
             "Take vitamin C (lemon, amla, orange) with meals to improve absorption.",
             "Get a blood test (CBC) to confirm and re-screen in 2 to 4 weeks."],
    "Moderate": ["Consult a doctor soon for a blood test (CBC) and treatment advice.",
                 "Iron rich diet; supplements only as prescribed by a doctor.",
                 "Follow up screening within 1 to 2 weeks."],
    "Severe": ["Please visit a doctor or hospital as early as possible.",
               "Do not rely on this app; get a lab hemoglobin test urgently.",
               "Watch for dizziness, breathlessness, fast heartbeat."],
}


def make_pdf(info, overlay_rgb):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "OPTI-HEMO Anemia Screening Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Date: " + datetime.now().strftime("%d-%m-%Y %H:%M"), ln=True)
    pdf.cell(0, 8, "Patient: " + info["name"], ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Estimated Hb: {info['hb']:.1f} g/dL", ln=True)
    pdf.cell(0, 9, f"Risk level: {info['risk']}", ln=True)
    pdf.cell(0, 9, f"Confidence: {info['conf']:.2f} (0 to 1)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, "Method: " + info["source"], ln=True)
    pdf.ln(2)
    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    buf.seek(0)
    pdf.image(buf, w=90)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for line in RECOMMENDATIONS[info["risk"]]:
        pdf.multi_cell(0, 7, "- " + line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Disclaimer: This is a screening estimate from a student prototype, "
                         "not a medical diagnosis. Confirm with a laboratory hemoglobin test.",
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


# =====================================================================
# UI
# =====================================================================
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def page_home():
    st.title("🩸 OPTI-HEMO")
    st.subheader("Smartphone anemia screening")
    st.write("Who is using the app?")
    c1, c2 = st.columns(2)
    if c1.button("👤 Guest User\n(one-time screening)", use_container_width=True):
        st.session_state.mode = "guest"
        st.rerun()
    if c2.button("📋 Registered User\n(track & monitor)", use_container_width=True):
        st.session_state.mode = "auth"
        st.rerun()


def page_auth():
    st.title("Registered User")
    if st.button("⬅ Back"):
        st.session_state.mode = "home"
        st.rerun()
    tab_login, tab_reg = st.tabs(["Login", "New registration"])
    with tab_login:
        phone = st.text_input("Phone number", key="l_phone")
        pw = st.text_input("Password", type="password", key="l_pw")
        if st.button("Login"):
            u = login_user(phone.strip(), pw)
            if u:
                st.session_state.user = u
                st.session_state.mode = "user"
                st.rerun()
            else:
                st.error("Wrong phone number or password. New user? Use the registration tab.")
    with tab_reg:
        name = st.text_input("Full name", key="r_name")
        age = st.number_input("Age", 1, 120, 20, key="r_age")
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="r_gender")
        phone = st.text_input("Phone number (used for login)", key="r_phone")
        pw = st.text_input("Create password (min 4 characters)", type="password", key="r_pw")
        if st.button("Register"):
            if not name.strip() or not phone.strip() or len(pw) < 4:
                st.warning("Please fill name, phone and a password of at least 4 characters.")
            elif register_user(name.strip(), int(age), gender, phone.strip(), pw):
                st.success("Registered! Now go to the Login tab and login.")
            else:
                st.error("This phone number is already registered. Please login.")


def page_screening(user):
    st.subheader("New screening")
    if user:
        gender, name = user["gender"], user["name"]
        st.caption(f"Patient: {name}")
    else:
        name = "Guest"
        gender = st.selectbox("Gender (used for Hb cut-off)", ["Female", "Male"])

    st.info("Pull the lower eyelid down gently, show the inner pink/red part (palpebral conjunctiva), "
            "use good natural light, no flash, keep the phone steady and close.")

    t1, t2 = st.tabs(["📷 Camera", "🖼 Upload photo"])
    with t1:
        cam = st.camera_input("Capture conjunctiva image")
    with t2:
        upl = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    file = cam if cam is not None else upl
    if file is None:
        return

    # ---- Pipeline ----
    with st.spinner("Processing image..."):
        raw = load_image(file)
        pre = preprocess(raw)
        mask = segment_roi(pre)
        checks, metrics = assess_quality(raw, mask)

    overlay = pre.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)

    with st.expander("Processing steps", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.image(raw, caption="1. Original (resized)", use_container_width=True)
        c2.image(pre, caption="2. Preprocessed", use_container_width=True)
        c3.image(overlay, caption="3. ROI (green)", use_container_width=True)
        st.markdown("**Image quality assessment**")
        for label, val, ok, _msg in checks:
            st.write(("✅ " if ok else "❌ ") + f"{label}: {val}")

    failed = [c for c in checks if not c[2]]
    if failed:
        st.error("Poor quality image. Please recapture.")
        for c in failed:
            st.write("• " + c[3])
        st.write("Click **Clear photo** (camera) or upload a new image to try again.")
        return

    # ---- Good quality: feature extraction -> Hb -> risk ----
    feats = extract_features(pre, mask)
    hb, source = estimate_hb(feats)
    risk, emoji = classify(hb, gender)
    conf = confidence_score(metrics, source)

    st.success("Good quality image. Result:")
    m1, m2, m3 = st.columns(3)
    m1.metric("Hemoglobin", f"{hb:.1f} g/dL")
    m2.metric("Anemia risk", f"{emoji} {risk}")
    m3.metric("Confidence", f"{conf:.2f}")
    st.progress(conf)
    st.caption("Method: " + source)

    with st.expander("Extracted features"):
        st.dataframe(pd.DataFrame([feats]).round(3).T.rename(columns={0: "value"}))

    st.markdown("### Recommendations")
    for line in RECOMMENDATIONS[risk]:
        st.write("• " + line)

    info = {"name": name, "hb": hb, "risk": risk, "conf": conf, "source": source}
    st.download_button("⬇ Download PDF report", make_pdf(info, overlay),
                       file_name="anemia_report.pdf", mime="application/pdf")

    if user:
        if st.button("💾 Save to my history"):
            save_record(user["id"], round(hb, 1), risk, conf)
            st.success("Saved. Check the My History tab.")
    else:
        st.caption("Guest mode: result is not saved. Register to track your history.")


def page_history(user):
    st.subheader("My history")
    df = get_records(user["id"])
    if df.empty:
        st.info("No saved screenings yet. Do a screening and press 'Save to my history'.")
        return
    st.line_chart(df.set_index("Date")["Hb (g/dL)"])
    st.dataframe(df, use_container_width=True)


def main():
    init_db()
    ss("mode", "home")
    ss("user", None)
    mode = st.session_state.mode

    if mode == "home":
        page_home()
    elif mode == "auth":
        page_auth()
    elif mode == "guest":
        top = st.columns([3, 1])
        top[0].title("Guest screening")
        if top[1].button("⬅ Exit"):
            st.session_state.mode = "home"
            st.rerun()
        page_screening(None)
    elif mode == "user":
        user = st.session_state.user
        top = st.columns([3, 1])
        top[0].title(f"Hi, {user['name']}")
        if top[1].button("Logout"):
            st.session_state.user = None
            st.session_state.mode = "home"
            st.rerun()
        tab_a, tab_b = st.tabs(["🩺 New screening", "📈 My history"])
        with tab_a:
            page_screening(user)
        with tab_b:
            page_history(user)


main()