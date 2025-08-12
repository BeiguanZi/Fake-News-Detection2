import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import nltk
import string
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import tempfile
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

st.set_page_config(page_title="Fake News Detection System", layout="centered")

try:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return " ".join([stemmer.stem(word) for word in tokens if word not in stop_words])

def safe_load(path, label):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load {label}: {e}")
        return None

# Load models
vectorizer = safe_load("tfidf_vectorizer.pkl", "TF-IDF")
lr_model = safe_load("logistic_model.pkl", "Logistic Regression")
nb_model = safe_load("naive_bayes_model.pkl", "Naive Bayes")
svm_model = safe_load("svm_model.pkl", "SVM")

if vectorizer is None:
    st.stop()

models = {}
if lr_model: models["Logistic Regression"] = lr_model
if nb_model: models["Naive Bayes"] = nb_model
if svm_model: models["SVM"] = svm_model
if len(models) == 3:
    models["Hybrid (LR + NB + SVM)"] = None

st.title("📰 Fake News Detection System")
st.markdown("Use machine learning to classify news as real or fake, with visualization and export support.")

uploaded = st.file_uploader("📄 Upload a TXT or PDF file (optional):", type=["txt", "pdf"])
file_text = ""
if uploaded:
    if uploaded.type == "text/plain":
        file_text = uploaded.read().decode("utf-8", errors="ignore")
    elif uploaded.type == "application/pdf":
        try:
            with fitz.open(stream=uploaded.read(), filetype="pdf") as doc:
                file_text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            st.warning(f"PDF extract error: {e}")

user_input = st.text_area("📝 Enter the news content to analyze:", value=file_text, height=200)

model_choice = st.selectbox("🤖 Choose a model:", list(models.keys()))
if model_choice == "Hybrid (LR + NB + SVM)":
    w_lr = st.slider("Weight: Logistic Regression", 0.0, 1.0, 0.33)
    w_nb = st.slider("Weight: Naive Bayes", 0.0, 1.0, 0.33)
    w_svm = max(0.0, 1.0 - w_lr - w_nb)
    st.caption(f"Total: LR {w_lr:.2f} + NB {w_nb:.2f} + SVM {w_svm:.2f}")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Run Prediction"):
    if not user_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    X = vectorizer.transform([preprocess(user_input)])

    def get_prob(model):
        return model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 1 / (1 + np.exp(-model.decision_function(X))[0])

    if model_choice.startswith("Hybrid"):
        p = (w_lr * get_prob(lr_model) + w_nb * get_prob(nb_model) + w_svm * get_prob(svm_model))
    else:
        model = models[model_choice]
        p = get_prob(model)

    label = "🔴 FAKE" if p >= 0.5 else "🟢 REAL"
    st.markdown(
        f"""
        <div style='padding:1em; border-left:4px solid {"#cc0000" if p >= 0.5 else "#007a33"}'>
            <h4>{label} NEWS</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(int(p * 100))
    st.metric("Fake probability", f"{p*100:.2f}%")

    tokens = vectorizer.get_feature_names_out()
    weights = X.toarray()[0]
    top_tokens = [tokens[i] for i in weights.argsort()[::-1][:10] if weights[i] > 0]

    st.markdown("#### 🧠 Top Tokens")
    st.write(", ".join(top_tokens))

    highlighted = user_input
    for word in top_tokens:
        highlighted = re.sub(
            rf"(?i)\b{re.escape(word)}\b",
            f"<mark style='background-color:#fff2cc; font-weight:bold'>{word}</mark>",
            highlighted
        )

    st.markdown("<div style='white-space: pre-wrap;'>" + highlighted + "</div>", unsafe_allow_html=True)

    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_choice,
        "Prediction": label,
        "Fake Probability": round(p, 4),
        "Text": user_input,
        "Top Tokens": ", ".join(top_tokens)
    })

if st.session_state.history:
    st.subheader("📄 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = BytesIO()
    df.to_csv(csv, index=False)
    csv.seek(0)
    st.download_button("📥 Download CSV", csv, "predictions.csv")

    pdf_buffer = BytesIO()
    avg = df.groupby("Model")["Fake Probability"].mean()
    fig, ax = plt.subplots()
    avg.plot(kind="bar", ax=ax, color="#cc0000")
    ax.set_title("Average Fake Probability by Model")
    ax.set_ylabel("Probability")
    fig.tight_layout()
    chart_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(chart_path.name)

    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        name='CustomStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6
    )

    story = []
    story.append(Paragraph("<para align='center'><font size=20 color='navy'><b>📄 Fake News Detection Report</b></font></para>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Image(chart_path.name, width=400, height=200))
    story.append(Spacer(1, 12))

    for _, row in df.iterrows():
        text = row['Text']
        for token in row['Top Tokens'].split(", "):
            text = re.sub(rf"(?i)\b{re.escape(token)}\b", f"<font color='red'><b>{token.upper()}</b></font>", text)

        story.append(Paragraph(f"<b>Timestamp:</b> {row['Timestamp']}", custom_style))
        story.append(Paragraph(f"<b>Model:</b> {row['Model']}", custom_style))
        story.append(Paragraph(f"<b>Prediction:</b> {row['Prediction']}", custom_style))
        story.append(Paragraph(f"<b>Fake Probability:</b> {row['Fake Probability']*100:.2f}%", custom_style))
        story.append(Paragraph("<b>Text:</b>", custom_style))
        story.append(Paragraph(text, ParagraphStyle(
            name='BodyText',
            parent=custom_style,
            fontSize=11,
            leading=16,
            spaceBefore=6,
            spaceAfter=12,
            textColor=colors.HexColor('#000000')
        )))
        story.append(Spacer(1, 10))

    doc.build(story)
    pdf_buffer.seek(0)
    st.download_button("📄 Download Report as PDF", pdf_buffer, file_name="fake_news_report.pdf", mime="application/pdf")

    if st.button("🧹 Clear Prediction History"):
        st.session_state.history.clear()
        st.rerun()
