import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime
from io import BytesIO
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from fpdf import FPDF

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Fake News Detection System", layout="centered")

# ------------------------------
# NLTK Setup (Stopwords + Stemmer)
# ------------------------------
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {
        'a','an','the','and','or','but','if','while','with','of','to','in','on','for','at','by','from',
        'is','are','was','were','be','been','being','as','that','this','these','those','it','its','you',
        'your','yours','he','she','they','them','we','us','our','ours'
    }

stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

def safe_load(path: str, name: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {name} at {path}. Error: {e}")
        return None

# ------------------------------
# Load Models & TF-IDF
# ------------------------------
tfidf = safe_load("tfidf_vectorizer.pkl", "TF-IDF Vectorizer")
lr_model = safe_load("logistic_model.pkl", "Logistic Regression")
nb_model = safe_load("naive_bayes_model.pkl", "Naive Bayes")
svm_model = safe_load("svm_model.pkl", "SVM")

if tfidf is None:
    st.stop()

model_dict = {}
if lr_model: model_dict["Logistic Regression"] = lr_model
if nb_model: model_dict["Naive Bayes"] = nb_model
if svm_model: model_dict["SVM"] = svm_model
if len(model_dict) == 3:
    model_dict["Hybrid (LR + NB + SVM)"] = None

if not model_dict:
    st.error("No models available. Please upload model and vectorizer files.")
    st.stop()
# ------------------------------
# Title & Upload
# ------------------------------
st.title("📰 Fake News Detection System")
st.markdown("Use machine learning to classify news content as real or fake, with keyword highlighting.")

uploaded_file = st.file_uploader("📂 Upload TXT or PDF file (optional):", type=["txt", "pdf"])

file_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/pdf":
        try:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                file_text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            st.warning(f"Could not extract PDF: {e}")

# ------------------------------
# Model Selection
# ------------------------------
user_input = st.text_area("📝 Enter the news content to analyze:", value=file_text, height=200)
options = list(model_dict.keys())
default_idx = options.index("Hybrid (LR + NB + SVM)") if "Hybrid (LR + NB + SVM)" in options else 0
model_option = st.selectbox("🤖 Choose a model:", options, index=default_idx)

# For Hybrid: show sliders
if model_option == "Hybrid (LR + NB + SVM)":
    w_lr = st.slider("Logistic Regression weight", 0.0, 1.0, 0.33, step=0.01)
    w_nb = st.slider("Naive Bayes weight", 0.0, 1.0, 0.33, step=0.01)
    remaining = max(0.0, 1.0 - w_lr - w_nb)
    w_svm = min(1.0, remaining)
    st.caption(f"Hybrid weights → LR: {w_lr:.2f}, NB: {w_nb:.2f}, SVM: {w_svm:.2f}")

# Init history
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Prediction
# ------------------------------
if st.button("Run Prediction", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    cleaned_input = preprocess(user_input)
    x_input = tfidf.transform([cleaned_input])

    if model_option == "Hybrid (LR + NB + SVM)":
        def get_prob(model):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(x_input)[0][1]
            else:
                decision = model.decision_function(x_input)
                return 1 / (1 + np.exp(-decision))[0]

        prob_lr = get_prob(lr_model)
        prob_nb = get_prob(nb_model)
        prob_svm = get_prob(svm_model)
        total_w = w_lr + w_nb + w_svm
        fake_prob = (w_lr * prob_lr + w_nb * prob_nb + w_svm * prob_svm) / total_w if total_w > 0 else 0.5
    else:
        model = model_dict[model_option]
        if hasattr(model, "predict_proba"):
            fake_prob = model.predict_proba(x_input)[0][1]
        else:
            decision = model.decision_function(x_input)
            fake_prob = 1 / (1 + np.exp(-decision))[0]

    label = "🔴 FAKE" if fake_prob >= 0.5 else "🟢 REAL"
    st.markdown(
        f"""<div style='padding:1em; border-left:4px solid {"#cc0000" if fake_prob>=0.5 else "#007a33"}; 
        background:{"#ffe6e6" if fake_prob>=0.5 else "#e6fff2"}'>
        <h4>{label} NEWS</h4></div>""", unsafe_allow_html=True
    )

    st.markdown("#### 📊 Fake Probability")
    st.progress(int(fake_prob * 100))
    st.metric("Estimated FAKE probability", f"{fake_prob * 100:.2f}%")
    # ------------------------------
    # Token Highlighting
    # ------------------------------
    feature_names = tfidf.get_feature_names_out()
    input_vec = x_input.toarray()[0]
    top_idx = input_vec.argsort()[::-1][:10]
    highlight_words = [feature_names[i] for i in top_idx if input_vec[i] > 0]

    st.markdown("#### 🧠 Top TF-IDF Tokens")
    st.write(", ".join(highlight_words) if highlight_words else "(No non-zero TF-IDF tokens)")

    highlighted_text = user_input
    for word in highlight_words:
        pattern = re.compile(rf"(?i)\\b{re.escape(word)}\\b")
        highlighted_text = pattern.sub(
            rf"<span style=\"background-color:#ffe6e6; color:#c00\"><b>{word}</b></span>",
            highlighted_text
        )

    st.markdown("""<div style='padding:0.5em; border-left:4px solid #c00; background:#fff0f0'>
    <strong>🧠 Highlighted Original Text</strong></div>""", unsafe_allow_html=True)
    st.markdown(highlighted_text, unsafe_allow_html=True)

    # ------------------------------
    # Save to History
    # ------------------------------
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_option,
        "Prediction": label,
        "Fake Probability": round(fake_prob, 4),
        "Text": user_input[:100].replace("\n", " ") + ("..." if len(user_input) > 100 else "")
    })

# ------------------------------
# Show History + Export
# ------------------------------
if st.session_state.history:
    st.markdown("### 📄 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    # CSV Export
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button("📥 Download Report as CSV", csv_buffer, file_name="fake_news_predictions.csv", mime="text/csv")
    # ------------------------------
    # PDF Export (with Charts)
    # ------------------------------
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", 'B', 16)
            self.cell(0, 10, "Fake News Detection Report", ln=True, align='C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f"Page {self.page_no()}", align='C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Summary Chart: Model Avg Prob
    fig1, ax1 = plt.subplots()
    model_avg = df.groupby("Model")["Fake Probability"].mean()
    model_avg.plot(kind="bar", ax=ax1, color="#cc0000")
    ax1.set_title("Average FAKE Probability per Model")
    ax1.set_ylabel("Probability")
    fig1.tight_layout()
    chart1_buf = BytesIO()
    fig1.savefig(chart1_buf, format="png")
    chart1_buf.seek(0)
    pdf.image(chart1_buf, w=180)

    # Pie chart: Prediction counts
    fig2, ax2 = plt.subplots()
    count_series = df["Prediction"].value_counts()
    ax2.pie(count_series, labels=count_series.index, autopct="%1.1f%%", colors=["#cc0000", "#00aa66"])
    ax2.set_title("Prediction Distribution")
    chart2_buf = BytesIO()
    fig2.savefig(chart2_buf, format="png")
    chart2_buf.seek(0)
    pdf.add_page()
    pdf.image(chart2_buf, w=150)

    # Records
    pdf.add_page()
    for _, row in df.iterrows():
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f"[{row['Timestamp']}]", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, (
            f"Model: {row['Model']}\n"
            f"Prediction: {row['Prediction']}\n"
            f"Fake Probability: {row['Fake Probability']*100:.2f}%\n"
            f"Text: {row['Text']}\n"
        ))
        pdf.ln(2)

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button("📄 Download Report as PDF", pdf_buffer, file_name="fake_news_report.pdf", mime="application/pdf")

    if st.button("🧹 Clear Prediction History"):
        st.session_state.history.clear()
        st.rerun()
