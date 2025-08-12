import streamlit as st
import joblib
import numpy as np
import re
import nltk
import string
import pandas as pd
from datetime import datetime
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
        'a','an','the','and','or','but','if','while','with','of','to','in','on','for','at','by','from','is','are','was','were','be','been','being','as','that','this','these','those','it','its','you','your','yours','he','she','they','them','we','us','our','ours'
    }

stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)


def safe_load(path: str, human_name: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Missing or unreadable {human_name}: '{path}'. Error: {e}")
        return None

# Load artifacts
tfidf = safe_load("tfidf_vectorizer.pkl", "TF-IDF vectorizer")
lr_model = safe_load("logistic_model.pkl", "Logistic Regression model")
nb_model = safe_load("naive_bayes_model.pkl", "Naive Bayes model")
svm_model = safe_load("svm_model.pkl", "SVM model")

if tfidf is None:
    st.stop()

model_dict = {}
if lr_model is not None:
    model_dict["Logistic Regression"] = lr_model
if nb_model is not None:
    model_dict["Naive Bayes"] = nb_model
if svm_model is not None:
    model_dict["SVM"] = svm_model
if lr_model and nb_model and svm_model:
    model_dict["Hybrid (LR + NB + SVM)"] = None

if not model_dict:
    st.error("No models are available. Please upload the required .pkl files.")
    st.stop()

# ------------------------------
# UI
# ------------------------------
st.title("📰 Fake News Detection System")
st.markdown("Use machine learning to classify a news text as real or fake, and highlight key tokens.")

user_input = st.text_area("📝 Enter the news content to analyze:", height=200)
options = list(model_dict.keys())
default_index = options.index("Hybrid (LR + NB + SVM)") if "Hybrid (LR + NB + SVM)" in options else 0
model_option = st.selectbox("🤖 Choose a model:", options, index=default_index)

if model_option == "Hybrid (LR + NB + SVM)":
    w_lr = st.slider("Logistic Regression weight (w_LR)", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
    w_nb = st.slider("Naive Bayes weight (w_NB)", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
    remaining = 1.0 - w_lr - w_nb
    w_svm = max(0.0, min(1.0, remaining))
    st.caption(f"Hybrid weights → LR: {w_lr:.2f}, NB: {w_nb:.2f}, SVM: {w_svm:.2f}")

# ------------------------------
# Inference and History
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Run Prediction", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter some text before running the prediction.")
    else:
        cleaned_input = preprocess(user_input)
        x_input = tfidf.transform([cleaned_input])

        if model_option == "Hybrid (LR + NB + SVM)":
            def get_prob(m):
                if hasattr(m, "predict_proba"):
                    return m.predict_proba(x_input)[0][1]
                else:
                    decision = m.decision_function(x_input)
                    return 1 / (1 + np.exp(-decision))[0]
            prob_lr = get_prob(lr_model)
            prob_nb = get_prob(nb_model)
            prob_svm = get_prob(svm_model)
            total_weight = w_lr + w_nb + w_svm
            fake_prob = (w_lr * prob_lr + w_nb * prob_nb + w_svm * prob_svm) / total_weight if total_weight > 0 else 0.5
        else:
            model = model_dict[model_option]
            if hasattr(model, "predict_proba"):
                fake_prob = model.predict_proba(x_input)[0][1]
            else:
                decision = model.decision_function(x_input)
                fake_prob = 1 / (1 + np.exp(-decision))[0]

        label = "🔴 FAKE" if fake_prob >= 0.5 else "🟢 REAL"
        if fake_prob >= 0.5:
            st.error(f"### Prediction: {label}")
        else:
            st.success(f"### Prediction: {label}")

        st.markdown("#### 📊 Predicted Probability of Fake News")
        st.progress(int(fake_prob * 100))
        st.metric("Model-estimated probability of FAKE", f"{fake_prob * 100:.2f}%")

        feature_names = tfidf.get_feature_names_out()
        input_vector = x_input.toarray()[0]
        top_indices = input_vector.argsort()[::-1][:10]
        highlight_words = [feature_names[i] for i in top_indices if input_vector[i] > 0]

        st.markdown("#### 🧠 Top TF-IDF Tokens")
        st.write(", ".join(highlight_words) if highlight_words else "(No tokens with non-zero TF-IDF weight)")

        highlighted_text = user_input
        for word in highlight_words:
            pattern = re.compile(rf"(?i)\\b{re.escape(word)}\\b")
            highlighted_text = pattern.sub(
                rf"<span style=\"background-color:#ffe6e6; color:#c00\"><b>{word}</b></span>",
                highlighted_text
            )

        st.markdown("#### 🧠 Highlighted Original Text (AI-style view)", unsafe_allow_html=True)
        st.markdown(highlighted_text, unsafe_allow_html=True)

        st.session_state.history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_option,
            "Prediction": label,
            "Fake Probability": round(fake_prob, 4),
            "Text": user_input[:100].replace("\n", " ") + ("..." if len(user_input) > 100 else "")
        })

        st.caption("Top tokens are extracted by TF-IDF and are for reference only. They are not guaranteed to reflect the final decision basis.")

# ------------------------------
# History and Report Export
# ------------------------------
if st.session_state.history:
    st.markdown("### 📄 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("📥 Download Report as CSV", buffer, file_name="fake_news_predictions.csv", mime="text/csv")

    if st.button("🧹 Clear Prediction History"):
        st.session_state.history.clear()
        st.success("Prediction history has been cleared.")
