import streamlit as st
import joblib
import numpy as np
import re
import nltk
import string
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
    # Fallback minimal stopword set if download is unavailable
    stop_words = {
        'a','an','the','and','or','but','if','while','with','of','to','in','on','for','at','by','from','is','are','was','were','be','been','being','as','that','this','these','those','it','its','you','your','yours','he','she','they','them','we','us','our','ours'
    }

stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    """Basic preprocessing: lowercase, remove URLs/mentions, keep letters, remove stopwords, apply stemming."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)


# ------------------------------
# Safe artifact loading helpers
# ------------------------------

def safe_load(path: str, human_name: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Missing or unreadable {human_name}: '{path}'. Error: {e}")
        return None

# NOTE: Keep file names consistent with your repository artifacts.
# Expecting: tfidf_vectorizer.pkl, logistic_model.pkl, naive_bayes_model.pkl, svm_model.pkl

# Load artifacts
# (Using safe loaders so the app shows friendly errors instead of crashing.)
tfidf = safe_load("tfidf_vectorizer.pkl", "TF-IDF vectorizer")
lr_model = safe_load("logistic_model.pkl", "Logistic Regression model")
nb_model = safe_load("naive_bayes_model.pkl", "Naive Bayes model")
svm_model = safe_load("svm_model.pkl", "SVM model")

# If TF-IDF is missing, classic models cannot run
if tfidf is None:
    st.stop()

# Build model options based on what is actually available
model_dict = {}
if lr_model is not None:
    model_dict["Logistic Regression"] = lr_model
if nb_model is not None:
    model_dict["Naive Bayes"] = nb_model
if svm_model is not None:
    model_dict["SVM"] = svm_model
# Hybrid option (LR + NB) only if both are available
if lr_model is not None and nb_model is not None:
    model_dict["Hybrid (LR + NB)"] = None

if not model_dict:
    st.error("No models are available. Please upload the required .pkl files.")
    st.stop()


# ------------------------------
# UI
# ------------------------------
st.title("📰 Fake News Detection System")
st.markdown("Use machine learning to classify a news text as real or fake, and highlight key tokens.")

user_input = st.text_area(
    "📝 Enter the news content to analyze:", height=200,
    help="You can paste a full article body or a short headline."
)

# Default to Hybrid if available; otherwise default to first available
options = list(model_dict.keys())
default_index = options.index("Hybrid (LR + NB)") if "Hybrid (LR + NB)" in options else 0
model_option = st.selectbox("🤖 Choose a model:", options, index=default_index)

# Show weight slider for the Hybrid (LR + NB)
if model_option == "Hybrid (LR + NB)":
    w_lr = st.slider("Logistic Regression weight (w_LR)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    w_nb = 1.0 - w_lr
    st.caption(f"Hybrid weights → LR: {w_lr:.2f}, NB: {w_nb:.2f}")


# ------------------------------
# Inference
# ------------------------------
if st.button("Run Prediction", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter some text before running the prediction.")
    else:
        cleaned_input = preprocess(user_input)
        x_input = tfidf.transform([cleaned_input])

        # --- Predict ---
        if model_option == "Hybrid (LR + NB)":
            # Weighted ensemble: P(fake) = w_lr * P_lr + (1 - w_lr) * P_nb
            prob_lr = lr_model.predict_proba(x_input)[0][1] if hasattr(lr_model, "predict_proba") else None
            prob_nb = nb_model.predict_proba(x_input)[0][1] if hasattr(nb_model, "predict_proba") else None
            if prob_lr is None:
                # Fallback using decision_function -> logistic
                decision_lr = lr_model.decision_function(x_input)
                prob_lr = 1 / (1 + np.exp(-decision_lr))[0]
            if prob_nb is None:
                decision_nb = nb_model.decision_function(x_input)
                prob_nb = 1 / (1 + np.exp(-decision_nb))[0]
            fake_prob = w_lr * prob_lr + (1.0 - w_lr) * prob_nb
        else:
            model = model_dict[model_option]
            if hasattr(model, "predict_proba"):
                fake_prob = model.predict_proba(x_input)[0][1]
            else:
                decision = model.decision_function(x_input)
                fake_prob = 1 / (1 + np.exp(-decision))[0]

        # --- Display prediction result ---
        label = "🔴 FAKE" if fake_prob >= 0.5 else "🟢 REAL"
        if fake_prob >= 0.5:
            st.error(f"### Prediction: {label}")
        else:
            st.success(f"### Prediction: {label}")

        # --- Probability progress ---
        st.markdown("#### 📊 Predicted Probability of Fake News")
        st.progress(int(fake_prob * 100))
        st.metric("Model-estimated probability of FAKE", f"{fake_prob * 100:.2f}%")

        # --- Top TF-IDF tokens (for highlighting only) ---
        feature_names = tfidf.get_feature_names_out()
        input_vector = x_input.toarray()[0]
        top_indices = input_vector.argsort()[::-1][:10]
        highlight_words = [feature_names[i] for i in top_indices if input_vector[i] > 0]

        st.markdown("#### 🧠 Top TF-IDF Tokens")
        st.write(", ".join(highlight_words) if highlight_words else "(No tokens with non-zero TF-IDF weight)")

        # --- Inline highlight in the original text ---
        highlighted_text = user_input
        for word in highlight_words:
            pattern = re.compile(rf"(?i)\\b{re.escape(word)}\\b")
            highlighted_text = pattern.sub(
                f"<span style='background-color:#ffcccc; color:#c00'><b>{word}</b></span>",
                highlighted_text
            )

        st.markdown("#### 🧠 Highlighted Original Text (AI-style view)", unsafe_allow_html=True)
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # --- Note ---
        st.caption("Top tokens are extracted by TF-IDF and are for reference only. They are not guaranteed to reflect the final decision basis.")
