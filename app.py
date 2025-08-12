# ============================================
# Fake News Detector (Streamlit + BERT + TF-IDF)
# ============================================

import re
import numpy as np
import streamlit as st
import joblib

# ----------------------------
# Page / layout
# ----------------------------
st.set_page_config(page_title="Fake News Detector (with BERT)", layout="centered")
st.title("📰 Fake News Detector")
st.caption("Classic ML + BERT with visual explanations. Red = pushes FAKE, Blue = pushes REAL.")

# ----------------------------
# Config: Hugging Face model repo
# ----------------------------
HF_MODEL_ID = "qq1244715496/fake-news-detection"  # Your public Hugging Face repo
HF_TOKEN = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None

# ----------------------------
# Load TF-IDF + classic models
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_classic():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    lr_model = joblib.load("logistic_model.pkl")
    nb_model = joblib.load("naive_bayes_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    return tfidf, lr_model, nb_model, svm_model

tfidf, lr_model, nb_model, svm_model = load_classic()

# ----------------------------
# Text preprocessing for TF-IDF ONLY
# ----------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords")
STOP = set(stopwords.words("english"))
STEM = PorterStemmer()

def preprocess(text: str) -> str:
    """Light clean + stemming for TF-IDF pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    cleaned = [STEM.stem(w) for w in tokens if w not in STOP]
    return " ".join(cleaned)

# ----------------------------
# BERT loader and inference
# ----------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource(show_spinner=True)
def load_bert(hf_id: str, token: str | None = None):
    """Load BERT model/tokenizer once."""
    tok = AutoTokenizer.from_pretrained(hf_id, token=token)
    mdl = AutoModelForSequenceClassification.from_pretrained(hf_id, token=token)
    device = torch.device("cpu")
    mdl.to(device).eval()
    return tok, mdl, device

def bert_prob_fake(raw_text: str, tok, mdl, device, max_len: int = 256) -> float:
    """Return P(FAKE) using BERT."""
    enc = tok(
        raw_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return float(probs[1])  # Class index 1 = FAKE

# ----------------------------
# Integrated Gradients for BERT
# ----------------------------
from captum.attr import IntegratedGradients

def bert_ig_attributions(text, tok, mdl, device,
                         max_len=256, target_label=1, n_steps=32):
    """Compute token-level attributions via IG."""
    enc = tok(text, return_tensors="pt", truncation=True,
              padding="max_length", max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    emb_layer = mdl.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids).detach().clone().requires_grad_(True)

    def forward_func(input_embeds):
        outputs = mdl(inputs_embeds=input_embeds, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, target_label]

    mdl.eval(); mdl.zero_grad()
    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        inputs=inputs_embeds, target=0, n_steps=n_steps, return_convergence_delta=True
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = tok.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    valid_len = int(attention_mask[0].sum().item())
    tokens = tokens[:valid_len]
    token_scores = token_scores[:valid_len]

    # Merge wordpieces
    merged_words, merged_scores = [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("[CLS]", "[SEP]"):
            i += 1; continue
        if t.startswith("##"):
            i += 1; continue
        word = t.replace("##", "")
        score = token_scores[i]
        j = i + 1
        while j < len(tokens) and tokens[j].startswith("##"):
            word += tokens[j][2:]
            score += token_scores[j]
            j += 1
        merged_words.append(word)
        merged_scores.append(score)
        i = j

    merged_scores = np.array(merged_scores)
    if merged_scores.std() > 1e-6:
        merged_scores = (merged_scores - merged_scores.mean()) / (merged_scores.std() + 1e-6)
    else:
        merged_scores = merged_scores * 0.0
    return merged_words, merged_scores

def build_colored_html(words, scores, top_k=30, min_abs_z=0.0, show_dir="both"):
    """Convert words+scores to HTML spans."""
    order = np.argsort(np.abs(scores))[::-1]
    keep = set(order[: max(1, top_k)])
    html = []
    for idx, (w, s) in enumerate(zip(words, scores)):
        if show_dir == "fake_only" and s <= 0:
            html.append(w); continue
        if show_dir == "real_only" and s >= 0:
            html.append(w); continue
        if idx in keep and abs(s) >= min_abs_z:
            alpha = float(min(1.0, 0.15 + 0.85 * min(1.0, abs(s) / 3.0)))
            if s > 0:
                html.append(f"<span style='background-color: rgba(255,0,0,{alpha}); color:black'>{w}</span>")
            elif s < 0:
                html.append(f"<span style='background-color: rgba(0,120,255,{alpha}); color:white'>{w}</span>")
            else:
                html.append(w)
        else:
            html.append(w)
    return " ".join(html)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Explanation Controls")
    explain_switch = st.checkbox("Show BERT token-level explanation (IG)", value=False)
    top_k = st.slider("Top-K highlighted words", 10, 60, 30, 1)
    min_abs_z = st.slider("|z|-threshold", 0.0, 3.0, 0.6, 0.1)
    show_dir = st.radio("Direction filter", ["both", "fake_only", "real_only"], index=0)

# ----------------------------
# Input + model choice
# ----------------------------
user_text = st.text_area("Enter news text:", height=220)
model_options = ["Fusion (LR + NB)", "Logistic Regression", "Naive Bayes", "SVM", "BERT", "Fusion (BERT + LR)"]
model_choice = st.selectbox("Choose model / fusion:", model_options, index=0)

# ----------------------------
# Run prediction
# ----------------------------
if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter valid text before analyzing.")
        st.stop()

    if model_choice in ["Logistic Regression", "Naive Bayes", "SVM", "Fusion (LR + NB)"]:
        cleaned = preprocess(user_text)
        X = tfidf.transform([cleaned])
        if model_choice == "Fusion (LR + NB)":
            p_lr = lr_model.predict_proba(X)[0][1]
            p_nb = nb_model.predict_proba(X)[0][1]
            fake_prob = (p_lr + p_nb) / 2
        else:
            mdl = {"Logistic Regression": lr_model, "Naive Bayes": nb_model, "SVM": svm_model}[model_choice]
            if hasattr(mdl, "predict_proba"):
                fake_prob = mdl.predict_proba(X)[0][1]
            else:
                score = mdl.decision_function(X)
                fake_prob = float(1 / (1 + np.exp(-score))[0])
        tok = mdl_obj = device = None
    else:
        tok, mdl, device = load_bert(HF_MODEL_ID, token=HF_TOKEN)
        p_bert = bert_prob_fake(user_text, tok, mdl, device)
        if model_choice == "BERT":
            fake_prob = p_bert
        else:
            cleaned = preprocess(user_text)
            X = tfidf.transform([cleaned])
            p_lr = lr_model.predict_proba(X)[0][1]
            fake_prob = (p_bert + p_lr) / 2

    # Show result
    label = "🔴 FAKE" if fake_prob >= 0.5 else "🟢 REAL"
    st.progress(int(fake_prob * 100))
    st.metric("P(FAKE)", f"{fake_prob * 100:.2f}%")

    # TF-IDF keywords
    cleaned_for_highlight = preprocess(user_text)
    vec = tfidf.transform([cleaned_for_highlight]).toarray()[0]
    feat_names = tfidf.get_feature_names_out()
    top_idx = vec.argsort()[::-1][:10]
    tfidf_words = [feat_names[i] for i in top_idx if vec[i] > 0]
    st.write("Top TF-IDF keywords:", ", ".join(tfidf_words) if tfidf_words else "(none)")

    # Optional: BERT IG
    if (model_choice in ["BERT", "Fusion (BERT + LR)"]) and explain_switch:
        words, scores = bert_ig_attributions(user_text, tok, mdl, device, max_len=256)
        html = build_colored_html(words, scores, top_k=top_k, min_abs_z=min_abs_z, show_dir=show_dir)
        st.markdown(html, unsafe_allow_html=True)
