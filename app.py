# app.py
# -----------------------------------------------------------------------------
# Fake News Detection Web App (Streamlit)
# - Classic Models: TF-IDF + (LogReg / Naive Bayes / Linear SVM) + Weighted Ensemble
# - BERT Model (lazy-loaded from Hugging Face Hub)
# - Explanations:
#     * TF-IDF: token importance highlighted via model coefficients / log-probs
#     * BERT: lightweight attention-based token visualization (illustrative)
#
# IMPORTANT (HF Spaces / Python 3.13 issue):
#   If your Space uses Python 3.13, `tokenizers` may fail to build.
#   Add `runtime.txt` with:   python-3.12
#   This allows transformers==4.42.3 to install cleanly.
# -----------------------------------------------------------------------------

import os
import re
import html
import glob
import json
import pickle
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import streamlit as st

# Silence some sklearn pickle warnings (version drift)
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator.*"
)

# -----------------------------------------------------------------------------
# App Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Hugging Face model repo id (change to your own repo if needed)
HF_MODEL_ID = "qq1244715496/fake-news-detection"  # BERT fine-tuned classifier id on HF Hub

# Where to look for classic models (vectorizer + pickled sklearn models)
DEFAULT_MODEL_DIRS = ["models", "artifacts", ".", "checkpoints"]

# Class names fallback (used if model.classes_ not found)
DEFAULT_LABELS = ["REAL", "FAKE"]  # index 0=REAL, 1=FAKE (common convention). Adjust if needed.


# -----------------------------------------------------------------------------
# Small Utilities
# -----------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _find_file(patterns: List[str], search_dirs: List[str]) -> Optional[str]:
    """Find the first file matching any pattern in any of the search dirs."""
    for d in search_dirs:
        for p in patterns:
            for f in glob.glob(os.path.join(d, p)):
                return f
    return None


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_model_labels(model) -> List[str]:
    """Get class labels from a scikit model or return default labels."""
    if hasattr(model, "classes_"):
        # Convert numpy types to str
        return [str(c) for c in list(model.classes_)]
    return DEFAULT_LABELS


def _html_highlight_tokens(
    text_tokens: List[str],
    scores: List[float],
    pos_color="#ffefef",
    neg_color="#eff7ff",
) -> str:
    """Render tokens with background intensity by absolute score."""
    # Normalize to [0,1]
    arr = np.array(scores, dtype=float)
    if len(arr) == 0:
        arr = np.zeros(len(text_tokens))
    max_abs = np.max(np.abs(arr)) if np.any(arr) else 1.0
    safe_tokens = [html.escape(t) for t in text_tokens]

    spans = []
    for tok, s in zip(safe_tokens, arr):
        alpha = 0.2 + 0.8 * (abs(s) / max_abs) if max_abs > 0 else 0.0
        bg = pos_color if s >= 0 else neg_color
        style = f"background: {bg}; opacity: 1; border-radius: 4px; padding: 0.05rem 0.15rem; margin: 0 1px;"
        # Use RGBA via overlay? Simpler: adjust via linear-gradient with alpha text-shadow. Streamlit HTML sandbox is limited.
        spans.append(f'<span style="{style}">{tok}</span>')
    return "<div style='line-height:1.9; font-size:1.05rem;'>" + " ".join(spans) + "</div>"


def _simple_word_tokenize(text: str) -> List[str]:
    # Very lightweight tokenization to align with TF-IDF vocabulary (space/punct split).
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


# -----------------------------------------------------------------------------
# Load TF-IDF Vectorizer and Classic Models
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_tfidf_and_models() -> Dict[str, object]:
    """
    Load TF-IDF vectorizer and classic sklearn models from disk.
    We try common filenames. Adjust patterns to your actual files if needed.
    """
    # Try to find vectorizer
    vec_path = _find_file(
        ["*tfidf*vectorizer*.pkl", "*vectorizer*.pkl", "*tfidf*.pkl"], DEFAULT_MODEL_DIRS
    )
    if not vec_path:
        raise FileNotFoundError(
            "TF-IDF vectorizer .pkl not found. Put it under ./models or set correct filename."
        )
    vectorizer = _load_pickle(vec_path)

    # Find models
    patterns = {
        "LR": ["*logistic*regression*.pkl", "*lr*.pkl"],
        "NB": ["*naive*bayes*.pkl", "*nb*.pkl", "*multinomial*.pkl"],
        "SVM": ["*linear*svc*.pkl", "*svm*.pkl", "*linsvc*.pkl"],
    }
    models = {}
    for key, pats in patterns.items():
        path = _find_file(pats, DEFAULT_MODEL_DIRS)
        if path:
            models[key] = _load_pickle(path)

    if not models:
        raise FileNotFoundError(
            "No classic models found. Expected files like model_lr.pkl / model_nb.pkl / model_svm.pkl."
        )

    return {"vectorizer": vectorizer, **models}


def predict_tfidf(text: str, model, vectorizer) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
    """
    Run prediction for a single text using a classic sklearn model.

    Returns:
        pred_label: predicted class label (str)
        prob_dict: class -> probability (best-effort for SVM via softmax on decision)
        token_scores: list of (token, score) contributions for explanation
    """
    X = vectorizer.transform([text])

    labels = _get_model_labels(model)

    # Probability / decision
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # SVM LinearSVC: use decision_function then softmax
        decision = model.decision_function(X)
        if np.ndim(decision) == 1:
            # binary
            proba = _softmax(np.array([ -decision[0], decision[0] ]))
            labels = labels if len(labels) == 2 else ["NEG", "POS"]
        else:
            proba = _softmax(decision[0])

    top_idx = int(np.argmax(proba))
    pred_label = labels[top_idx]
    prob_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}

    # Token-level explanation via coefficients/log-probs
    # Map text tokens to vectorizer features (lowercased tokens are typical for TF-IDF)
    tokens = _simple_word_tokenize(text)
    token_scores = []

    # Choose target class index for explanation (class with max prob)
    target_idx = top_idx

    # Get weight vector (coef) or log prob differences
    if hasattr(model, "coef_"):
        # For binary: coef_[0] corresponds to positive class (usually class 1)
        # For multi: take coef_[target_idx]
        if model.coef_.ndim == 2 and model.coef_.shape[0] > 1:
            w = model.coef_[target_idx]
        else:
            w = model.coef_[0]
        # Intercept not used per-token
        feature_names = vectorizer.get_feature_names_out()
        vocab = {t: i for i, t in enumerate(feature_names)}
        for tok in tokens:
            key = tok.lower()
            if key in vocab:
                token_scores.append((tok, float(w[vocab[key]])))
            else:
                token_scores.append((tok, 0.0))
    elif hasattr(model, "feature_log_prob_"):  # MultinomialNB
        flp = model.feature_log_prob_
        feature_names = vectorizer.get_feature_names_out()
        vocab = {t: i for i, t in enumerate(feature_names)}
        # Use difference from mean or from competing class (approx)
        for tok in tokens:
            key = tok.lower()
            if key in vocab:
                col = vocab[key]
                score = float(flp[target_idx, col] - np.mean(flp[:, col]))
                token_scores.append((tok, score))
            else:
                token_scores.append((tok, 0.0))
    else:
        # Fallback: no per-feature weights available
        token_scores = [(tok, 0.0) for tok in tokens]

    return pred_label, prob_dict, token_scores


def ensemble_predict(
    text: str, models: Dict[str, object], vectorizer, weights: Dict[str, float]
) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
    """Simple weighted probability (or decision) fusion across LR/NB/SVM."""
    votes = {}
    token_score_accum: Dict[str, float] = {}
    token_counts: Dict[str, int] = {}

    # Normalize weights
    total_w = sum(weights.get(k, 0.0) for k in ["LR", "NB", "SVM"])
    if total_w <= 0:
        weights = {"LR": 1.0, "NB": 1.0, "SVM": 1.0}
        total_w = 3.0

    for name in ["LR", "NB", "SVM"]:
        if name in models:
            label, probs, tok_scores = predict_tfidf(text, models[name], vectorizer)
            w = weights.get(name, 1.0) / total_w
            for c, p in probs.items():
                votes[c] = votes.get(c, 0.0) + w * p
            for tok, s in tok_scores:
                token_score_accum[tok] = token_score_accum.get(tok, 0.0) + w * s
                token_counts[tok] = token_counts.get(tok, 0, 0) + 1

    if not votes:
        return "N/A", {}, [(t, 0.0) for t in _simple_word_tokenize(text)]

    # Normalize vote to sum=1
    total = sum(votes.values())
    if total > 0:
        for k in list(votes.keys()):
            votes[k] /= total

    pred_label = max(votes.items(), key=lambda kv: kv[1])[0]

    # Average token scores across models that produced a score
    tokens = _simple_word_tokenize(text)
    token_scores = [(t, token_score_accum.get(t, 0.0) / max(1, token_counts.get(t, 1))) for t in tokens]

    return pred_label, votes, token_scores


# -----------------------------------------------------------------------------
# Lazy BERT Loader
# -----------------------------------------------------------------------------
def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=True)
def load_bert(model_id: str, token: Optional[str] = None):
    """
    Lazy-load BERT from Hugging Face Hub.
    Returns: (tokenizer, model, device)
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        raise RuntimeError(
            f"Transformers not available: {e}. "
            "If running on HF Spaces, add 'runtime.txt' with 'python-3.12' to avoid tokenizers build errors on Python 3.13."
        )

    # Use auth token if the repo is private or gated
    use_auth = token if token else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=use_auth)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        use_auth_token=use_auth,
        output_attentions=True,            # enable attentions for a lightweight viz
        torch_dtype="auto"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def bert_predict_with_attention(text: str, tokenizer, model, device) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
    """
    Run BERT and extract a simple token-importance proxy using last-layer attention.
    NOTE: Attention is not a faithful explanation, but gives an intuitive heatmap.
    """
    import torch

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_attention_mask=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    logits = out.logits[0].detach().cpu().numpy()
    probs = _softmax(logits)
    # Labels
    if hasattr(model.config, "id2label") and model.config.id2label:
        labels = [model.config.id2label[i] for i in range(len(probs))]
    else:
        labels = DEFAULT_LABELS[: len(probs)]

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # Attention-based token importance from the last layer: mean across heads, focus on CLS attention distribution
    # Shape: (layers, batch, heads, seq_len, seq_len)
    attns = out.attentions[-1]  # last layer
    att_mean = attns.mean(dim=2)  # average over heads -> (batch, seq_len, seq_len)
    # take attention from [CLS] to others, index 0 token
    cls_att = att_mean[0, 0].detach().cpu().numpy()  # (seq_len,)

    # Map back to tokens
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    # Create scores for visible tokens only (ignore special tokens)
    vis_tokens, vis_scores = [], []
    for tok, sc in zip(tokens, cls_att):
        if tok in ("[CLS]", "[SEP]", tokenizer.pad_token):
            continue
        # Clean up "##" subwords -> merge by summing or taking max. We'll sum within a simple combiner.
        vis_tokens.append(tok)
        vis_scores.append(float(sc))

    # Merge WordPiece tokens into words
    merged_tokens, merged_scores = [], []
    buf_tok, buf_score = "", 0.0
    for tok, sc in zip(vis_tokens, vis_scores):
        if tok.startswith("##"):
            buf_tok += tok[2:]
            buf_score += sc
        else:
            if buf_tok:
                merged_tokens.append(buf_tok)
                merged_scores.append(buf_score)
            buf_tok, buf_score = tok, sc
    if buf_tok:
        merged_tokens.append(buf_tok)
        merged_scores.append(buf_score)

    # Center scores around 0 for a symmetric colormap (subtract mean)
    if merged_scores:
        ms = np.array(merged_scores)
        ms = ms - ms.mean()
        merged_scores = ms.tolist()

    return pred_label, prob_dict, list(zip(merged_tokens, merged_scores))


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("📰 Fake News Detection (TF‑IDF / BERT)")
st.caption("Type or paste a news headline or paragraph. Choose a model to predict and see which words influenced the result.")

with st.sidebar:
    st.header("Model Selection")
    model_type = st.radio(
        "Choose a model",
        ["Logistic Regression (TF‑IDF)", "Naive Bayes (TF‑IDF)", "Linear SVM (TF‑IDF)", "Weighted Ensemble (TF‑IDF)", "BERT (Hugging Face)"],
        index=0
    )
    show_probs = st.checkbox("Show class probabilities", value=True)

    if model_type == "Weighted Ensemble (TF‑IDF)":
        st.subheader("Ensemble Weights")
        w_lr = st.slider("Weight: LR", 0.0, 3.0, 1.0, 0.1)
        w_nb = st.slider("Weight: NB", 0.0, 3.0, 1.0, 0.1)
        w_svm = st.slider("Weight: SVM", 0.0, 3.0, 1.0, 0.1)
        ens_weights = {"LR": w_lr, "NB": w_nb, "SVM": w_svm}
    else:
        ens_weights = None

    st.markdown("---")
    st.subheader("BERT Options")
    st.caption("Loaded only if you pick BERT.")
    hf_id = st.text_input("Hugging Face Model ID", HF_MODEL_ID)
    use_secret_token = st.checkbox("Use st.secrets['HF_TOKEN'] if available", value=True)
    manual_token = st.text_input("Or paste an HF access token (optional)", type="password")

st.markdown("### ✍️ Input Text")
default_text = "Breaking: Government announces new policy to cut taxes by 50% starting tomorrow."
text = st.text_area("Enter a headline or news paragraph:", value=default_text, height=150, placeholder="Paste your text here...")

col_left, col_right = st.columns([1,1])
with col_left:
    run_btn = st.button("🔎 Predict", type="primary", use_container_width=True)
with col_right:
    clear_btn = st.button("🧹 Clear", use_container_width=True)
    if clear_btn:
        st.experimental_rerun()

st.markdown("---")

# -----------------------------------------------------------------------------
# Inference Flow
# -----------------------------------------------------------------------------
if run_btn and text.strip():
    if "TF‑IDF" in model_type:
        # Load classic models
        with st.spinner("Loading TF‑IDF vectorizer and classic models..."):
            obj = load_tfidf_and_models()
        vectorizer = obj["vectorizer"]
        models = {k: v for k, v in obj.items() if k in ["LR", "NB", "SVM"]}

        if model_type.startswith("Logistic"):
            if "LR" not in models:
                st.error("Logistic Regression model file not found.")
            else:
                pred, probs, tok_scores = predict_tfidf(text, models["LR"], vectorizer)
        elif model_type.startswith("Naive"):
            if "NB" not in models:
                st.error("Naive Bayes model file not found.")
            else:
                pred, probs, tok_scores = predict_tfidf(text, models["NB"], vectorizer)
        elif model_type.startswith("Linear SVM"):
            if "SVM" not in models:
                st.error("Linear SVM model file not found.")
            else:
                pred, probs, tok_scores = predict_tfidf(text, models["SVM"], vectorizer)
        else:  # Ensemble
            if not models:
                st.error("No classic models available for ensemble.")
            else:
                pred, probs, tok_scores = ensemble_predict(text, models, vectorizer, ens_weights or {"LR":1,"NB":1,"SVM":1})

        if 'pred' in locals():
            st.subheader("📌 Prediction")
            st.success(f"**Predicted Label:** {pred}")
            if show_probs and probs:
                st.markdown("**Class Probabilities:**")
                st.json({k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])})

            # Render token highlights
            st.subheader("🔍 Word Influence (TF‑IDF)")
            toks, scores = zip(*tok_scores) if tok_scores else ([], [])
            html_block = _html_highlight_tokens(list(toks), list(scores))
            st.markdown(html_block, unsafe_allow_html=True)

    else:
        # BERT path (lazy import + load)
        if not _transformers_available():
            st.error(
                "Transformers is not available. On Hugging Face Spaces, add a `runtime.txt` with `python-3.12` "
                "to avoid tokenizers build failure on Python 3.13. You can still use the TF‑IDF models."
            )
        else:
            # Prepare token
            hf_token = None
            if use_secret_token and isinstance(st.secrets, dict):
                hf_token = st.secrets.get("HF_TOKEN")
            if manual_token:
                hf_token = manual_token

            with st.spinner(f"Loading BERT from {hf_id} ..."):
                try:
                    tok, mdl, device = load_bert(hf_id, token=hf_token)
                except Exception as e:
                    st.exception(e)
                    st.stop()

            with st.spinner("Running BERT inference..."):
                pred, probs, tok_scores = bert_predict_with_attention(text, tok, mdl, device)

            st.subheader("📌 Prediction (BERT)")
            st.success(f"**Predicted Label:** {pred}")
            if show_probs and probs:
                st.markdown("**Class Probabilities:**")
                st.json({k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])})

            st.subheader("🧠 Token Attention (Illustrative)")
            toks, scores = zip(*tok_scores) if tok_scores else ([], [])
            html_block = _html_highlight_tokens(list(toks), list(scores), pos_color="#fff5e6", neg_color="#e6f7ff")
            st.markdown(html_block, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer / Help
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Deployment Notes / Troubleshooting"):
    st.markdown(
        """
- **Python 3.13 & tokenizers 错误**：在项目根目录添加 `runtime.txt`，内容为 `python-3.12`，然后重新部署。
- **模型文件路径**：默认会在 `./models`, `./artifacts`, `./`, `./checkpoints` 中查找：
  - TF‑IDF 向量器：包含 `tfidf` / `vectorizer` 字样的 `.pkl`
  - 经典模型：
    - Logistic Regression: 名称含 `logistic` 或 `lr`
    - Naive Bayes: 名称含 `naive` / `nb` / `multinomial`
    - Linear SVM: 名称含 `linear` / `svm` / `linsvc`
  如有不同，请修改 `load_tfidf_and_models()` 中的通配符。
- **BERT 私有/受限模型**：把访问令牌配置到 `st.secrets["HF_TOKEN"]` 或在侧边栏粘贴。
- **解释说明**：
  - TF‑IDF 使用模型权重对词做高亮，具有一定可解释性。
  - BERT 的注意力可视化仅作直观参考，非严格可解释方法。
        """
    )
