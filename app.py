import streamlit as st
import joblib
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

# 加载模型
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr_model = joblib.load("logistic_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
svm_model = joblib.load("svm_model.pkl")

model_dict = {
    "Logistic Regression": lr_model,
    "Naive Bayes": nb_model,
    "SVM": svm_model,
    "混合模型（LR + NB）": None
}

# 🧠 页面设置
st.set_page_config(page_title="假新闻识别系统", layout="centered")
st.title("📰 假新闻智能识别系统")
st.markdown("通过机器学习判断一段新闻是真实还是虚假，并标出模型重点关注的词汇。")

# ✍️ 文本输入框
user_input = st.text_area("📝 请输入你要检测的新闻内容：", height=200, help="可输入完整正文或简要标题")

# 模型选择（默认选混合）
model_option = st.selectbox("🤖 选择分类模型：", list(model_dict.keys()), index=3)

# 🚀 按钮触发
if st.button("开始检测"):

    if not user_input.strip():
        st.warning("请输入有效文本后再开始检测。")
    else:
        cleaned_input = preprocess(user_input)
        x_input = tfidf.transform([cleaned_input])

        # 模型推理
        if model_option == "混合模型（LR + NB）":
            prob_lr = lr_model.predict_proba(x_input)[0][1]
            prob_nb = nb_model.predict_proba(x_input)[0][1]
            fake_prob = (prob_lr + prob_nb) / 2
        else:
            model = model_dict[model_option]
            if hasattr(model, "predict_proba"):
                fake_prob = model.predict_proba(x_input)[0][1]
            else:
                decision = model.decision_function(x_input)
                fake_prob = 1 / (1 + np.exp(-decision))[0]

        # 🧾 判定结果
        label = "🔴 假新闻" if fake_prob >= 0.5 else "🟢 真实新闻"
        if fake_prob >= 0.5:
            st.error(f"### 🧾 判定结果：{label}")
        else:
            st.success(f"### 🧾 判定结果：{label}")

        # 🎯 显示预测概率进度条
        st.markdown("#### 📊 假新闻概率预测")
        st.progress(int(fake_prob * 100))
        st.metric("模型判断该文本为假新闻的概率", f"{fake_prob * 100:.2f}%")

        # 🔍 关键词提取 + 高亮
        feature_names = tfidf.get_feature_names_out()
        input_vector = x_input.toarray()[0]
        top_indices = input_vector.argsort()[::-1][:10]
        highlight_words = [feature_names[i] for i in top_indices if input_vector[i] > 0]

        st.markdown("#### 🧠 模型关注关键词：")
        st.write(", ".join(highlight_words))

        # ✨ 原文关键词染色显示
        highlighted_text = user_input
        for word in highlight_words:
            pattern = re.compile(rf"(?i)\b{re.escape(word)}\b")
            highlighted_text = pattern.sub(
                f"<span style='background-color:#ffcccc; color:#c00'><b>{word}</b></span>",
                highlighted_text
            )

        st.markdown("#### 🧠 关键词高亮原文展示（仿 AI 检测工具）", unsafe_allow_html=True)
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # 💡 展示提示
        st.caption("关键词仅基于 TF-IDF 提取，并不代表最终判定依据，仅供参考。")
