import warnings
# Suppress background warnings (like scikit-learn version differences) for a clean presentation
warnings.filterwarnings("ignore")

import os
import tempfile
import requests
import streamlit as st
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ==========================================
# CONSTANTS & PATHING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIERS_FILENAME = "pidginpadi_classifiers.pkl"
CLASSIFIERS_URL = "https://github.com/chimdiuto203/PidginPadi/releases/download/v1.0/pidginpadi_classifiers.pkl"

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PidginPadi | AI Moderator", page_icon="🇳🇬", layout="wide")

# ==========================================
# 2. HELPER & MODEL CACHING
# ==========================================
def download_file(url: str, dest_path: str) -> None:
    """Download a file via streaming and write atomically via a temporary file."""
    tmp_path = f"{dest_path}.tmp"
    try:
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


@st.cache_resource(show_spinner=False)
def load_models():
    # A. Load NaijaBERT Classifiers (Local first, else download from GitHub Releases)
    local_classifiers_path = os.path.join(BASE_DIR, CLASSIFIERS_FILENAME)
    temp_classifiers_path = os.path.join(tempfile.gettempdir(), CLASSIFIERS_FILENAME)

    if os.path.exists(local_classifiers_path):
        classifiers_path = local_classifiers_path
    elif os.path.exists(temp_classifiers_path):
        classifiers_path = temp_classifiers_path
    else:
        target_dir = BASE_DIR if os.access(BASE_DIR, os.W_OK) else tempfile.gettempdir()
        classifiers_path = os.path.join(target_dir, CLASSIFIERS_FILENAME)
        try:
            download_file(CLASSIFIERS_URL, classifiers_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not download '{CLASSIFIERS_FILENAME}' from GitHub Releases ({CLASSIFIERS_URL}): {e}"
            ) from e

    classifiers = joblib.load(classifiers_path)

    # B. Load NaijaBERT Engine (Local Folder or Hugging Face Fallback)
    offline_model_path = os.path.join(BASE_DIR, "naijabert_offline")

    if os.path.exists(offline_model_path):
        tokenizer = AutoTokenizer.from_pretrained(offline_model_path, local_files_only=True)
        encoder = AutoModel.from_pretrained(offline_model_path, local_files_only=True)
    else:
        hf_repo = "HausaNLP/naija-twitter-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        encoder = AutoModel.from_pretrained(hf_repo)

    # C. Load Qwen GGUF (Check local first, download via HuggingFace Hub if missing)
    local_gguf = os.path.join(BASE_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

    if os.path.exists(local_gguf):
        model_path = local_gguf
    else:
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )

    # Initialize llama-cpp model
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        verbose=False
    )

    return classifiers, tokenizer, encoder, llm


# Display loading state while models initialize
try:
    with st.spinner("Initializing AI Models (downloading models on first run, this may take a minute)..."):
        classifiers, tokenizer, encoder, llm = load_models()
except Exception as e:
    st.error(f"Failed to initialize AI models: {e}")
    st.stop()

# ==========================================
# 3. ANALYSIS PIPELINE
# ==========================================
def analyze_with_naijabert(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = encoder(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    results = {
        "Sentiment": classifiers["sentiment"].predict(cls_embedding)[0],
        "Emotion": classifiers["emotion"].predict(cls_embedding)[0],
        "Moral": classifiers["moral"].predict(cls_embedding)[0],
        "Toxicity": classifiers["toxicity"].predict(cls_embedding)[0]
    }
    return results


def generate_ai_explanation(tweet, analytical_data):
    prompt = f"""<|im_start|>system
You are PidginPadi, an expert AI in Nigerian Pidgin English, psychology, and content moderation. 
Analyze the user's tweet based on the provided raw data. Keep your explanation professional, concise (under 4 sentences), and explain how the Pidgin phrasing leads to this conclusion.<|im_end|>
<|im_start|>user
Tweet: "{tweet}"
Raw Data: Sentiment={analytical_data['Sentiment']}, Emotion={analytical_data['Emotion']}, Moral Violation={analytical_data['Moral']}, Toxicity Scale={analytical_data['Toxicity']}/4.
Provide your expert explanation.<|im_end|>
<|im_start|>assistant
"""
    response = llm(prompt, max_tokens=250, stop=["<|im_end|>"], temperature=0.3)
    return response['choices'][0]['text'].strip()

# ==========================================
# 4. WEB INTERFACE
# ==========================================
st.title("🇳🇬 PidginPadi: AI Moderator")
st.markdown("**Architecture:** Dual-Model Pipeline (NaijaBERT + Qwen-0.5B-GGUF Edge Inference)")
st.markdown("By **Chinatu Anyanwu**")
st.divider()

user_input = st.text_area(
    "Paste a Nigerian Pidgin tweet below to analyze:",
    height=100,
    placeholder="e.g., why dont you wanna stream my song if na because of my character no vex oya stream am..."
)

if st.button("Run AI Analysis", type="primary"):
    if not user_input.strip():
        st.warning("Abeg, type something first!")
    else:
        with st.spinner("Extracting linguistic features via NaijaBERT..."):
            raw_data = analyze_with_naijabert(user_input)

        with st.spinner("Synthesizing reasoning via Qwen LLM..."):
            explanation = generate_ai_explanation(user_input, raw_data)

        st.success("Analysis Complete!")

        # Metric Columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sentiment", raw_data['Sentiment'])
        col2.metric("Plutchik Emotion", raw_data['Emotion'])
        col3.metric("Moral Violation", raw_data['Moral'])

        tox_val = raw_data['Toxicity']
        try:
            tox_int = int(tox_val)
            if tox_int >= 3:
                col4.error(f"Toxicity Level: {tox_val}/4")
            elif tox_int > 0:
                col4.warning(f"Toxicity Level: {tox_val}/4")
            else:
                col4.success(f"Toxicity Level: {tox_val}/4")
        except ValueError:
            col4.info(f"Toxicity Level: {tox_val}")

        st.divider()

        st.subheader("🧠 PidginPadi Psychological & Linguistic Breakdown")
        st.info(explanation)
