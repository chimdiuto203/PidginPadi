"""
PidginPadi - AI Moderator (Streamlit)

Expected classifier bundle (produced by train_classifiers.py):

    classifiers = {
        "sentiment": <fitted sklearn estimator>,
        "emotion":   <fitted sklearn estimator>,
        "moral":     <fitted sklearn estimator>,
        "toxicity":  <fitted sklearn estimator>,
    }

Loading notes:
  * The bundle is read with joblib.load ONLY. A joblib file embeds raw numpy
    bytes, so the plain `pickle.load` fallback that used to be here always
    failed with "invalid load key" and only produced a misleading second error.
  * Every download is validated before use. The artifact this project shipped
    before was 99.8% 0xFF padding, which starts with a perfectly valid pickle
    header and therefore slipped past the old 16-byte header check.
  * The NaijaBERT encoder MUST be the same encoder used during training.
    Swapping it for a different checkpoint silently produces nonsense scores.
    It is published in the same Hugging Face repo as the classifiers, under the
    'naijabert_offline/' folder.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import shutil
import zipfile

import joblib
import requests
import streamlit as st
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from llama_cpp import Llama
from transformers import AutoModel, AutoTokenizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


HF_REPO_ID = "pidginpadi/pkl"
CLASSIFIERS_FILENAME = "pidginpadi_classifiers.pkl"
EXPECTED_KEYS = ("sentiment", "emotion", "moral", "toxicity")

ENCODER_SUBFOLDER = "naijabert_offline"


NAIJABERT_RELEASE_URL = os.environ.get("NAIJABERT_RELEASE_URL", "")

# Optional override: a different HF repo id that holds the encoder.
NAIJABERT_HF_REPO = os.environ.get("NAIJABERT_HF_REPO", "")

# --- AI Model ---
QWEN_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
QWEN_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


st.set_page_config(
    page_title="PidginPadi | AI Moderator", page_icon="🇳🇬", layout="wide"
)


def cache_root() -> str:
    """Writable directory for downloaded assets (the repo mount is read-only)."""
    base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    root = os.path.join(base, "pidginpadi")
    os.makedirs(root, exist_ok=True)
    return root


def file_looks_corrupt(path: str):
    """Return (is_corrupt, reason) for a downloaded artifact."""
    if not os.path.exists(path):
        return True, "file does not exist"
    if not os.path.isfile(path):
        return True, "path is not a regular file"

    size = os.path.getsize(path)
    if size < 100:
        return True, f"file is only {size} bytes"

    with open(path, "rb") as f:
        header = f.read(16)
        f.seek(max(0, size - 8192))
        tail = f.read(8192)

    if header.lstrip().startswith(b"<"):
        return True, "file contains HTML (404 / error page saved instead of the asset)"
    if header.startswith(b"\xff\xff\xff\xff"):
        return True, "file starts with 0xFF padding"
    if len(tail) == 8192 and tail.count(b"\xff") == 8192:
        return True, "file ends with 0xFF padding (truncated or padded upload)"
    return False, ""


def purge_hf_cache(path: str) -> None:
    """Remove a cached Hugging Face download so the next call re-downloads it."""
    for candidate in (path, os.path.realpath(path)):
        try:
            if os.path.isfile(candidate):
                os.remove(candidate)
        except OSError:
            pass


def download_file(url: str, dest_path: str) -> None:
    """Stream a URL to disk atomically."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.part"
    try:
        with requests.get(url, stream=True, timeout=180, allow_redirects=True) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_classifiers(classifiers_path: str):
    """Load and validate the classifier bundle."""
    corrupt, reason = file_looks_corrupt(classifiers_path)
    if corrupt:
        raise RuntimeError(
            f"Classifier bundle '{classifiers_path}' is not usable: {reason}. "
            f"(size={os.path.getsize(classifiers_path)} bytes)"
        )

    try:
        bundle = joblib.load(classifiers_path)
    except Exception as err:
        raise RuntimeError(
            f"Could not unpickle '{classifiers_path}'. The artifact is corrupt, "
            f"truncated, or was built with an incompatible scikit-learn version. "
            f"Original error: {type(err).__name__}: {err}"
        ) from err

    if not isinstance(bundle, dict):
        raise RuntimeError(
            f"Classifier bundle must be a dict with keys {EXPECTED_KEYS}, "
            f"got {type(bundle).__name__}."
        )

    missing = [key for key in EXPECTED_KEYS if key not in bundle]
    if missing:
        raise RuntimeError(
            f"Classifier bundle is missing keys {missing}. "
            f"Found: {sorted(bundle.keys())}"
        )

    for key, estimator in bundle.items():
        if not hasattr(estimator, "predict"):
            raise RuntimeError(
                f"Classifier '{key}' is a {type(estimator).__name__} and has no "
                f"predict() method."
            )

    return bundle


def fetch_classifiers() -> str:
    """Download the classifier bundle, recovering from a bad cached copy."""
    path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=CLASSIFIERS_FILENAME
    )

    corrupt, reason = file_looks_corrupt(path)
    if corrupt:
        st.warning(
            f"Cached classifier bundle was unusable ({reason}). Re-downloading "
            f"from Hugging Face..."
        )
        purge_hf_cache(path)
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CLASSIFIERS_FILENAME,
            force_download=True,
        )
        corrupt, reason = file_looks_corrupt(path)
        if corrupt:
            raise RuntimeError(
                f"The classifier bundle published at "
                f"'{HF_REPO_ID}/{CLASSIFIERS_FILENAME}' is corrupt "
                f"({reason}). You must re-generate it and upload a new version."
            )

    return path


def is_encoder_folder(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def ensure_naijabert():
    """Return (source, subfolder) for the NaijaBERT encoder.

    `source` is either a local directory path or a Hugging Face repo id.
    `subfolder` is the subfolder inside that repo, or None for a local path.
    """
    local_dir = os.path.join(BASE_DIR, ENCODER_SUBFOLDER)
    if is_encoder_folder(local_dir):
        return local_dir, None

    cached_dir = os.path.join(cache_root(), ENCODER_SUBFOLDER)
    if is_encoder_folder(cached_dir):
        return cached_dir, None

    if NAIJABERT_HF_REPO:
        return NAIJABERT_HF_REPO, ENCODER_SUBFOLDER

    try:
        st.info(
            f"Downloading the NaijaBERT encoder from "
            f"'{HF_REPO_ID}/{ENCODER_SUBFOLDER}/' (about 700 MB, this happens "
            f"once)..."
        )
        snapshot_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[
                f"{ENCODER_SUBFOLDER}/config.json",
                f"{ENCODER_SUBFOLDER}/tokenizer*",
                f"{ENCODER_SUBFOLDER}/special_tokens_map.json",
                f"{ENCODER_SUBFOLDER}/vocab.*",
                f"{ENCODER_SUBFOLDER}/merges.txt",
                f"{ENCODER_SUBFOLDER}/added_tokens.json",
                f"{ENCODER_SUBFOLDER}/model.safetensors",
                f"{ENCODER_SUBFOLDER}/pytorch_model.bin",
            ],
        )
        downloaded = os.path.join(snapshot_dir, ENCODER_SUBFOLDER)
        if is_encoder_folder(downloaded):
            return downloaded, None
        raise RuntimeError(
            f"'{HF_REPO_ID}/{ENCODER_SUBFOLDER}' does not contain a "
            f"config.json."
        )
    except Exception as err:
        st.warning(f"Could not fetch the encoder from Hugging Face: {err}")

    if NAIJABERT_RELEASE_URL:
        try:
            st.info(f"Downloading the NaijaBERT encoder from {NAIJABERT_RELEASE_URL}")
            zip_path = os.path.join(cache_root(), f"{ENCODER_SUBFOLDER}.zip")
            if not os.path.exists(zip_path):
                download_file(NAIJABERT_RELEASE_URL, zip_path)
            extract_dir = os.path.join(cache_root(), "encoder_extract")
            shutil.rmtree(extract_dir, ignore_errors=True)
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)

            for candidate in (
                os.path.join(extract_dir, ENCODER_SUBFOLDER),
                extract_dir,
            ):
                if is_encoder_folder(candidate):
                    shutil.rmtree(cached_dir, ignore_errors=True)
                    shutil.move(candidate, cached_dir)
                    return cached_dir, None

            raise RuntimeError(
                f"'{zip_path}' did not contain a '{ENCODER_SUBFOLDER}' folder "
                f"with a config.json."
            )
        except Exception as err:
            st.warning(f"Could not fetch the encoder release asset: {err}")

    raise RuntimeError(
        f"Could not locate the NaijaBERT encoder. Expected it at "
        f"'{HF_REPO_ID}/{ENCODER_SUBFOLDER}/' on Hugging Face. Upload the "
        f"'naijabert_offline' folder there, or set NAIJABERT_RELEASE_URL to a "
        f"GitHub Release asset, or set NAIJABERT_HF_REPO to the repo id used "
        f"during training."
    )


# ============================================================
# PIDGIN DETECTION
# ============================================================

# Common Nigerian Pidgin vocabulary and patterns
PIDGIN_VOCABULARY = {
    # Pronouns & basic words
    "una", "dey", "na", "wey", "wetin", "weti", "make", "abi", "shey", "seh",
    "don", "go", "wan", "fit", "sabi", "sab", "no", "nor", "nor be", "no be",
    "i", "you", "u", "we", "dem", "im", "am", "e", "o", "oh", "eh", "ah",
    
    # Common verbs
    "chop", "commot", "waka", "siddon", "gats", "gatz", "dey", "don", "done",
    "fit", "no fit", "nor fit", "wan", "want", "dey go", "dey come", "dey do",
    "carry", "drop", "pick", "put", "take", "give", "collect", "buy", "sell",
    "hustle", "grind", "ball", "flex", "chill", "catch", "cruise", "vibe",
    
    # Common nouns/terms
    "oga", "madam", "bros", "bro", "sis", "padi", "gee", "guy", "babe", "bobo",
    "mumu", "olodo", "ode", "agbaya", "ashawo", "olosho", "runs", "belle",
    "head", "hand", "leg", "eye", "ear", "mouth", "teeth", "nose", "face",
    "wahala", "wahala", "koboko", "cane", "whip", "slap", "beat", "flog",
    "thunder", "lightning", "sango", "amadioha", "ogun", "ayelala", "juju",
    "babalawo", "native", "doctor", "witch", "wizard", "ritual", "sacrifice",
    "shawarma", "eba", "egusi", "amala", "pounded", "yam", "rice", "beans",
    "garri", "akpu", "fufu", "semovita", "semolina", "spaghetti", "indomie",
    
    # Expressions & markers
    "abeg", "pls", "please", "biko", "ejo", "ejoo", "make", "let", "make we",
    "make i", "make una", "make dem", "i dey", "u dey", "you dey", "we dey",
    "dem dey", "una dey", "i don", "u don", "you don", "we don", "dem don",
    "i go", "u go", "you go", "we go", "dem go", "una go", "i wan", "u wan",
    "i fit", "u fit", "you fit", "we fit", "dem fit", "una fit", "i no", "u no",
    "you no", "we no", "dem no", "una no", "nor", "nor be", "no be", "na so",
    "na wa", "nawa", "na wa o", "chai", "chei", "cheii", "cheiii", "hian",
    "hian!", "wow", "omo", "omo!", "omo o", "eh", "eh?", "eh!", "ah", "ah!",
    
    # Greetings & social
    "how far", "howfar", "how u dey", "how body", "how life", "how work",
    "good morning", "good afternoon", "good evening", "happy sunday",
    "happy monday", "happy new", "happy birthday", "congratulations", "congrats",
    "well done", "welldone", "you try", "you too much", "respect", "hail",
    "big up", "shout out", "shoutout", "love", "bless", "blessed", "favour",
    "breakthrough", "testimony", "miracle", "open door", "season", "time",
    
    # Negation & emphasis
    "nor", "nor be", "no be", "no be say", "nor be say", "i swear", "i swr",
    "i swear to god", "wallahi", "wallah", "tallahi", "by god", "by grace",
    "god willing", "inshallah", "masha allah", "alhamdulillah", "subhanallah",
    
    # Question words in Pidgin context
    "wetin dey happen", "wetin happen", "wetin you dey do", "where you dey",
    "when you go come", "why you dey", "how you take", "who be that", "which one",
    
    # Money & business
    "kobo", "naira", "change", "balance", "account", "transfer", "alert",
    "pos", "atm", "bank", "loan", "debt", "credit", "debit", "savings",
    "investment", "business", "client", "customer", "profit", "loss",
    
    # Common phrases
    "no wahala", "no problem", "easy", "easy oh", "small small", "small small oh",
    "step by step", "one by one", "day by day", "small time", "quick quick",
    "sharp sharp", "fast fast", "slow slow", "gentle gentle", "cool down",
    "calm down", "relax", "take am easy", "no vex", "no worry", "no panic",
    
    # Technology & social media
    "follow", "unfollow", "like", "comment", "share", "retweet", "quote",
    "dm", "message", "chat", "whatsapp", "telegram", "instagram", "twitter",
    "facebook", "tiktok", "youtube", "snapchat", "link", "bio", "profile",
    "post", "story", "reel", "video", "photo", "picture", "screenshot",
    
    # Time
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "week", "month", "year", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "now", "later", "soon", "just now", "small time ago", "long time", "long time ago",
}

# Compile regex patterns for common Pidgin structures
PIDGIN_PATTERNS = [
    r"\b(una|dey|wey|wetin|weti|make|abi|shey|seh)\b",
    r"\b(na|no|nor)\s+(be|so|wa|wao|good|bad|true|false)\b",
    r"\b(i|u|you|we|dem|im|am|e)\s+(dey|don|go|wan|fit|no|nor)\b",
    r"\b(abeg|biko|ejo|pls|please)\b",
    r"\b(oga|madam|bros|padi|gee|guy|babe|bobo|mumu|olodo|ode|agbaya)\b",
    r"\b(wahala|koboko|cane|whip|slap|beat|flog)\b",
    r"\b(thunder|sango|amadioha|ogun|ayelala|juju|babalawo)\b",
    r"\b(chop|commot|waka|siddon|gats|gatz|hustle|grind|ball|flex|chill)\b",
    r"\b(omo|chai|chei|hian|wallahi|wallah|tallahi)\b",
    r"\b(how\s+(far|body|life|work|u\s+dey)|wetin\s+(dey|happen|you\s+dey))\b",
    r"\b(no\s+wahala|no\s+problem|small\s+small|sharp\s+sharp|step\s+by\s+step)\b",
    r"\b(god\s+(bless|blessed|good|willing)|inshallah|masha\s+allah|alhamdulillah)\b",
]

import re
PIDGIN_REGEX = [re.compile(p, re.IGNORECASE) for p in PIDGIN_PATTERNS]


def is_nigerian_pidgin(text: str, threshold: float = 0.3) -> tuple[bool, float, dict]:
    """
    Detect if text is Nigerian Pidgin English.
    
    Returns:
        (is_pidgin: bool, confidence: float, details: dict)
    """
    if not text or not text.strip():
        return False, 0.0, {"reason": "Empty text"}
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = len(words)
    
    if total_words == 0:
        return False, 0.0, {"reason": "No words found"}
    
    # Count vocabulary matches
    vocab_matches = sum(1 for w in words if w in PIDGIN_VOCABULARY)
    vocab_ratio = vocab_matches / total_words
    
    # Count pattern matches
    pattern_matches = sum(1 for regex in PIDGIN_REGEX if regex.search(text_lower))
    pattern_score = min(pattern_matches / 5.0, 1.0)  # Normalize
    
    # Check for Pidgin-specific orthography (phonetic spelling)
    pidgin_ortho_patterns = [
        r"\b(don|dey|wen|wetin|weti|wey|sabi|sab|fit|wan|gats|gatz)\b",
        r"\b(una|dem|im|am|nor|nor be|no be|make|abi|shey|seh)\b",
        r"\b(oga|bros|padi|mumu|olodo|ashawo|wahala|chai|omo)\b",
    ]
    ortho_matches = sum(1 for p in pidgin_ortho_patterns if re.search(p, text_lower))
    ortho_score = min(ortho_matches / 3.0, 1.0)
    
    # Combined score (weighted)
    confidence = (vocab_ratio * 0.4) + (pattern_score * 0.35) + (ortho_score * 0.25)
    
    # Boost for very short texts with clear Pidgin markers
    if total_words <= 5 and (vocab_matches > 0 or pattern_matches > 0 or ortho_matches > 0):
        confidence = max(confidence, 0.5)
    
    details = {
        "total_words": total_words,
        "vocab_matches": vocab_matches,
        "vocab_ratio": round(vocab_ratio, 3),
        "pattern_matches": pattern_matches,
        "ortho_matches": ortho_matches,
        "confidence": round(confidence, 3),
        "matched_words": [w for w in words if w in PIDGIN_VOCABULARY][:20]
    }
    
    return confidence >= threshold, confidence, details


def analyze_with_naijabert(tweet):
    inputs = tokenizer(
        tweet, return_tensors="pt", truncation=True, max_length=128, padding=True
    )
    with torch.no_grad():
        outputs = encoder(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    results = {
        "Sentiment": classifiers["sentiment"].predict(cls_embedding)[0],
        "Emotion": classifiers["emotion"].predict(cls_embedding)[0],
        "Moral": classifiers["moral"].predict(cls_embedding)[0],
        "Toxicity": classifiers["toxicity"].predict(cls_embedding)[0],
    }
    return results


def generate_ai_explanation(tweet, analytical_data):
    prompt = f"""<|system|>
You are PidginPadi, an expert AI in Nigerian Pidgin English, psychology, and content moderation.
Analyze the user's tweet based on the provided raw data. Keep your explanation professional, concise (under 4 sentences), and explain how the Pidgin phrasing leads to this conclusion.
<|user|>
Tweet: "{tweet}"
Raw Data: Sentiment={analytical_data['Sentiment']}, Emotion={analytical_data['Emotion']}, Moral Violation={analytical_data['Moral']}, Toxicity Scale={analytical_data['Toxicity']}/4.
Provide your expert explanation.
<|assistant|>
"""
    response = llm(prompt, max_tokens=250, stop=["<|user|>"], temperature=0.3)
    return response["choices"][0]["text"].strip()


@st.cache_resource(show_spinner=False)
def load_models():
    # A. Classifiers
    classifiers = load_classifiers(fetch_classifiers())

    # B. NaijaBERT encoder
    encoder_source, encoder_subfolder = ensure_naijabert()
    if os.path.isdir(encoder_source):
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_source, local_files_only=True
        )
        encoder = AutoModel.from_pretrained(
            encoder_source, local_files_only=True
        )
    elif encoder_subfolder:
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_source, subfolder=encoder_subfolder
        )
        encoder = AutoModel.from_pretrained(
            encoder_source, subfolder=encoder_subfolder
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(encoder_source)
        encoder = AutoModel.from_pretrained(encoder_source)

    # C. Offline Model
    local_gguf = os.path.join(BASE_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    if os.path.exists(local_gguf):
        model_path = local_gguf
    else:
        model_path = hf_hub_download(
            repo_id=QWEN_REPO_ID, filename=QWEN_FILENAME
        )

    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    return classifiers, tokenizer, encoder, llm


try:
    with st.spinner(
        "Initializing AI Models (downloading models on first run, this may take a minute)..."
    ):
        classifiers, tokenizer, encoder, llm = load_models()
except Exception as e:
    st.error(f"Failed to initialize AI models: {type(e).__name__}: {e}")
    st.exception(e)
    st.stop()


st.title("🇳🇬 PidginPadi: AI Moderator")
st.markdown(
    "**Architecture:** Dual-Model Pipeline (NaijaBERT + Qwen-0.5B-GGUF Edge Inference)"
)
st.markdown("By **Chinatu Anyanwu**")
st.divider()

# Language info box
with st.expander("ℹ️ About PidginPadi - Language Requirements", expanded=False):
    st.markdown("""
    **PidginPadi only analyzes Nigerian Pidgin English.** 
    
    **Supported:** Nigerian Pidgin (Naija), e.g.:
    - *"You too much bro God bless you!"*
    - *"Thunder fire you and your fake promises"*
    - *"Abeg who get charger make I borrow?"*
    - *"E no go beta for you, failure for life"*
    
    **Not supported:** Standard English, other languages, or code without Pidgin.
    
    The system will automatically detect and reject non-Pidgin text.
    """)

user_input = st.text_area(
    "Paste a Nigerian Pidgin tweet below to analyze:",
    height=100,
    placeholder="e.g., why dont you wanna stream my song if na because of my character no vex oya stream am...",
)

if st.button("Run AI Analysis", type="primary"):
    if not user_input.strip():
        st.warning("Abeg, type something first!")
    else:
        # --- PIDGIN DETECTION CHECK ---
        with st.spinner("🔍 Verifying Nigerian Pidgin..."):
            is_pidgin, confidence, details = is_nigerian_pidgin(user_input)
        
        if not is_pidgin:
            st.error("❌ **This text does not appear to be Nigerian Pidgin English.**")
            st.markdown(f"""
            **Detection confidence:** {confidence:.1%} (threshold: 30%)
            
            **Details:**
            - Words analyzed: {details.get('total_words', 0)}
            - Pidgin vocabulary matches: {details.get('vocab_matches', 0)} ({details.get('vocab_ratio', 0):.1%})
            - Pattern matches: {details.get('pattern_matches', 0)}
            - Orthography matches: {details.get('ortho_matches', 0)}
            - Matched Pidgin words: {', '.join(details.get('matched_words', [])[:10]) or 'None'}
            
            **Please enter text in Nigerian Pidgin English (Naija).**
            
            *Examples of valid Pidgin:*
            - *"You too much bro God bless you!"*
            - *"Thunder fire you and your fake promises"*
            - *"Abeg who get charger make I borrow?"*
            - *"E no go beta for you, failure for life"*
            """)
            st.stop()
        
        # Show detection success
        st.success(f"✅ Nigerian Pidgin detected (confidence: {confidence:.1%})")
        
        # --- RUN ANALYSIS ---
        with st.spinner("Extracting linguistic features via NaijaBERT..."):
            raw_data = analyze_with_naijabert(user_input)

        with st.spinner("Synthesizing reasoning via Qwen LLM..."):
            explanation = generate_ai_explanation(user_input, raw_data)

        st.success("Analysis Complete!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sentiment", raw_data["Sentiment"])
        col2.metric("Plutchik Emotion", raw_data["Emotion"])
        col3.metric("Moral Violation", raw_data["Moral"])

        tox_val = raw_data["Toxicity"]
        try:
            tox_int = int(tox_val)
            if tox_int >= 3:
                col4.error(f"Toxicity Level: {tox_val}/4")
            elif tox_int > 0:
                col4.warning(f"Toxicity Level: {tox_val}/4")
            else:
                col4.success(f"Toxicity Level: {tox_val}/4")
        except (ValueError, TypeError):
            col4.info(f"Toxicity Level: {tox_val}")

        st.divider()

        st.subheader("🧠 PidginPadi Psychological & Linguistic Breakdown")
        st.info(explanation)
