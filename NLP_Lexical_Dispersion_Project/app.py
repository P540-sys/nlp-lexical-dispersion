import re

import streamlit as st
import nltk
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Optional spaCy import (Python 3.14 pe fail ho sakta hai)
# ---------------------------------------------------

try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Model nahi mila to blank tokenizer use karo
        nlp = spacy.blank("en")
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    nlp = None
    SPACY_AVAILABLE = False

# ---------------------------------------------------
# NLTK setup (download fail ho to regex tokenizer fallback)
# ---------------------------------------------------

try:
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import word_tokenize

    NLTK_TOKENIZER_OK = True
except Exception:
    NLTK_TOKENIZER_OK = False


def tokenize_nltk(text):
    if NLTK_TOKENIZER_OK:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\w+|[^\w\s]", text)


def tokenize_spacy(text):
    return [token.text for token in nlp(text)]


def find_positions(tokens, target_words):
    positions = {word: [] for word in target_words}
    for index, token in enumerate(tokens):
        key = token.lower()
        if key in positions:
            positions[key].append(index)
    return positions


def plot_dispersion(word_positions, target_words, title):
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, word in enumerate(target_words):
        pos = word_positions[word]
        ax.plot(pos, [i] * len(pos), "|", markersize=15, label=word)

    ax.set_yticks(range(len(target_words)))
    ax.set_yticklabels(target_words)
    ax.set_xlabel("Word Position")
    ax.set_ylabel("Target Words")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    return fig


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="NLP Lexical Dispersion Analyzer",
    page_icon="📚",
    layout="wide",
)

st.title("📚 NLP Lexical Dispersion Analyzer")

st.write(
    "This application analyzes the position and distribution "
    "of selected words in a text using NLTK and spaCy."
)

st.divider()

# ---------------------------------------------------
# Text Input
# ---------------------------------------------------

st.subheader("📝 Enter Your Text")

default_text = """
Moby Dick is a famous novel about the sea and a great whale.
Captain Ahab is searching for the white whale.
The ship travels across the sea.
The captain watches the sea carefully.
The whale appears near the ship.
"""

text = st.text_area(
    "Enter or paste your text here:",
    value=default_text,
    height=200,
)

# ---------------------------------------------------
# Target Words
# ---------------------------------------------------

st.subheader("🔎 Target Words")

words_input = st.text_input(
    "Enter words separated by commas:",
    value="whale, ship, sea, captain",
)

target_words = [
    word.strip().lower()
    for word in words_input.split(",")
    if word.strip()
]

# ---------------------------------------------------
# Choose NLP Library
# ---------------------------------------------------

st.subheader("🧠 Choose NLP Library")

if SPACY_AVAILABLE:
    options = ["NLTK", "spaCy"]
else:
    options = ["NLTK"]
    st.warning(
        "spaCy is not working in this Python environment, "
        "so only NLTK is available."
    )

library = st.radio("Select a method:", options, horizontal=True)

# ---------------------------------------------------
# Analyze Button
# ---------------------------------------------------

if st.button("🔍 Analyze Text"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    if not target_words:
        st.warning("Please enter at least one target word.")
        st.stop()

    if library == "NLTK":
        tokens = tokenize_nltk(text)
    else:
        tokens = tokenize_spacy(text)

    word_positions = find_positions(tokens, target_words)

    st.success(f"Analysis completed using {library}!")

    fig = plot_dispersion(
        word_positions,
        target_words,
        f"Lexical Dispersion Plot using {library}",
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📊 Results")

    for word in target_words:
        count = len(word_positions[word])
        st.write(f"**{word}** → Found {count} time(s)")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "NLP Mini Project | Lexical Dispersion Analysis using NLTK and spaCy"
)