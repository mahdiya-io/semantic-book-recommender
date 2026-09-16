# Semantic Book Recommender

An semantic recommender engine that moves beyond simple keyword and title searches. By pairing dense vector representations with fine-grained emotion profiling and zero-shot category classification, this application surfaces books based on concepts, themes, plot details, and desired emotional tones.

Built with **LangChain**, **ChromaDB**, **Hugging Face Transformers**, and **Gradio**

---

## Features

* **Natural Language Semantic Search:** Matches descriptive narrative prompts (e.g., *"a story about friendship and loss in space"*) using dense vector embeddings powered by `BAAI/bge-large-en-v1.5` and stored in a local Chroma vector database.
* **Emotion-Driven Re-Ranking:** Ranks results dynamically across tones like Joy, Surprise, Anger, Fear, and Sadness using emotion scores extracted sentence-by-sentence with `j-hartmann/emotion-english-distilroberta-base`.
* **Zero-Shot Category Categorization:** Fills missing metadata and standardizes categories using `facebook/bart-large-mnli` zero-shot classification.
* **Interactive Gradio Dashboard:** 
  * Responsive 16-card gallery featuring high-resolution Google Books thumbnails.
  * Instant Light/Dark mode toggle.
  * Clean truncated card captions with an on-click **Full Book Details** view to read unabridged book descriptions with zero UI overflow.
 
---

## Getting Started
1. Clone the repo
```bash
git clone https://github.com/mahdiya-io/semantic-book-recommender.git
cd semantic-book-recommender
```

2. Set up a virtual environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the dashboard!
```bash
python dashboard.py
```
Open http://127.0.0.1:7860 in your browser.

---

## Project Structure
```text
├── books_with_emotions.csv     # Final processed dataset with categories & emotion scores
├── data-preprocessing.ipynb    # Data cleaning, null checks, and length thresholding
├── text-classification.ipynb   # Zero-shot category mapping with BART-large-MNLI
├── sentiment-analysis.ipynb    # Sentence-level DistilRoBERTa emotion scoring pipeline
├── vector-search.ipynb         # Embedding generation, ChromaDB setup, and search testing
├── dashboard.py                # Full Gradio application script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Architecture & Workflow

```text
[Raw Kaggle Dataset] (~7k books)
        │
        ▼
[Data Preprocessing] ──► Deduplicate, filter short descriptions (<25 words)
        │
        ▼
[Text Classification] ──► facebook/bart-large-mnli for zero-shot categorization
        │
        ▼
[Emotion Profiling] ────► j-hartmann/emotion-english-distilroberta-base per sentence
        │
        ▼
[Vector Store & DB] ────► BAAI/bge-large-en-v1.5 embeddings stored in ChromaDB
        │
        ▼
[Gradio Web UI] ────────► Semantic query retrieval + emotion re-sorting + book detail view
```
