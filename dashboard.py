import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

# get the largest thumbnail size bc sizes differ
# if there is no thumbnail, use replacement photo
books["large thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large thumbnail"] = np.where(
    books["large thumbnail"].isna(),
    "placeholder.jpg",
    books["large thumbnail"]
)

# Create tagged_description if it is not already in books_with_emotions.csv
if "tagged_description" not in books.columns:
    books["tagged_description"] = books["isbn13"].astype(str) + " " + books["description"].astype(str)

loader = DataFrameLoader(books, page_content_column="tagged_description")
documents = loader.load()
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())


def get_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,  # looks better w/ 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by=["joy"], ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by=["surprise"], ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by=["anger"], ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by=["fear"], ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by=["sadness"], ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recs = get_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recs.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large thumbnail"], caption))

    # Return both the gallery items and the raw DataFrame stored in State
    return results, recs


def display_selected_book(evt: gr.SelectData, recs_df: pd.DataFrame):
    if recs_df is None or recs_df.empty:
        return ""

    # evt.index provides the index of the clicked book in the gallery
    selected_row = recs_df.iloc[evt.index]

    authors_split = selected_row["authors"].split(";")
    if len(authors_split) == 2:
        authors_str = f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    else:
        authors_str = selected_row["authors"]

    return (
        f"### {selected_row['title']}\n"
        f"**Author(s):** {authors_str} | **Category:** {selected_row['simple_categories']}\n\n"
        f"{selected_row['description']}"
    )


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Ocean()) as dashboard:
    # State variable to keep track of the current recommended books DataFrame
    current_recs_state = gr.State()

    with gr.Row():
        gr.Markdown("# Semantic Book Recommender")
        toggle_dark_btn = gr.Button("🌓 Toggle", scale=0)

    with gr.Row():
        user_query = gr.Textbox(label="Please describe a book you want to read!",
                                placeholder="e.g. A story about friendship")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Recommend me something!")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books (Click any book to view full details below)", columns=8, rows=2)

    # Dedicated box for the unabridged description
    gr.Markdown("### Full Book Details")
    book_details = gr.Markdown(
        value="*Click on any book above to read its complete description.*",
        container=True
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[output, current_recs_state]
    )

    # Triggered whenever any card in the gallery is clicked
    output.select(
        fn=display_selected_book,
        inputs=[current_recs_state],
        outputs=book_details
    )

    toggle_dark_btn.click(
        fn=None,
        js="""
            () => {
                document.body.classList.toggle('dark');
            }
            """
    )

    if __name__ == "__main__":
        dashboard.launch()