import re
import pandas as pd
import openai
from collections import Counter
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_random, retry_if_exception_type

# LangChain imports (using ChatPromptTemplate only)
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import plotly.graph_objects as go

# Set your OpenAI API key from Streamlit secrets.
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Use GPT-4 for all model calls.
MODEL_NAME = "gpt-3.5-turbo"

# Backoff strategies for API calls.
backoff_classification = {
    "wait": wait_random_exponential(min = 0.15, max = 0.35),
    "stop": stop_after_attempt(20)
}
backoff_attribute = {
    "wait": wait_random(min = 0.15, max = 0.20),
    "stop": stop_after_attempt(5)
}


@st.cache_data(show_spinner = False)
def get_df():
    """Loads the raw reviews CSV. Adjust the path as needed."""
    df = pd.read_csv("data/combined_speaker_reviews.csv")
    return df[:50]


@retry(wait = backoff_classification["wait"], stop = backoff_classification["stop"], reraise = True)
def run_classification_with_backoff(chain, **kwargs):
    return chain.batch(**kwargs)


@retry(wait = backoff_attribute["wait"], stop = backoff_attribute["stop"], reraise = True)
def run_attribute_extraction_with_backoff(chain, **kwargs):
    return chain.batch(**kwargs)


def filter_special_characters(text):
    """Removes non-alphabetic characters from the text."""
    return re.sub(r"[^a-zA-Z\s]", "", text).strip()


def get_attributes(reviews_df):
    """
    Extracts candidate attributes from reviews using an LLM.
    Processes up to 500 reviews per product and returns the top 5 most common attributes.
    Synonyms are standardized—for example, if a review mentions either "sound" or "sound quality",
    always return "sound quality".
    """
    # Limit to 500 reviews per product (grouped by 'asin')
    reviews_df = reviews_df.groupby("asin").head(500)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Here is a product review of a bluetooth speaker: {review}

        Extract the key product features mentioned in the review.
        For example, if the review mentions both "sound" and "sound quality", return only one of them since they mean the same thing. Be intelligent
        Only return product attributes that are explicitly mentioned, as nouns.
        Each attribute must be at most two words and duplicates should be avoided.
        Limit your answer to a maximum of 5 attributes.
        """
    )

    output_parser = StrOutputParser()
    model = ChatOpenAI(model = MODEL_NAME, temperature = 0, max_tokens = 45, top_p = 1)
    chain = ({"review": RunnablePassthrough()} | prompt_template | model | output_parser)

    review_texts = reviews_df["reviewText"].astype(str).tolist()
    # Truncate reviews to 1000 characters to speed up processing.
    review_texts = [text[:1000] for text in review_texts]

    attributes = run_attribute_extraction_with_backoff(chain, inputs = review_texts, config = {"max_concurrency": 10})

    all_attrs = []
    for attr in attributes:
        cleaned = filter_special_characters(attr)
        parts = cleaned.split("\n")
        for part in parts:
            part_clean = part.strip().lower()
            if part_clean in ("sound", "sound quality"):
                part_clean = "sound quality"
            if part_clean:
                all_attrs.append(part_clean)

    attribute_counts = Counter(all_attrs)
    top_attributes = [attr for attr, _ in attribute_counts.most_common(5)]
    return top_attributes


def score_reviews(reviews_df, attributes):
    """
    For each review, uses an LLM to assign scores (0 to 3) for each attribute.
    Expected output per review: "feature:score, feature:score, ..."
    """
    op_structure = "feature:score"
    prompt_template = ChatPromptTemplate.from_template(
        f"""
        Here is a product review: {{review}}
        Determine how the reviewer rates this product in relation to these features:
        {", ".join(attributes)}
        Only provide a score (between 0 and 3) for each feature.
        If a feature is not mentioned, assign a score of 0.
        Use the following format without additional commentary:
        {op_structure}
        """
    )
    output_parser = StrOutputParser()
    model = ChatOpenAI(model = MODEL_NAME, temperature = 0.5, max_tokens = 45)
    chain = ({"review": RunnablePassthrough()} | prompt_template | model | output_parser)

    review_texts = reviews_df["reviewText"].astype(str).tolist()
    review_texts = [text[:1000] for text in review_texts]

    results = run_classification_with_backoff(chain, inputs = review_texts, config = {"max_concurrency": 20})
    reviews_df = reviews_df.copy()
    reviews_df["scores"] = results
    return reviews_df


def generate_summary_input(df):
    """
    Prepares a list of dictionaries (one per review) with review text and scores.
    """
    inputs = []
    for _, row in df.iterrows():
        inputs.append({
            "review": row["reviewText"],
            "scores": row["scores"]
        })
    return inputs


def summarize_reviews(product):
    """
    For a given product, loads the persisted classified reviews,
    converts them into Document objects, and uses a custom map-reduce routine
    (built with the new RunnableSequence style) to generate a consolidated summary.
    """
    df = pd.read_csv("classified_reviews.csv")
    df = df[df["productName"] == product]
    data = generate_summary_input(df)

    # Convert our list of dictionaries into Document objects.
    # The RecursiveJsonSplitter returns Documents with a "page_content" attribute.
    splitter = RecursiveJsonSplitter()
    docs = splitter.create_documents(texts = data)

    # Create the LLM instance.
    model = ChatOpenAI(model = MODEL_NAME, temperature = 0)

    # --- Mapping Step ---
    # Define the mapping prompt (expects {page_content}).
    map_prompt = ChatPromptTemplate.from_template(
        "You are provided with a single product review in JSON format:\n{page_content}\nGenerate a concise summary of this review. Summary:"
    )
    # Create the mapping chain using the pipe operator.
    mapping_chain = map_prompt | model
    mapped_outputs = []
    for doc in docs:
        result = mapping_chain.invoke({"page_content": doc.page_content})
        mapped_str = str(result) if result is not None else ""
        mapped_outputs.append(mapped_str)

    # --- Reducing Step ---
    # Define the reducing prompt (expects {summaries}).
    combine_prompt = ChatPromptTemplate.from_template(
        "You are provided with several summaries:\n{summaries}\nCombine these into one final, concise summary. Final Summary:"
    )
    reduce_chain = combine_prompt | model
    summaries_str = "\n".join(mapped_outputs)
    final_summary = reduce_chain.invoke({"summaries": summaries_str})

    return final_summary


def classify_reviews():
    """
    Orchestrates the classification process:
      - Loads the raw reviews.
      - Dynamically extracts attributes.
      - Scores each review.
      - Saves the results to 'classified_reviews.csv'.
    """
    df = get_df()
    attributes = get_attributes(df)
    scored_df = score_reviews(df, attributes)
    scored_df.to_csv("classified_reviews.csv", index = False)



# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner = False)
def load_classified_reviews():
    try:
        df = pd.read_csv("classified_reviews.csv")
        return df
    except FileNotFoundError:
        return None


def parse_scores(score_str):
    """Parses a score string into a dictionary of {feature: score}.
       Also standardizes features like 'sound' and 'sound quality' to 'sound quality'."""
    score_dict = {}
    try:
        parts = score_str.split("\n")
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip().lower()
                # Standardize synonyms.
                if key in ("sound", "sound quality"):
                    key = "sound quality"
                try:
                    score = float(value.strip())
                except ValueError:
                    score = 0
                # If the key already exists, average the scores.
                if key in score_dict:
                    score_dict[key] = (score_dict[key] + score) / 2.0
                else:
                    score_dict[key] = score
    except Exception:
        pass
    return score_dict


def aggregate_scores(df):
    """
    Aggregates scores per product.

    Returns:
      - averaged: a dictionary mapping product -> {feature: average_score}
      - all_attributes: a set of all features encountered.
    """
    product_scores = {}
    product_counts = {}
    all_attributes = set()
    for _, row in df.iterrows():
        product = row.get("productName")
        if not product:
            continue
        score_str = row.get("scores", "")
        scores = parse_scores(score_str)
        all_attributes.update(scores.keys())
        if product not in product_scores:
            product_scores[product] = {}
            product_counts[product] = 0
        for attr, val in scores.items():
            product_scores[product][attr] = product_scores[product].get(attr, 0) + val
        product_counts[product] += 1
    averaged = {}
    for product, score_dict in product_scores.items():
        avg_dict = {}
        for attr in all_attributes:
            avg_dict[attr] = score_dict.get(attr, 0) / product_counts[product]
        averaged[product] = avg_dict
    return averaged, all_attributes


def generate_radar_chart(df, selected_product):
    """
    Generates a radar chart based on review scores.

    If selected_product is "All Products", then one trace per product is plotted.
    Otherwise, only the selected product's averaged scores are plotted.
    """
    averaged, all_attributes = aggregate_scores(df)
    # Sort attributes for consistent ordering.
    categories = sorted(list(all_attributes))
    fig = go.Figure()
    if selected_product == "All Products":
        # Plot each product on the same radar chart.
        for product, scores in averaged.items():
            values = [scores.get(attr, 0) for attr in categories]
            # Close the loop
            values += [values[0]]
            theta = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(r = values, theta = theta, fill = 'toself', name = product))
    else:
        # Plot only the selected product.
        scores = averaged.get(selected_product, {})
        values = [scores.get(attr, 0) for attr in categories]
        values += [values[0]]
        theta = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(r = values, theta = theta, fill = 'toself', name = selected_product))
    fig.update_layout(
        polar = dict(
            radialaxis = dict(visible = True, range = [0, 3])
        ),
        showlegend = True,
        margin = dict(l = 20, r = 20, t = 40, b = 20)
    )
    return fig


# ------------------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title = "Amazon Speaker Reviews Classification", layout = "wide")
    st.title("Amazon Speaker Reviews Classification")
    st.session_state.classification_complete = False

    # Buttons to run or load the classifier.
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run classifier and generate radar chart & summary"):
            with st.spinner("Running classifier..."):
                st.session_state.classification_complete = False
                classify_reviews()
            st.success("Classification complete!")
            st.session_state.classification_complete = True
    with col2:
        if st.button("Load persisted data and generate radar chart & summary"):
            st.info("Using persisted classified reviews.")
            st.session_state.classification_complete = True


    if st.session_state.get("classification_complete", False):
        df = load_classified_reviews()
        if df is None or df.empty:
            st.error("classified_reviews.csv not found or empty. Please run the classifier first.")
            return

        st.subheader("Classified Reviews Data")
        st.dataframe(df, use_container_width = True)

        st.subheader("Review Scores")
        if "productName" in df.columns:
            unique_products = sorted(df["productName"].unique())
        else:
            st.error("Column 'productName' not found in classified reviews data.")
            return
        # Create a selectbox with an "All Products" option.
        options = ["All Products"] + unique_products
        selected_product = st.selectbox("Select a product to view its radar chart", options = options)
        fig = generate_radar_chart(df, selected_product)
        st.plotly_chart(fig, use_container_width = True)

        st.subheader("Generate Product Review Summary")
        if unique_products:
            selected_summary_product = st.selectbox("Select a product for summary", options = unique_products)
            if selected_summary_product:
                with st.spinner("Generating summary..."):
                    summary = summarize_reviews(selected_summary_product)
                st.markdown("### Summary")
                st.write(summary)
        else:
            st.warning("No products available for summary.")


if __name__ == "__main__":
    main()
