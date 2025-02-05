import re
import pandas as pd
import openai
from collections import Counter
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_random

# LangChain imports
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set your OpenAI API key from Streamlit secrets.
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Use GPT-4 for all model calls.
MODEL_NAME = "gpt-4"

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
    return df[:300]


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
        For example, if the review mentions aspects related to audio, always return "sound quality"
        (do not return both "sound" and "sound quality").
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

    attributes = run_attribute_extraction_with_backoff(chain, inputs = review_texts, config = {"max_concurrency": 20})

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
    map_prompt = PromptTemplate.from_template(
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
    combine_prompt = PromptTemplate.from_template(
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
