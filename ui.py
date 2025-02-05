import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from classify import classify_reviews, summarize_reviews


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
    st.set_page_config(page_title = "Amazon Earbud Reviews Classification", layout = "wide")
    st.title("Amazon Earbud Reviews Classification")

    # Buttons to run or load the classifier.
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run classifier and generate radar chart & summary"):
            with st.spinner("Running classifier..."):
                classify_reviews()
            st.success("Classification complete!")
    with col2:
        if st.button("Load persisted data and generate radar chart & summary"):
            st.info("Using persisted classified reviews.")

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
