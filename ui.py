import pandas as pd
import streamlit as st
from streamlit.logger import get_logger
from classify import classify_reviews


def generate_radar_chart():
    st.write("Generating radar chart")
    # check if classified_reviews.csv file exists
    try:
        op_df = pd.read_csv('classified_reviews.csv')
        # generate radar chart
    except FileNotFoundError:
        st.write("classified_reviews.csv not found. Please run the classifier first")
        return
    

def handle_classification():
    classify_reviews()
    generate_radar_chart()

def run():
    # uncomment below line to run the classifier. use the already generated classified_reviews.csv to generate the chart 
    # classify_reviews()
    st.set_page_config(
        page_title="Amazon Earbud reviews classification",
    )

    st.write("# Amazon Earbud reviews classification! 👋")

    st.button(
        label="Geneerate radar chart and summary using persisted data",
        on_click=generate_radar_chart,
        type="primary"
    )

    st.button(
        label="Run classifier and generate radar chart and summary",
        on_click=handle_classification
    )


if __name__ == "__main__":
    run()
