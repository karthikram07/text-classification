import pandas as pd
import streamlit as st
from streamlit.logger import get_logger
from classify import classify_reviews, summarize_reviews


class OpenAITextClassifier():
    def __init__(self):
        self.is_classified = False
        self.generate_clicked = False

    def generate_radar_chart(self):
        self.generate_clicked = True
        # check if classified_reviews.csv file exists
        try:
            op_df = pd.read_csv('classified_reviews.csv')
            st.dataframe(op_df, use_container_width=True)

            # generate radar chart
        except FileNotFoundError:
            st.write("classified_reviews.csv not found. Please run the classifier first")
            return
        

    def handle_classification(self):
        classify_reviews()
        self.is_classified = True
        self.generate_radar_chart()

    def get_summaries(self, brand):
        docs = summarize_reviews(brand)
        st.write(docs)


    def run(self):
        st.set_page_config(
            page_title="Amazon Earbud reviews classification",
        )

        st.write("# Amazon Earbud reviews classification! 👋")

        print('self.is_classified', self.is_classified)
        print('self.generate_clicked', self.generate_clicked)


        # if not self.is_classified and not self.generate_clicked:
        #     st.button(
        #         label="Generate radar chart and summary using persisted data",
        #         on_click=self.generate_radar_chart,
        #         type="primary"
        #     )

        #     st.button(
        #         label="Run classifier and generate radar chart and summary",
        #         on_click=self.handle_classification
        #     )

        # elif self.is_classified:
        #     st.write('Classification complete. Generating radar chart')

        # elif not self.is_classified and self.generate_clicked:
        #     st.write('Generating radar chart and summary')
        

        st.write("# Generate product review summary!")
        option = st.selectbox(
                label='Please select the brand',
                options=('sony', 'boat', 'samsung', 'bose', 'oneplus'),
                index=None)
        if option:
            self.get_summaries(option)
        



if __name__ == "__main__":
    classifier = OpenAITextClassifier()
    classifier.run()
