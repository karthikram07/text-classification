# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

@retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(5),reraise=True)
def completion_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)



def get_df():
    file_paths = ['data/B09PC695Q9.csv', 'data/B09N3XMZ5F.csv', 'data/B09CKF166Y.csv', 'data/B08C4KWM9T.csv', 'data/B0BYJ6ZMTS.csv']
    # read all csvs into a dataframe

    sony_buds_df = pd.read_csv(file_paths[0])
    sony_buds_df['brand'] = 'sony'

    boat_buds_df = pd.read_csv(file_paths[1])
    boat_buds_df['brand'] = 'boat'


    samsung_buds_df = pd.read_csv(file_paths[2])
    samsung_buds_df['brand'] = 'samsung'

    bose_buds_df = pd.read_csv(file_paths[3])
    bose_buds_df['brand'] = 'bose'

    oneplus_buds_df = pd.read_csv(file_paths[4])
    oneplus_buds_df['brand'] = 'oneplus'

    # limiting to 100 in interest of time. uncomment as necessary

    sony_buds_df = sony_buds_df.head(100)
    boat_buds_df = boat_buds_df.head(100)
    samsung_buds_df = samsung_buds_df.head(100)
    bose_buds_df = bose_buds_df.head(100)
    oneplus_buds_df = oneplus_buds_df.head(100)

    combined_df = pd.concat([sony_buds_df,boat_buds_df, samsung_buds_df, bose_buds_df, oneplus_buds_df], ignore_index=True)
    

    combined_df.to_csv('data/combined.csv', index=False)

    return combined_df

def get_attributes():
    attributes = ['build quality', 'price', 'comfort', 'design', 'battery life', 'sound quality']
    return attributes
    

def score_reviews(reviews_df):
    op_structure = "feature:score"
    prompt = ChatPromptTemplate.from_template(
                            """
                            Here is a product review: {review}
                            Determine how the reviewer rates this product in relation to these features:  
                            %s
                            Only provide a score (between 0 to 3). If the feature is not mentioned, provide a score of 0. Use the following format. No additional commentary.
                            %s
                            """ % (', '.join(get_attributes()), op_structure)
                        )
    output_parser = StrOutputParser()
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=60)
    chain = (
            {"review": RunnablePassthrough()} 
            | prompt
            | model
            | output_parser
    )
    review_texts = reviews_df['text'].to_list()
    # restrict all reviews to 12000 characters
    review_texts = [text[:12000] for text in review_texts]
    res = chain.batch(review_texts)
    for i, row in reviews_df.iterrows():
        reviews_df.at[i, 'scores'] = res[i]

    return reviews_df
    




def classify_reviews():
    reviews_df = get_df()
    scores = score_reviews(reviews_df)

    scores.to_csv('classified_reviews.csv', index=False)






        



