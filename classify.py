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
import re
from collections import Counter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter

from langchain.docstore.document import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed,
    wait_random
)
from langchain_text_splitters import RecursiveJsonSplitter



openai.api_key = st.secrets["OPENAI_API_KEY"]

backoff_times = {
    "classification" :{
        "wait": wait_random_exponential(0.15, 0.35),
        "stop": stop_after_attempt(20)
    },
    "attribute_extraction" :{
        "wait": wait_random(0.15, 0.20),
        "stop": stop_after_attempt(5)
    }
}




def get_df():
    speaker_reviews_df = pd.read_csv('data/combined_speaker_reviews.csv')
    return speaker_reviews_df

@retry(wait=backoff_times["classification"]["wait"], stop=stop_after_attempt(5),reraise=True)
def run_classification_with_backoff(chain, **kwargs):
    return chain.batch(**kwargs)

@retry(wait=backoff_times["attribute_extraction"]["wait"], stop=stop_after_attempt(5),reraise=True)
def run_attribute_extraction_with_backoff(chain, **kwargs):
    return chain.batch(**kwargs)

def filter_special_characters(attribute):
    return re.sub(r'[^a-zA-Z\s]', '', attribute).strip()
    
def get_attributes(reviews_df):
    # pick only 500 reviews per asin
    reviews_df = reviews_df.groupby('asin').head(500)
    prompt = ChatPromptTemplate.from_template(
                            """
                            Here is a product review of a bluetooth speaker: {review}
                            
                            Extract the features that the reviewer mentions that describes the product.
                             
                            For example: 'sound quality', 'price'.
                            
                            Only provide product features that are mentioned in the review. Do not return adjectives or other descriptive verbs. Do not return entire sentences. Return only nouns.
                            
                            Do not return user emotions such as 'love it', 'hate it', 'very useful' etc.
                            
                            Make sure that each feature is at most 2 words long. Don't return features longer than this.
                            
                            Make sure that the features returned make sense as an attribute of a bluetooth speaker. For example, 'kitchen use' is not a valid feature. 'ease of connection' is not a valid feature. 'sound quality' is a valid feature. 'connectivity' is a valid feature.
                           
                            Do not return duplicates. For Eg: if the review mentions a feature in relation to multiple scenarios, return just the characteristic. If the speaker talks about connectivity with TV and phone,  do not return both of them. Just return "connectivity". 
                           
                            Disregard the other products in all reviews. 
                            
                            Return maximum 5 features per review. If there are more than 5 features, return the top 5 features.
                            
                            """
                        )
    output_parser = StrOutputParser()
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=45, top_p=1)
    chain = (
            {"review": RunnablePassthrough()} 
            | prompt
            | model
            | output_parser
    )
    review_texts = reviews_df['reviewText'].to_list()
    review_texts = [str(text) for text in review_texts]
    # restrict all reviews to 12000 characters
    review_texts = [text[:12000] for text in review_texts]
    attributes = run_attribute_extraction_with_backoff(chain, inputs = review_texts, config={"max_concurrency":10})
    print('attributes:', attributes)

    #  return unique attributes
    attributes = list(set(attributes))
    list_of_attr = []
    for attr in attributes:
        attr = filter_special_characters(attr)
        attrs = attr.split('\n')
        list_of_attr.extend([at.strip().lower() for at in attrs])

    attribute_counts = Counter(list_of_attr)
    top_5_attributes = [attr for attr, _ in attribute_counts.most_common(5)]
    print(f"Attributes generated: {list_of_attr}")
    print("\n\n")
    print(top_5_attributes)

    return top_5_attributes
    

def score_reviews(reviews_df, attributes):
    op_structure = "feature:score"
    prompt = ChatPromptTemplate.from_template(
                            """
                            Here is a product review: {review}
                            Determine how the reviewer rates this product in relation to these features:  
                            %s
                            Only provide a score (between 0 to 3). If the feature is not mentioned, provide a score of 0. Use the following format. No additional commentary.
                            %s
                            """ % (', '.join(attributes), op_structure)
                        )
    output_parser = StrOutputParser()
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=45)
    chain = (
            {"review": RunnablePassthrough()} 
            | prompt
            | model
            | output_parser
    )
    review_texts = reviews_df['reviewText'].to_list()
    # make sure all reviews are strings
    review_texts = [str(text) for text in review_texts]
    review_texts = [text[:12000] for text in review_texts]
    res = run_classification_with_backoff(chain, inputs = review_texts, config={"max_concurrency":10})
    for i, row in reviews_df.iterrows():
        reviews_df.at[i, 'scores'] = res[i]

    return reviews_df


def generate_summary_input(df):
    each_label_text = []
    for index, row in df.iterrows():
        # each_label_text += f"review: {row['reviewText']}, scores: {row['scores']}\n"
        each_label_text.append({
            "review": row['reviewText'],
            "scores": row['scores']
        })

    return each_label_text

@retry(wait=wait_random_exponential(0.25, 1), stop=stop_after_attempt(5),reraise=True)
def run_summarization_with_backoff(chain, **kwargs):
    return chain.run(**kwargs)


def _summarize_reviews(product):
    op_df = pd.read_csv('classified_reviews.csv')
    op_df = op_df[op_df['productName'] == product]
    data = generate_summary_input(op_df)
    splitter = RecursiveJsonSplitter()
    docs = splitter.create_documents(texts = data)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt_template = """
                    You are provided with a dataset containing amazon product reviews for bluetooth speakers.
                    The format of the review is:
                    'review':'review_text', 'scores': 'attribute1': 5, 'attribute2': 4, 'attribute3': 3, 'attribute4': 2, 'attribute5': 1
                    Each review is labeled with a score out of 5 for attributes such as Build Quality, Price, Comfort, Design, Battery life, Sound Quality etc.
                    Below is the review:
                    {docs}
                    Your task is to generate a concise summary of the reviews
                    summarise reviews from multiple customers and provide a concise summary of the reviews
                    Ensure that the generated summaries are coherent, relevant, and accurately represent the content of the original reviews.
                    CONCISE SUMMARY:
                    """
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "with some more context below.\n"
        "------------\n"
        "{docs}\n"
        "------------\n"
        "Given the new context, refine the original summary"
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=model,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        input_key="docs",
        output_key="output_text",
        document_variable_name="docs",
    )
    result = chain({"docs": docs}, return_only_outputs=True)
    return result['output_text']

def summarize_reviews(product):
    op_df = pd.read_csv('classified_reviews.csv')
    op_df = op_df[op_df['productName'] == product]
    data = generate_summary_input(op_df)
    splitter = RecursiveJsonSplitter()
    docs = splitter.create_documents(texts = data)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    map_prompt_template = """You are provided with a dataset containing amazon product reviews for bluetooth speakers.
                    The format of the review is:
                    "{'review':'review_text', 'scores': {'attribute1': 5, 'attribute2': 4, 'attribute3': 3, 'attribute4': 2, 'attribute5': 1}}"
                    Each review is labeled with a score out of 5 for attributes such as Build Quality, Price, Comfort, Design, Battery life, Sound Quality etc.
                    Below is the review:
                    {docs}
                    Your task is to generate a concise summary of the reviews
                    Ensure that the generated summaries are coherent, relevant, and accurately represent the content of the original reviews.
                    Limit the summary to a maximum of 20000 characters
                    Summary:"""
    map_prompt = PromptTemplate.from_template(map_prompt_template)
    map_chain = LLMChain(llm=model, prompt = map_prompt)

    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary.
    Concise summary:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    reduce_chain = LLMChain(llm = reduce_model  , prompt = reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    # document_prompt = PromptTemplate.from_template(
    #     input_variables = ["page_content"],
    #     template = "{page_content}"
    # )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain = reduce_chain, document_variable_name = "docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain = combine_documents_chain,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain = map_chain,
        # Reduce chain
        reduce_documents_chain = reduce_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name = "docs",
    )

    # text_splitter = TokenTextSplitter(chunk_size=4000)
    # split_docs = text_splitter.split_documents(Document(page_content=data), max_concurrency = 10)

    result = map_reduce_chain.run(docs)
    print('result:', result)
    return result['output_text']

def classify_reviews():
    reviews_df = get_df()
    attributes = get_attributes(reviews_df)
    scores = score_reviews(reviews_df, attributes)

    scores.to_csv('classified_reviews.csv', index=False)






        



