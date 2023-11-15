import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import openai as ai
import re
import nltk
import time
import pickle 


# SECTION 1: Scaping the data.

# Gets the contents page, and creates a scaper for that page.
page = requests.get('https://rajeshjain.com/marketing/').content
soup = BeautifulSoup(page, 'html.parser')

# Gets the desired content, which is a list of names of all the articles by the author.
hrefs = soup.find("div", {"class": "entry-content"}).find_all('a', href=True)
articles = [tag['href'] for tag in hrefs][:52]

# Data cleanup for each article.


def cleanup_content(soup):
    for node in soup.findAll("div")[-5:] + soup.findAll("p")[:1]:
        node.decompose()
    for node in soup(['style', 'script']):
        node.decompose()
    return ' '.join(soup.stripped_strings)


# Given the article names, iterate over all the articles and get the desired content from each.
data = {'title': [], 'content': []}
for c, article in enumerate(articles):
    curr_page = requests.get(article).content
    curr_soup = BeautifulSoup(curr_page, 'html.parser')
    try:
        title = curr_soup.find(
            "header", {"class": "entry-header"}).getText().strip()
        content = curr_soup.find("div", {"class": "entry-content"})
        content = cleanup_content(content)

        data['title'].append(title)
        data['content'].append(content)
    except AttributeError:
        print(curr_soup.getText())


# SECTION 2: Create the tokenized corpus.

# Tokenize the corpus.
def get_tokenized_corpus(data):
    df = pd.DataFrame(data)
    corpus = ""
    for c, row in df.iterrows():
        article = "Title - " + row["title"] + ". " + row["content"] + " "
        corpus += article
    lines = nltk.word_tokenize(corpus)
    return lines

# Split the corpus into blocks (chunks) of size block_size, incrementing by fixed amount each time.


def block_text(lines, token_limit):
    blocks = list()
    start = 0
    end = token_limit
    while end < len(lines):
        while lines[end] != '.':
            end += 1
        blocks.append(' '.join(lines[start: end + 1]))
        start = end + 1
        end += token_limit
    return blocks


# Create and save the corpus.
lines = get_tokenized_corpus(data)
chunks = block_text(lines, 500)
chunks_df = pd.DataFrame(chunks, columns=["content"])
with open('chunks.pkl', 'wb') as file:
    pickle.dump(chunks, file)


# SECTION 3: Create an Embeddings Database.
MODEL_NAME = "curie"
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

# Use openAI's embeddings API to embed a chunk of text.


def get_embedding(text, model):
    try:
        result = ai.Embedding.create(
            model=model,
            input=text
        )

        print("success")
        return result["data"][0]["embedding"]

    except Exception as e:
        print("sleeping")
        time.sleep(60)
        return get_embedding(text, model)


def get_doc_embedding(text, index):
    print(f"chunk {index}: ", end="")
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)


def get_query_embedding(text):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def compute_doc_embeddings(df):
    return {
        idx: get_doc_embedding(r.content.replace("\n", " "), idx) for idx, r in df.iterrows()
    } 


document_embeddings = compute_doc_embeddings(chunks_df)
# TODO: Save the embeddings to csv file.
