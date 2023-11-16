import pandas as pd
import numpy as np
import openai as ai
import csv 
import nltk
import pickle 

class DataFormatter:
    '''    
    Formatter to 1) tokenize, 2) chunk and optionally 3) save the corpus.
    Input: dict like {title: [str], content: [str]}
    '''
    def __call__(self, data, token_limit=500, save_file=''):
        data = self.get_tokenized_corpus(data)
        chunks = self.chunk_text(data, token_limit)
        if save_file: self.save_chunks_pkl(chunks, save_file)
        
        return chunks

    def get_tokenized_corpus(self, data):
        df = pd.DataFrame(data)
        corpus = ""
        for c, row in df.iterrows():
            article = "Title - " + row["title"] + ". " + row["content"] + " "
            corpus += article
        lines = nltk.word_tokenize(corpus)
        return lines

    # Split the corpus into blocks (chunks) of size block_size, incrementing by fixed amount each time.
    def chunk_text(self, lines, token_limit):
        chunks = list()
        start = 0
        end = token_limit
        while end < len(lines):
            while lines[end] != '.':
                end += 1
            chunks.append(' '.join(lines[start: end + 1]))
            start = end + 1
            end += token_limit
        return chunks
    
    def save_chunks_pkl(self, chunks, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(chunks, file)




class Embedder: 
    '''    
    Wrapper around openAI embeddings API, to embed and optionally save the data. 
    Input: pd.DataFrame with 'content' column. 
    '''
    def __init__(self, model="curie", debug_flags=False):
        self.model = model
        self.doc_embedding_model = f"text-search-{model}-doc-001"
        self.query_embedding_model = f"text-search-{model}-query-001"
        self.debug_flags = debug_flags


    def __call__(self, chunks_df, save_file=''):
        if not isinstance(chunks_df, pd.DataFrame):
            chunks_df = pd.DataFrame(chunks_df, columns=["content"])

        embeddings = {
            idx: self.get_doc_embedding(r.content.replace("\n", " "), idx) for idx, r in chunks_df.iterrows()
        } 

        if save_file: self.save_embeddings_csv(embeddings, save_file)

        return embeddings
    

    def get_embedding(self, text, model):
        try:
            result = ai.Embedding.create(
                model=model,
                input=text
            )
            if self.debug_flags: 
                print("Embedded string successfully")

            return result["data"][0]["embedding"]

        except Exception as e:
            print("Error while creating embedding: ", e)


    def get_doc_embedding(self, text, index):
        if self.debug_flags: 
                print(f"chunk {index}: ", end="")
        return self.get_embedding(text, self.doc_embedding_model)


    def get_query_embedding(self, text):
        return self.get_embedding(text, self.query_embedding_model)
    
    def save_embeddings_csv(self, embeddings, filepath):
        pd.DataFrame(embeddings).to_csv(filepath)




class Retrieval:
    ''' 
    Wrapper around openai Embeddings. 
    Input: Takes in path to text chunks and embeddings file. 
    Call: takes a question and returns context. 
    '''
    def __init__(self, chunks_file, embeddings_file) -> None:
        with open(chunks_file, 'rb') as f:
            self.documents = pickle.load(f)

        self.document_embeddings = pd.read_csv(embeddings_file).drop('Unnamed: 0', axis=1) 
    
    def __call__(self, question):
        try: 
            info = self.order_document_sections_by_query_similarity(question)[:5]
            contexts = [self.documents[int(chunk[1])] for chunk in info ]

            return contexts
        except Exception as e: 
            return []
        
    # Similarity function to find similarity score between a chunk and a query.
    def vector_similarity(self, x, y):
        return np.dot(np.array(x), np.array(y))
    
    def get_query_embedding(self, text, model ="text-search-curie-query-001"):
        try:
            result = ai.Embedding.create(model=model, input=text)
            return result["data"][0]["embedding"]

        except Exception as e:
            assert Exception("Could not embed the query.")


    # Find relevant chunks for a query.
    def order_document_sections_by_query_similarity(self, query):
        try:
            query_embedding = self.get_query_embedding(query)
            document_similarities = sorted([
                (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in self.document_embeddings.items()
            ], reverse=True)

            return document_similarities
        except Exception as e: 
            return e 
        