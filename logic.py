import numpy as np
import pandas as pd
import openai as ai
import pickle
import time 

class Retrieval:
    def __init__(self) -> None:
        with open('chunks.pkl', 'rb') as f:
            self.documents = pickle.load(f)

        self.document_embeddings = pd.read_csv("embeddings.csv").drop('Unnamed: 0', axis=1) 
            
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

    def get_context(self, question):
        try: 
            info = self.order_document_sections_by_query_similarity(question)[:5]
            contexts = [self.documents[int(chunk[1])] for chunk in info ]

            return contexts
        except Exception as e: 
            return []

class ChatBot:
    def __init__(self,  model='curie'):
        self.model = model
        self.retrieval = Retrieval()

    def can_initialize(): 
        try:
            ai.Model.list()
        except ai.error.AuthenticationError as e:
            return False
        else:
            return True

    def get_response(self, question):
        contexts = self.retrieval.get_context(question)
        context_as_string = ' '.join(contexts)
        prompt = """Answer the question as truthfully as possible using the provided text, if the answer is not contained in the text, say ' I don't know'."

        Context: 
        {ctx}

        Q: {question}
        A:""".format(ctx=context_as_string, question=question)

        try: 
            response = ai.Completion.create(
                prompt=prompt,
                temperature=0,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model="text-davinci-002"
            )["choices"][0]["text"].strip(" \n")

            return contexts, response
        except Exception as e: 
            return 'No context found', 'No response'

