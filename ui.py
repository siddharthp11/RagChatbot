from logic import ChatBot
from data_utils import Retrieval
import streamlit as st 
import openai as ai 


if not ChatBot.can_initialize():
    st.title('Retrieval Augmented Generation ChatBot!')
    st.info('This chatbot has access to information about internet marketing.')

    api_key = st.text_input('Enter your API Key')
    ai.api_key = api_key 

else: 
    st.title('Retrieval Augmented Generation ChatBot!')
    st.info('This chatbot has access to information about internet marketing.')
    
    if 'chatbot' not in st.session_state: 
        DIRECTORY = 'sid-playground-data'
        try: 
            retrieval = Retrieval(chunks_file= f'{DIRECTORY}/chunks.pkl', embeddings_file=f'{DIRECTORY}/embeddings.csv')
            st.session_state['chatbot'] = ChatBot(retrieval)
        except:
            st.error('Failed to access vector DB!')

    chatbot =  st.session_state['chatbot']
    input = st.text_input("Enter your question:")
    contexts, response = '', ''

    if input: 
        contexts, response = chatbot.get_response(input)

    st.text_area('Model response:', response)

    if contexts:
        for c, ctx in enumerate(contexts): 
            st.text_area(f"Retrieved context #{c}", ctx)
