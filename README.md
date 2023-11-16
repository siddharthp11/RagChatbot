# RAG Pipline to automatically create, and talk to a vector database from data. 
Built as a demo for the NYC NetCore Cloud office's content creators.

## To create a vector DB:-
- create a directory in the root folder. 
- within this directory create `scrape_data.py`
- within this module, write a function called `scrape_data` which formats and returns your data as a dict like: 
  - {'title': [str], 'content': [str]}
- now, in the root directory open `create_vector_database`.
  -  specify your directory name in `DIRECTORY`
  -  enter your OpenAI API key in `openai.api_key`
- run `create_vector_database`. Since you are using openAI's API, you'll be charged for this! 
- now, in *your* directory, you will see two folders: 
  - 1. `chunks.pkl` are the chunks of text from your corpus. 
  - 2. `embeddings.csv` is the embeddings database. 

## To interface with your data using the vector DB and an openai model:-
- In `ui.py` specify the {DIRECTORY}.
- Run `streamlit run ui.py`
- Enter your API Key. 
- You should be able to use the chatbot and view the retrieval context that was provided to it. 


## Contribute:-
- ### OpenAI API flexibility- 
  - User should be able to specify the number of retrieved articles with cost constraints. 
  - Users should be able to see cost estimates. 
  - Users should be able to choose an openai model. 
- ### Reading the data files- 
  - Users should be able to specify the data path on the UI. 
- ### Creating the data files- 
  - Users should be able to create the data files using some UI, where they can write the scrape data function. 
