import importlib.util
import sys
import os

import openai 
from data_utils import DataFormatter, Embedder

openai.api_key = ''
DIRECTORY = 'sid-playground-data' 


def import_scrape_data_from_directory(directory):
    module_file = os.path.join(directory, 'scrape_data.py')

    # Create a module name based on the directory
    module_name = os.path.basename(directory) + "_module"

    # Load the scrape_data_function 
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Return the scrape_data function from the module
    return getattr(module, 'scrape_data', None)

scrape_data_function = import_scrape_data_from_directory(DIRECTORY)
if scrape_data_function:
    
    data = scrape_data_function()
    chunker, embedder = DataFormatter(), Embedder()
    chunks = chunker(data, token_limit=2, save_file=f"{DIRECTORY}/chunks.pkl")
    embeddings = embedder(chunks, save_file=f"{DIRECTORY}/embeddings.pkl")

