import os

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from tools import MemoryRetrieval

load_dotenv()
OLLAMA_MODEL = os.getenv('ollama_model')
OLLAMA_URL = os.getenv('ollama_url')
QDRANT_URL = os.getenv('qdrant_url')
QDRANT_COLLECTION = "rag_test_3"

llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
embedding = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

print(
    MemoryRetrieval(llm, embedding, QDRANT_URL, QDRANT_COLLECTION)
    ._run("What do we know about Martha?")
)