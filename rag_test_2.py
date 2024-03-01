import os

from dotenv import load_dotenv
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

load_dotenv()
OLLAMA_MODEL = os.getenv('ollama_model')
OLLAMA_URL = os.getenv('ollama_url')
QDRANT_URL = os.getenv('qdrant_url')
QDRANT_COLLECTION = "rag_test_2"

llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL)

embedding = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL
)

vectorstore = Qdrant(
    client=QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
    ),
    collection_name=QDRANT_COLLECTION,
    embeddings=embedding
)

retriever = vectorstore.as_retriever()

# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

result = rag_chain.invoke("What do we know about the power consumption of the entertainment system?")
print(result)
