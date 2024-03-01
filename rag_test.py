import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv('ollama_model')
OLLAMA_URL = os.getenv('ollama_url')
QDRANT_URL = os.getenv('qdrant_url')
QDRANT_COLLECTION = "blog_test"

llm = Ollama(model=OLLAMA_MODEL,
             base_url=OLLAMA_URL)

embedding = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL
)

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Qdrant.from_documents(
    documents=splits,
    embedding=embedding,
    url=QDRANT_URL,
    prefer_grpc=True,
    collection_name=QDRANT_COLLECTION,
    force_recreate=True
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

result = rag_chain.invoke("What is Task Decomposition?")
print(result)
