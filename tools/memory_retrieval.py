from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from typing import Any
from qdrant_client import QdrantClient
from langchain import hub


class MemoryRetrieval(BaseTool):
    """Tool that retrieves memorized information"""

    rag_chain: Any

    name: str = "memoryretrieval"
    description: str = (
        "recall previously memorized information"
    )

    def __init__(self, llm, embeddings, qdrant_url, qdrant_collection, **kwargs):
        super().__init__(**kwargs)

        client = QdrantClient(
            url=qdrant_url,
            prefer_grpc=True,
        )

        collections = client.get_collections()
        if any(candidate.name == qdrant_collection for candidate in collections.collections):
            print(f"collection {qdrant_collection} exists")
            vectorstore = Qdrant(
                client=client,
                collection_name=qdrant_collection,
                embeddings=embeddings
            )
        else:
            client.close()
            print(f"collection {qdrant_collection} needs to be created")
            vectorstore = Qdrant.from_texts(
                texts=["Initial text"],
                embedding=embeddings,
                url=qdrant_url,
                prefer_grpc=True,
                collection_name=qdrant_collection,
                force_recreate=True
            )
            vectorstore.add_texts([
                "Steve is a nice guy.",
                "Martha sucks.",
                "The idle power consumption of the entertainment system is about 5 wats, any reading above this value indicates that some device was left on.",
                "Xianyun (Chinese: 闲云 Xiányún), also known by her adeptus name Cloud Retainer, is a playable Anemo character in Genshin Impact."
            ])

        retriever = vectorstore.as_retriever()

        # You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        # Question: {question}
        # Context: {context}
        # Answer:
        prompt = hub.pull("rlm/rag-prompt")

        self.rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _run(self, question) -> str:
        """Use the tool"""
        return self.rag_chain.invoke(question)
