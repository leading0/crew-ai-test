from crewai import Agent, Task, Crew, Process
from tools import MemoryRetrieval
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv('ollama_model')
OLLAMA_URL = os.getenv('ollama_url')
QDRANT_URL = os.getenv('qdrant_url')
QDRANT_COLLECTION = "ha_test2"

llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
embedding = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
#embedding = HuggingFaceEmbeddings()
memory_retrieval = MemoryRetrieval(llm, embedding, QDRANT_URL, QDRANT_COLLECTION)

butler = Agent(
    role='Butler',
    goal='Assist the house owners by making sure everything about the house is in order',
    backstory="""You are an ai assistant watching over a smart home""",
    verbose=True,
    allow_delegation=True,
    tools=[],
    llm=llm,
)

memory = Agent(
    role='Memory Retriever',
    goal='Retrieve previously memorized facts and information about the house to assist in decision making using your tool',
    backstory="""You are an ai assistant that has access to memorized information""",
    verbose=True,
    allow_delegation=False,
    tools=[memory_retrieval],
    llm=llm,
)

check_power_levels = Task(
    description="""
    The owner of the house is about to go to bed. You are looking at the power consumption readings of the house to determine if there are any devices which are still powered on but should not be.
    The readings are as follows:
    - Entertainment System: 7 wats
    You can ask co-workers to check about expected values for these systems.
    Your reply should state whether all readings are nominal or if there is something that should be checked by the owner.
    """,
    agent=butler
)

crew = Crew(
    agents=[memory],
    tasks=[check_power_levels],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
    manager_llm=llm,  # The manager's LLM that will be used internally
    process=Process.hierarchical  # Designating the hierarchical approach
)
result = crew.kickoff()

print("######################")
print(result)
