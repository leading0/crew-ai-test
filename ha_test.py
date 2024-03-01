import os

from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_community.llms import Ollama

from tools import WeatherReport

load_dotenv()
OLLAMA_MODEL = os.getenv('ollama_model')
OLLAMA_URL = os.getenv('ollama_url')

ollama_llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

weather_report = WeatherReport()

# Define your agents with roles and goals
researcher = Agent(
    role='Data Acquisition',
    goal='Gather data from different sources',
    backstory="""Your job is to provide and interpret data from different sources.""",
    verbose=True,
    allow_delegation=False,
    tools=[weather_report],
    # You can pass an optional llm attribute specifying what mode you wanna use.
    # It can be a local model through Ollama / LM Studio or a remote
    # model like OpenAI, Mistral, Antrophic or others (https://python.langchain.com/docs/integrations/llms/)
    #
    # Examples:
    #
    # from langchain_community.llms import Ollama
    llm=ollama_llm  # was defined above in the file
    #
    # from langchain_openai import ChatOpenAI
    # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
)
writer = Agent(
    role='Presenter',
    goal='Compile information for consumption by humans',
    backstory="""You create a report from all the information that have been passed to you""",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# Create tasks for your agents
task1 = Task(
    description="""Gather information about the current and forecasted weather """,
    agent=researcher
)

task2 = Task(
    description="""Write a short report about the current and future weather""",
    agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
