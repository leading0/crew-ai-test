from crewai import Agent, Task, Crew, Process
from tools import LightStatus, LightSwitch
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv('ollama_model')
OLLAMA_URL = os.getenv('ollama_url')

llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

lighting = Agent(
    role='Lighting Manager',
    goal='Assist the house owners by managing light sources',
    backstory="""You are an ai assistant watching over a smart home, you know about the light sources in the house and can turn them on and off""",
    verbose=True,
    allow_delegation=True,
    tools=[LightStatus(), LightSwitch()],
    llm=llm,
)

going_to_bed = Task(
    description="""
    The owner of the house is going to sleep. 
    Check the state of the house and do what you think needs to be done in the given situation.
    Delegate any tasks to your co-workers.
    """,
    agent=lighting
)

crew = Crew(
    agents=[lighting],
    tasks=[going_to_bed],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
    manager_llm=llm,  # The manager's LLM that will be used internally
    process=Process.hierarchical  # Designating the hierarchical approach
)
result = crew.kickoff()

print("######################")
print(result)
