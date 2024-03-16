import os
from langchain.llms import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.prompts import PromptTemplate

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# from langchain.chat_models import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

OpenAI.openai_api_key = os.environ['OPENAI_API_KEY']
OpenAI.openai_api_key = os.environ["SERPAPI_API_KEY"]

def init():
    llm = OpenAI(temperature=0)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    
 
def create_agent():

    # csv_files = os.listdir('./csv-dataset')

    # for i in range(len(csv_files)):
    #     csv_files[i] = './csv-dataset/' +  csv_files[i]

    # agent = create_csv_agent(OpenAI(temperature=0), csv_files, verbose=True)

    agent = create_csv_agent(OpenAI(temperature=0), ['state.csv', 'suburb.csv'], verbose=True)

    while True:
        question = input("what do you want to ask?")

        if question == 'exit':
            break

        agent.run(question)

def start():

    init()

    create_agent()

start()

