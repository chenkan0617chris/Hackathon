from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
#OpenAI.openai_api_key = os.environ["SERPAPI_API_KEY"]


def init():
    llm = OpenAI(temperature=0)

    tools = load_tools(["llm-math"], llm=llm)

    #agent = initialize_agent(tools, llm
    # , agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    #return "Initialization sucessful"


    
@app.route('/input', methods=['POST']) 
def create_agent():
    csv_files = os.listdir('./csv-dataset')

    for i in range(len(csv_files)):
        csv_files[i] = './csv-dataset/' +  csv_files[i]

    llm = OpenAI(temperature=0)

    # tools = load_tools(["serpapi", "llm-math"], llm=llm)

    prompt = hub.pull("hwchase17/openai-functions-agent")

    # agent = create_csv_agent(llm, csv_files, verbose=False,memory = ConversationBufferMemory(memory_key = 'chat_history'))

    from langchain_openai import ChatOpenAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
    import pandas as pd

    df = pd.read_csv("state.csv")

    loader = CSVLoader(file_path="./state.csv")

    data = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_text(data)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    db = ''.join(Chroma.from_documents(texts, embeddings))
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",
        verbose=True
    )

    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # agent = create_csv_agent(OpenAI(temperature=0), ['state.csv', 'suburb.csv'], verbose=False)

    while True:
        question = input("what do you want to ask?")
        answer = agent.run(question)
        if question == 'exit':
           break
        
         
        
        
        return answer

def start():

    init()

    create_agent()

if __name__ == '__main__':
    app.run(debug=True)


start()

