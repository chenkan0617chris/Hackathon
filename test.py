# import os
# import openai
# import sys
# sys.path.append('../..')

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# from langchain_community.document_loaders.csv_loader import CSVLoader

# from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain_openai import ChatOpenAI, OpenAI

# persist_directory = 'docs/chroma/'
# embedding = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# openai.api_key  = os.environ['OPENAI_API_KEY']


# loader = CSVLoader(file_path='./customer.csv')
# data = loader.load()

# agent = create_csv_agent(
#     ChatOpenAI(temperature=0, openai_api_key=openai.api_key, model="gpt-3.5-turbo-0613"),
#     "customer.csv",
#     verbose=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
# )

# agent.run("how many rows are there?")

# print(data)

arr = ['a', 'b', 'c', 'd', 'e', 'f']

for i in range(len(arr)):
    arr[i] += 'x'

print(arr)