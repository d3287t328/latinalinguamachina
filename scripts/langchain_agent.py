# export OPENAI_API_KEY=sk-*
# export SERAPI_API_KEY=c
# pip install openai langchain google-search-results

import os
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent

# Get the OpenAI API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Get the SerpAPI API key from the environment variable
serpapi_api_key = os.environ.get("SERPAPI_API_KEY")

search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history")

# Update the OpenAI initialization to use model_kwargs instead of the api_key parameter
llm = OpenAI(temperature=0, model_kwargs={"api_key": openai_api_key})

agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

while True:
    user_input = input("Enter your message: ")
    response = agent_chain.run(input=user_input)
    print("Response:", response)
