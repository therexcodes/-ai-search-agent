import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResultResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]



llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  #LLM
parser= PydanticOutputParser(pydantic_object=ResultResponse)  #PARSER


#PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a helpful assistant that will help generate a research report.
            Answer the user query and use neccesary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

#TOOL
tools = [
    search_tool,
    wiki_tool,
    save_tool
]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
) 

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
) 

query = input("what are you looking for? ")
raw_response = agent_executor.invoke({"query": query })


try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e, "raw response:", raw_response)
