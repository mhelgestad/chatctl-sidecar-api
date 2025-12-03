from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from fastapi import FastAPI, Header
from sentence_transformers import SentenceTransformer

from models.requests import AgentInitRequest, EmbeddingRequest, ExplainRequest
from models.response import ExplainResponse

executor = None
llm = None
agent = None

l6_model = SentenceTransformer('all-MiniLM-L6-v2')
l12_model = SentenceTransformer('all-MiniLM-L12-v2')
mpnet_model = SentenceTransformer('all-mpnet-base-v2')

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

def filter_unrelated_query_tool(query: str):
    prompt = """
    Determine if the following query is related to programming, command line, or terminal errors.
    If it is related, return the query as is.
    If it is not related, we will want to explain that this assistant only handles programming and command line errors.
    Query: {query}
    """
    response = llm.invoke({"messages": [{"role": "system", "content": prompt.format(query=query)}]})
    return response.content

filter_unrelated_tool = Tool(
    name="filter_unrelated_query",
    func=filter_unrelated_query_tool,
    description="Filter queries that are unrelated to programming or command line errors",
)

parser = PydanticOutputParser(pydantic_object=ExplainResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful command line assistant that helps explain program and command errors that appear in the terminal, and provides suggestions on how to fix them.
            When given a query, provide a detailed explanation of the error and suggest possible solutions.
            If the query is unrelated to programming or command line errors, explain that you can only help with those topics.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),

    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, filter_unrelated_tool]

app = FastAPI()

def initialize_agent(model: str, token: str):
    global llm, agent, executor
    llm = ChatOpenAI(model=model, api_key=token)
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )
    executor = AgentExecutor(agent=agent, tools=tools)

@app.get("/")
def healthy():
    return {"message": "healthy"}

@app.post("/init")
def agent_init(init_request: AgentInitRequest, authorization: str = Header(None)):
    model = init_request.model or "gpt-5-nano"
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    initialize_agent(model, token)
    return {"message": f"agent initialized with model '{model}'"}

@app.post("/explain")
async def explain(explain_request: ExplainRequest):
    if executor is None:
        return {"error": "Agent not initialized, run \"chatctl initAgent\" first."}
    try:
        raw_response = executor.invoke({"query": explain_request.text})
        structured_response = parser.parse(raw_response.get("output"))
        return structured_response.model_dump(mode="json")
    except Exception as e:
        return {"error": str(e)}

@app.post("/embedding")
async def embedding(embedding_request: EmbeddingRequest):
    model_name = embedding_request.model
    if model_name == "all-MiniLM-L6-v2":
        model = l6_model
    elif model_name == "all-MiniLM-L12-v2":
        model = l12_model
    elif model_name == "all-mpnet-base-v2":
        model = mpnet_model
    else:
        return {"error": f"Model '{model_name}' not supported for embeddings."}
    embedding = model.encode(embedding_request.text).tolist()
    return {"embedding": embedding}
  