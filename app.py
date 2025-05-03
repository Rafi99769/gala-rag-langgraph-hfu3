import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage
from langchain_openai import AzureChatOpenAI
from retriever import guest_info_tool
from tools import search_tool, weather_info_tool, hub_stats_tool

load_dotenv()

tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o",
    api_version="2025-01-01-preview",
    api_key=os.getenv("OPENAI_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

alfred = builder.compile()

response = alfred.invoke({
    "messages": "I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"
})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
