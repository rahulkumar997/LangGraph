import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

# 1. Load environment variables from .env
load_dotenv()

# 2. Define State Schema (must be 'messages', plural)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Initialize LLM (using active Groq model)
llm = ChatGroq(model="openai/gpt-oss-20b")

# 4. Define Chatbot Node Function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 5. Build and Compile Graph
graph_builder = StateGraph(State)
graph_builder.add_node("llmchatbot", chatbot)
graph_builder.add_edge(START, "llmchatbot")
graph_builder.add_edge("llmchatbot", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    user_prompt = "Hi How are you?"
    print(f"User: {user_prompt}\n")

    print("--- Stream Response ---")
    for event in graph.stream({"messages": user_prompt}):
        for value in event.values():
            print("Bot:", value["messages"][-1].content)
