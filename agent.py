import os
import sys
import asyncio 
from dotenv import load_dotenv

# Add the project root directory to the Python path
# This allows finding modules in the 'tools' directory when running agent.py directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from tools.calculator import add, subtract, multiply, divide # Importing calculator functions
from tools.wiki_search import wiki_search # Importing wiki search tool
from tools.web_search import web_search # Corrected import alias if needed, or use web_search_tool directly
from tools.analyze_csv import analyze_csv 
from tools.analyze_excel import analyze_excel
from tools.download_file import download_file
from tools.extract_text_from_image import extract_text_from_image
from tools.read_file import read_and_save_file
#switch to using gemini 2.0 model 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage 

#use LangGraph to create the agent
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.tools.base import BaseTool


load_dotenv()

tools = [
    add,
    subtract,
    multiply,
    divide,
    wiki_search,
    web_search,
    analyze_csv,
    analyze_excel,
    download_file,
    extract_text_from_image,
    read_and_save_file,
]

def create_agent(): #build graph
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", 
        convert_system_message_to_human=True)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None 
        
    try:
        llm_with_tools = llm.bind_tools(tools)

        def assistant(state: MessagesState):
            """Assistant node"""
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        react_graph = builder.compile()

        print("Agent created successfully.")
        return react_graph
    except Exception as e:
        print(f"Error creating Agent {e}")
        return None


def main(): # Define an async main function
    agent = create_agent()
    if agent:
        print("\nAgent ready. Enter your query (or type 'quit' to exit):")
        while True:
            try:
                query = input("> ") # input() is blocking, consider aioconsole for fully async input if needed
                if query.lower() == 'quit':
                    break
                if query:
                    input_msg = [HumanMessage(content=query)]
                    # Assuming agent.run is the correct async method for FunctionAgent
                    response = agent.invoke({"messages": input_msg})

                    for m in response['messages']:
                        m.pretty_print()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred during chat: {e}")
        print("Exiting agent chat.")
    else:
        print("Agent creation failed.")

#write a simple test here to check if the agent is working as expected
if __name__ == '__main__':
    try:
        main() # Run the async main function
    except KeyboardInterrupt:
        print("\nExiting program.")