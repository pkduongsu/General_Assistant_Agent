import os
import sys
from typing import List, TypedDict, Annotated, Optional
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
from tools.analyze_image import analyze_image
from tools.analyze_audio import analyze_audio
from tools.analyze_youtube import answer_question_about_youtube_video # Importing YouTube analysis toolS
#switch to using gemini 2.0 model 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition



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
    analyze_image,
    analyze_audio,
    answer_question_about_youtube_video,]

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system = f.read()

system_message = SystemMessage(content=system)

class AgentState(TypedDict):
    input_file: Optional[str] #contains the input file path if there is any
    messages: Annotated[List[AnyMessage], add_messages] #contains the messages exchanged between the user and the agent


def create_agent(): #build graph
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
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
                    # Assuming agent.run is the correct async method for FunctionAgent
                    # Construct the initial messages list including the system prompt
                    initial_messages = [
                        system_message, # Include the system prompt read earlier
                        HumanMessage(content=query)
                    ]
                    # Invoke the agent with the messages state
                    response = agent.invoke({"messages": initial_messages})

                    # The final response from the graph is in the 'messages' list
                    # Get the last message, which should be the AI's response
                    answer = response["messages"][-1].content
                    # Print only the final answer without the "Agent: " prefix
                    print(answer)
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