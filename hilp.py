import os
import uuid
import json
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
#from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import Tool
from langchain_core.messages import ToolMessage
# from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv(overide=True)

#define model (gpt-4o) - try different models
# Initialize LLM with the handler
model = AzureChatOpenAI(
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version = os.getenv('AZURE_OPENAI_API_VERSION'),
    api_key = os.getenv('AZURE_OPENAI_API_KEY'),
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    callbacks=[human_intervention_handler]
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

## define two tools- internet search and simply add
@tool
def add(x,y):
    "adding two numbers"
    return x+y

tools = [TavilySearchResults(max_results=1), add]

#tool_executor = ToolExecutor(tools)
model = model.bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def call_tools(state):
    """Simple sequential tool calling helper."""
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = tool_call.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        # We use the response to create a ToolMessage
        tool_message = ToolMessage(
            content=tool_call["response"], name=tool_call["name"], tool_call_id=tool_call["id"]
        )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}



# Define a new graph
workflow = StateGraph(State)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# Helper function to construct message asking for verification
def generate_verification_message(message: AIMessage) -> None:
    """Generate "verification message" from message with tool calls."""
    serialized_tool_calls = json.dumps(
        message.tool_calls,
        indent=2,
    )
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n\n"
            "Type 'y' if you do, anything else to stop.\n\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )

# Helper function to stream output from the graph
def stream_app_catch_tool_calls(inputs, thread) -> Optional[AIMessage]:
    """Stream app, catching tool calls."""
    tool_call_message = None
    for event in app.stream(inputs, thread, stream_mode="values"):
        message = event["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
        else:
            #print(message)
            message.pretty_print()
            if isinstance(message, AIMessage):
                st.write(message.content)

    return tool_call_message

st.title('Human in The Loop - Agent')

user_input = st.text_input("Enter your question:", key="input1")
#if st.button("Submit Question"):

if user_input:
    thread = {"configurable": {"thread_id": "5"}}
    #inputs = [HumanMessage(content="what's the weather in sf now?")]

    inputs = [HumanMessage(content=user_input)]
    # for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    #     event["messages"][-1].pretty_print()

    tool_call_message = stream_app_catch_tool_calls(
        {"messages": inputs},
        thread,
    )

    # tool name:
    tool_name=tool_call_message.tool_calls[-1]['name']
    #st.write(tool_call_message.tool_calls[-1])
    st.write(f":blue[tool invoked]: {tool_name} ")

    st.write(":green[Please approve the tool picked up by the agent - select either 'yes' or 'no' ]")

    verification_message = generate_verification_message(tool_call_message)
    #verification_message.pretty_print()

    #st.write(verification_message)

    #human_input=input("Please provide your response")
    human_input = st.text_input('Please provide your response', key='keyname')
    if human_input:

        input_message = HumanMessage(human_input)
        # if input_message.content == "exit":
        #     break

        #st.write(input_message)
        #input_message.pretty_print()

        # First we update the state with the verification message and the input message.
        # note that `generate_verification_message` sets the message ID to be the same
        # as the ID from the original tool call message. Updating the state with this
        # message will overwrite the previous tool call.
        snapshot = app.get_state(thread)
        snapshot.values["messages"] += [verification_message, input_message]

        if input_message.content == "yes":
            tool_call_message.id = str(uuid.uuid4())
            # If verified, we append the tool call message to the state
            # and resume execution.
            snapshot.values["messages"] += [tool_call_message]
            app.update_state(thread, snapshot.values, as_node="agent")
            tool_call_message = stream_app_catch_tool_calls(None, thread)
        else:
            # Otherwise, resume execution from the input message.
            app.update_state(thread, snapshot.values, as_node="__start__")
            tool_call_message = stream_app_catch_tool_calls(None, thread)