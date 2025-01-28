import os
from IPython.display import Image, display
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
import requests

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
import functools
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

os.environ['OPENAI_API_KEY'] = ""
os.environ['TAVILY_API_KEY'] = ""

def create_agent_interpreter(llm, tools, system_message: str):
    """Crea el agente intérprete."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an interpreter AI, initiating the analysis of the environment. "
                "Use the available tools to retrieve and send the package containing the prompt, which contains observations, "
                "inventory, and valid actions to the decision-making AIs. Relay the results back and continue until 'is_completed' is received as True. "
                "If 'is_completed' is True then answer with FINISHED as a prefix to your answer."
                "You have access to the following tools: {tool_names}.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def create_agent_decider_no_tools(llm, system_message: str):
    """Create a decision-making agent without tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a decision-making AI tasked with analyzing actions based on the environment's current state, observation, valid actions "
                "and inventory. You will recieve a package with a prompt containing all of that."
                "Discuss with the other decision-making AI until a consensus is reached on the best action to take. "
                "Once decided, send only your choiced valid_action back to the interpreter AI using INTERPRETER as a prefix to your answer.\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm  # No need to bind tools
# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

@tool
def init_env(self):
    """La primera llamada que hace el modelo es para iniciar el entorno y recibir la prompt inicial"""
    url = "http://localhost:5000/init"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            prompt_data = response.get("prompt")
            is_completed = response.get("is_completed")
            return prompt_data, is_completed
        else:
            print(f"Error en la llamada al servidor: {response.status_code}")
            return None, False
    except requests.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")
@tool
def step(self, action):
    """Hace la llamada a localhost:5000/step con la acción tomada."""
    
    url = "http://localhost:5000/step"

    try:
        response = requests.get(url,json=action)
        if response.status_code == 200:
            # Procesamos la respuesta de cada paso
            prompt_data = response.get("prompt")
            is_completed = response.get("is_completed")
            #print(f"Respuesta después del paso: {prompt_data}")
            return prompt_data, is_completed
        else:
            print(f"Error en la llamada al servidor_ {response.status_code}")
            return None, False
    except requests.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


llm = ChatOpenAI(model="gpt-4o")
# Research agent and node
interpreter_agent = create_agent_interpreter(
    llm,
    [init_env, step],
    system_message="You are an interpreter, you should use the provided tool to get the environment's status and goals.",
)
interpreter_node = functools.partial(agent_node, agent=interpreter_agent, name="Interpreter")

# chart_generator
decider1_agent = create_agent_decider_no_tools(
    llm,
    system_message="Help determine the best action by discussing with your counterpart.",
)
decider1_node = functools.partial(agent_node, agent=decider1_agent, name="Decider1")


decider2_agent = create_agent_decider_no_tools(
    llm,
    system_message="Help determine the best action by discussing with your counterpart.",
)
decider2_node = functools.partial(agent_node, agent=decider2_agent, name="Decider2")

tools = [init_env, step]
tool_node = ToolNode(tools)



def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINISHED" in last_message.content:
        # Any agent decided the work is done
        return END
    if "INTERPRETER" in last_message.content:
        return "interpreter"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("Interpreter", interpreter_node)
workflow.add_node("Decider1", decider1_node)
workflow.add_node("Decider2", decider2_node)
workflow.add_node("Env_Interact", tool_node)

workflow.add_conditional_edges(
    "Interpreter",
    router,
    {"continue": "Decider1", "call_tool": "Env_Interact", END: END},
)
workflow.add_conditional_edges(
    "Decider1",
    router,
    {"continue": "Decider2", "interpreter": "Interpreter", END: END},
)
workflow.add_conditional_edges(
    "Decider2",
    router,
    {"continue": "Decider1", "interpreter": "Interpreter", END: END},
)
workflow.add_conditional_edges(
    "Env_Interact",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Interpreter": "Interpreter",
    },
)
workflow.add_edge(START, "Interpreter")
graph = workflow.compile()
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Only on the first call you should fetch info from localhost:5000/init"
                "You should use that information and pass it to your friends so they can tell you which one is the best action"
                "Then you should take the best action and send it back to localhost:5000/step"
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 10},
)
# Iterar sobre los eventos y formatear la salida
#for event in events:
#    formatted_event = format_event(event)
#    print(formatted_event)
#    print("==== End of Event ====\n")
for s in events:
    print(s)
    print("----")