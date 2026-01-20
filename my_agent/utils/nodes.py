from functools import lru_cache
#from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    """
    Restituisce il modello configurato.
    NOTA: Modificato per forzare l'uso di OpenAI GPT-4o in ogni caso,
    risolvendo l'errore 404 di Anthropic.
    """
    
    # Indipendentemente dall'input (model_name), usiamo GPT-4o
    # Se hai la chiave OPENAI_API_KEY settata su LangSmith, questo funzionerà.
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # Se volessi ripristinare la logica in futuro, il codice era:
    # if model_name == "openai":
    #     model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    # elif model_name == "anthropic":
    #     model = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)
