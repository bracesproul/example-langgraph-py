from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

@lru_cache(maxsize=4)
def _get_model(model_spec: str, temperature: float, max_tokens: int):
    if model_spec.startswith("openai/"):
        model_name = model_spec.split("/", 1)[1]
        if model_name.startswith("o"):
            # Don't pass temperature for openai 'o' models
            model = ChatOpenAI(model_name=model_name, max_tokens=max_tokens)
        else:
            model = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)
    elif model_spec.startswith("anthropic/"):
        model_name = model_spec.split("/", 1)[1]
        model = ChatAnthropic(temperature=temperature, model_name=model_name, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported model spec: {model_spec}. Expected format 'openai/<model_name>' or 'anthropic/<model_name>'")

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

# Define the function that calls the model
def call_model(state, config):
    model_name = config.get('configurable', {}).get("model_name", "anthropic/claude-3-7-sonnet-latest")
    temperature = config.get('configurable', {}).get("temperature", 0.7)
    max_tokens = config.get('configurable', {}).get("max_tokens", 4000)
    system_prompt = config.get('configurable', {}).get("system_prompt", "Be a helpful assistant")

    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages

    model = _get_model(model_name, temperature, max_tokens)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)
