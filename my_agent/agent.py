from typing import Optional
from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, should_continue, tool_node
from my_agent.utils.state import AgentState
from pydantic import BaseModel, Field

class GraphConfigPydantic(BaseModel):
    model_name: Optional[str] = Field(
        default="anthropic/claude-3-7-sonnet-latest",
        metadata={
            "x_lg_ui_config": {
                "type": "select",
                "default": "anthropic/claude-3-7-sonnet-latest",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic/claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic/claude-3-5-sonnet-latest",
                    },
                    {"label": "GPT 4o", "value": "openai/gpt-4o"},
                    {"label": "GPT 4.1", "value": "openai/gpt-4.1"},
                    {"label": "o3", "value": "openai/o3"},
                    {"label": "o3 mini", "value": "openai/o3-mini"},
                    {"label": "o4", "value": "openai/o4"},
                ],
            }
        }
    )
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_lg_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        }
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_lg_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        }
    )
    system_prompt: Optional[str] = Field(
        default=None,
        metadata={
            "x_lg_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": "The system prompt to use in all generations",
            }
        }
    )

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfigPydantic)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

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

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
