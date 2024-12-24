# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

from pydantic import BaseModel, Field
from typing import Optional, Callable, Union, Set, List


class Agent(BaseModel):
    agent_id: str = Field(..., description="The unique identifier for the agent")
    role: str = Field(..., description="The role or type of agent performing tasks")
    goal: str = Field(..., description="The objective or purpose of the agent")
    attributes: Optional[str] = Field(
        None, description="Additional attributes or characteristics of the agent"
    )
    llm: Optional[Union[Callable, List[Callable]]] = Field(
        None,
        description="The language model function(s) to be used by the agent. Can be a single function or a list of functions",
    )
    tools: Optional[Set[Callable]] = Field(
        default=None,
        description="Optional set of tool functions. Can be a single function or a set of functions",
    )
    temperature: Optional[float] = Field(
        default=0.7, description="The temperature for the language model. Default is 0.7"
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        description="The maximum number of tokens for the language model. Default is 4000",
    )
    model_config = {"arbitrary_types_allowed": True}
