from agentFramework.tool import Tool, extract_tool
from agentFramework.tracker import Tracker

from typing import Sequence, Callable, ClassVar, Generic, Literal, TypeVar, Union, Optional, Any, Type, List, Dict, Annotated, Unpack, get_type_hints, get_origin, get_args

from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field, field_validator

import json
from uuid import uuid4
import time
import random
from datetime import datetime

import mlflow
import mlflow.tracing
import mlflow.entities
from mlflow.types.chat import ChatMessage as MLFlowChatMessage, ToolCall as MLFlowToolCall, Function as MLFlowFunction

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('args', mode='before')
    @classmethod
    def parse_args(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        return value

    def getMLFlowToolCall(self) -> MLFlowToolCall:
        return MLFlowFunction(name=self.name, arguments=json.dumps(self.args)).to_tool_call(self.id)

class ExecutionResult(BaseModel):
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None

    def toString(self) -> str:
        if self.success:
            if self.output is None:
                return ""
            return str(self.output)
        else:
            return str(self.error)

class ToolResult(BaseModel):
    id: str
    name: str
    result: ExecutionResult

    def getMLFlowToolCall(self) -> MLFlowToolCall:
        return MLFlowFunction(name=self.name, arguments=(self.result.model_dump_json(exclude_none=True))).to_tool_call(self.id)

class MessageRole(str, Enum):
    MODEL = "model"
    USER = "user"

class TextMessage(BaseModel):
    type: Literal["text"] = "text"
    role: MessageRole
    text: str

    def export(self):
        return self.text
    
    def getMLFlowMessage(self) -> MLFlowChatMessage:
        return MLFlowChatMessage(role='user' if self.role == MessageRole.USER else 'assistant', content=self.text)

class ToolMessage(BaseModel):
    type: Literal["tool"] = "tool"
    role: MessageRole
    # Only one of these should be present
    tool_calls: List[ToolCall] = []
    tool_results: List[ToolResult] = []
    def export(self):
        if len(self.tool_calls) > 0:
            return [tool_call.model_dump(exclude_none=True) for tool_call in self.tool_calls]
        if len(self.tool_results) > 0:
            return [tool_result.model_dump(exclude_none=True) for tool_result in self.tool_results]
        return []

    def getMLFlowMessage(self) -> MLFlowChatMessage:
        if len(self.tool_calls) > 0:
            return MLFlowChatMessage(role='assistant', tool_calls=[tool_call.getMLFlowToolCall() for tool_call in self.tool_calls])
        if len(self.tool_results) > 0:
            return MLFlowChatMessage(role='tool', tool_calls=[tool_result.getMLFlowToolCall() for tool_result in self.tool_results])
        raise RuntimeError("ToolMessage must have either tool_calls or tool_results")

    def get(self, name: str) -> Optional[ExecutionResult]:
        for tool_result in self.tool_results:
            if tool_result.name == name:
                return tool_result.result
        return None

Message = Union[TextMessage, ToolMessage]

class ToolMode(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    NONE = "none"

class RetryLater(Exception):
    """
    Exception to indicate that the LLM should retry later.
    This is used for rate limiting and other temporary issues.
    """
    def __init__(self, reason: str, original: Any):
        super().__init__(reason)
        self.reason = reason
        self.original = original

class LLMParameters(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

    tracker: Tracker = Field(default_factory=Tracker)
    temperature: float = 0.5
    end_tool_mode: bool = False

Params = TypeVar('Params', bound=LLMParameters)

class LLM(ABC, Generic[Params]):
    """
    Abstract class to unify access to different LLM providers.
    """
    parameters_class: Type[Params]

    def __init__(self, **kwargs):
        self.parameters = self.parameters_class(**kwargs)
        self.tracker = self.parameters.tracker

    def setAgent(self, agent):
        if self.tracker is not None:
            self.tracker.setAgent(agent)
            
    def generate(self, systemprompt: str, messages: Sequence[Message], tools: Optional[Sequence[Tool]] = None, tool_mode: ToolMode = ToolMode.AUTO) -> Message:
        if tools is None:
            tools = []
        with mlflow.start_span(name=self.__class__.__name__, span_type='LLM') as span:
            span.set_attributes({'tools': tools, 'tool_mode': tool_mode})
            span.set_inputs(messages[-1].export())
            mlflow.tracing.set_span_chat_messages(span, [MLFlowChatMessage(role='system', content=systemprompt)])
            mlflow.tracing.set_span_chat_messages(span, [m.getMLFlowMessage() for m in messages], True)

            systemprompt_tool = ""
            for tool in tools:
                if tool.system_prompt:
                    systemprompt_tool += f"\n{tool.system_prompt}"
            
            # if len(systemprompt_tool) > 0:
            #     systemprompt += f"\nAvailable Tools:{systemprompt_tool}"

            result = self._infer(systemprompt, messages, tools, tool_mode)
            mlflow.tracing.set_span_chat_messages(span, [result.getMLFlowMessage()], True)
            span.set_outputs(result.export())
            return result

    def _infer(self, systemprompt: str, messages: Sequence[Message], tools: Sequence[Tool], tool_mode: ToolMode) -> Message:
        # Track last request time at class level
        if not hasattr(LLM, '_last_request_time'):
            LLM._last_request_time = datetime.now().timestamp() - 10  # Initialize to allow first request
        
        # Ensure minimum 1 second between requests system-wide
        current_time = datetime.now().timestamp()
        time_since_last_request = current_time - LLM._last_request_time
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        
        # Exponential backoff parameters
        max_retries = 5
        retries = 0
        while retries < max_retries:
            LLM._last_request_time = datetime.now().timestamp()
            try:                
                return self._inferModel(systemprompt, messages, tools, tool_mode)                
            except RetryLater as e:
                base_delay = 5
                delay = base_delay * (2 ** (retries)) + random.uniform(0, 1)
                retries += 1
                span = mlflow.get_current_active_span()
                if span:
                    span.add_event(mlflow.entities.SpanEvent('exception', attributes={'reason': e.reason, 'message': f'Exponential backoff\nStep {retries} retrying in {delay} seconds','original': str(e.original)}))
                print(f'Exponential backoff\nStep {retries} retrying in {delay} seconds')
                print(e)

                time.sleep(delay)

        raise RuntimeError("Max retries exceeded")        

    @abstractmethod
    def _inferModel(self, systemprompt: str, messages: Sequence[Message], tools: Sequence[Tool], tool_mode: ToolMode) -> Message:
        pass
