from agentFramework.llm import LLM, LLMParameters, Message, ToolCall, MessageRole, TextMessage, ToolMessage, ToolMode, RetryLater
from agentFramework.tool import Tool
from typing import Any, Sequence, Optional, Dict
import os
import json

from openai import OpenAI as OpenAIClient
from openai.types.responses import FunctionToolParam

class DeepseekParameters(LLMParameters):
    model: str = 'deepseek-chat'
    api_key: Optional[str] = None

class Deepseek(LLM[DeepseekParameters]):
    parameters_class = DeepseekParameters

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.parameters.api_key or os.getenv("DEEPSEEK_API_KEY")
        
        self.client = OpenAIClient(api_key=self.api_key, base_url="https://api.deepseek.com")

    def _toolToOpenAI(self, tool: Tool) -> FunctionToolParam:
        schema = tool.to_schema()
        return {**schema, "type": "function"} # type: ignore

    def _toolToDeepseek(self, tool: Tool):
        schema = tool.to_schema()
        return {'function': schema, "type": "function"}

    def _messageToDeepseek(self, message: Message) -> Sequence[Dict[str, Any]]:
        if isinstance(message, TextMessage):
            return [{
                "role": "user" if message.role == MessageRole.USER else "assistant",
                "content": message.text
            }]
        elif isinstance(message, ToolMessage):
            if message.tool_calls:
                return [{
                    "role": "assistant",
                    "tool_calls": [{'id': tc.id, 'function': {'name': tc.name, 'arguments': json.dumps(tc.args)}, 'type': 'function'} for tc in message.tool_calls]
                }]
            else:
                return [{'role': 'tool', 'tool_call_id': tr.id, 'content': tr.result.toString()} for tr in message.tool_results]

        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    
    def _inferModel(self, systemprompt: str, messages: Sequence[Message], tools: Sequence[Tool], tool_mode: ToolMode) -> Message:

        # Convert messages to OpenAI format
        formatted_messages = [{"role": "system", "content": systemprompt}]
        for message in messages:
            formatted_messages.extend(self._messageToDeepseek(message))
        # Call OpenAI API

        mode = 'auto'
        if tool_mode == ToolMode.FORCE:
            mode = 'required'
        elif tool_mode == ToolMode.NONE:
            mode = 'none'


        response = self.client.chat.completions.create(
            model=self.parameters.model,
            messages=formatted_messages,
            tools=[self._toolToDeepseek(tool) for tool in tools],
            stream=False,
            parallel_tool_calls=True,
            tool_choice=mode,
            temperature=self.parameters.temperature,
        )
        
        self.tracker.logTokens(response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        self.tracker.logMetadata({'model': self.parameters.model, 'tool_mode': tool_mode, 'temperature': self.parameters.temperature})
        # Process response
        
        if response.choices[0].message.tool_calls and len(response.choices[0].message.tool_calls) > 0:
            tool_calls = [ToolCall(id=tc.id, name=tc.function.name, args=tc.function.arguments) for tc in response.choices[0].message.tool_calls]
            return ToolMessage(role=MessageRole.MODEL, tool_calls=tool_calls)
        
        return TextMessage(role=MessageRole.MODEL, text=response.choices[0].message.content)
