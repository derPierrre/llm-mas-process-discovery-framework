from agentFramework.llm import LLM, LLMParameters, Message, ToolCall, MessageRole, TextMessage, ToolMessage, ToolMode, RetryLater
from agentFramework.tool import Tool
from typing import Any, Sequence, Optional, Dict
import os
import json

from mistralai import Mistral as MistralClient
from mistralai.models import SDKError

class MistralParameters(LLMParameters):
    model: str = 'mistral-large-2411'
    api_key: Optional[str] = None

class Mistral(LLM[MistralParameters]):
    parameters_class = MistralParameters

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.parameters.api_key or os.getenv("MISTRAL_API_KEY")
        self.client = MistralClient(api_key=self.api_key)
        
    def _toolToMistral(self, tool: Tool):
        schema = tool.to_schema()
        return {'function': schema, "type": "function"}
 
    def _messageToMistral(self, message: Message) -> Sequence[Dict[str, Any]]:
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
        formatted_messages = [{"role": "system", "content": systemprompt}]
        for message in messages:
            formatted_messages.extend(self._messageToMistral(message))

        mode = 'auto'
        if tool_mode == ToolMode.FORCE:
            mode = 'any'
        elif tool_mode == ToolMode.NONE:
            mode = 'none'

        transformed_tools = [self._toolToMistral(tool) for tool in tools]
        try:
            response = self.client.chat.complete(
                model=self.parameters.model,
                messages=formatted_messages,
                tools=transformed_tools,
                tool_choice=mode,
                parallel_tool_calls=True,
                temperature=self.parameters.temperature,
            )
        except SDKError as err:
            if err.status_code == 429:
                raise RetryLater("Rate limit exceeded", err)
            else:
                raise err
        
        self.tracker.logTokens(response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
        self.tracker.logMetadata({'model': self.parameters.model, 'tool_mode': tool_mode, 'temperature': self.parameters.temperature})
        
        # Process response
        if response.choices[0].message.tool_calls and len(response.choices[0].message.tool_calls) > 0:
            tool_calls = [ToolCall(id=tc.id, name=tc.function.name, args=tc.function.arguments) for tc in response.choices[0].message.tool_calls]
            return ToolMessage(role=MessageRole.MODEL, tool_calls=tool_calls)
        
        return TextMessage(role=MessageRole.MODEL, text=response.choices[0].message.content)
