from google import genai
import os

from agentFramework.llm import LLM, Message, ToolCall, MessageRole, TextMessage, ToolMessage, ToolMode, LLMParameters
from agentFramework.tool import Tool
from typing import Optional, Any, Sequence, List, Set, Tuple, Union, get_origin, get_args


TYPE_MAPPING = {
    'array': genai.types.Type.ARRAY,
    'boolean': genai.types.Type.BOOLEAN,
    'integer': genai.types.Type.INTEGER,
    'number': genai.types.Type.NUMBER,
    'object': genai.types.Type.OBJECT,
    'string': genai.types.Type.STRING,
    # Common aliases or variations
    'int': genai.types.Type.INTEGER,
    'float': genai.types.Type.NUMBER,
    'double': genai.types.Type.NUMBER,
    'str': genai.types.Type.STRING,
    'dict': genai.types.Type.OBJECT,
    'list': genai.types.Type.ARRAY,
    'tuple': genai.types.Type.ARRAY,
    'bool': genai.types.Type.BOOLEAN,
}


class GeminiParameters(LLMParameters):
    model: str = 'gemini-2.0-flash'
    api_key: Optional[str] = None

class Gemini(LLM[GeminiParameters]):
    parameters_class = GeminiParameters

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.parameters.api_key or os.getenv("GEMINI_API_KEY")
        self.model = self.parameters.model

    def _messageToGemini(self, message: Message) -> Any:
        if message.role == MessageRole.USER:
            parts = None
            if message.type == 'text':
                parts = message.text
            if message.type == 'tool':
                parts = [genai.types.Part.from_function_response(name=tool_result.name, response=tool_result.result.model_dump(exclude_none=True, exclude=set('success'))) for tool_result in message.tool_results]
            return genai.types.UserContent(parts)
        if message.role == MessageRole.MODEL:
            parts = None
            if message.type == 'text':
                parts = message.text
            if message.type == 'tool':
                parts = [genai.types.Part.from_function_call(name=tool_call.name, args=tool_call.args) for tool_call in message.tool_calls]
            return genai.types.ModelContent(parts)

        raise ValueError(f"{message.model_dump()} is not supported for gemini yet")

    def _toolToGemini(self, tool: Tool) -> Any:
        paramter_schema = tool.parameters.model_json_schema()
        parameters = dict()
        for key in paramter_schema['properties'].keys():
            property = paramter_schema['properties'][key]
            parameters[key] = genai.types.Schema(
                type=TYPE_MAPPING.get(property['type'].lower(), 'type_unspecified'),
                description=property.get('description', '')
                )

        return genai.types.FunctionDeclaration(
            name=tool.name,
            description=tool.description, 
            parameters=genai.types.Schema(properties=parameters, type='object', required=tool.required),
        )


    def _inferModel(self, systemprompt: str, messages: Sequence[Message], tools: Sequence[Tool], tool_mode: ToolMode) -> Message:
        client = genai.Client(api_key=self.api_key)
        history = messages[0:-1]
        request = messages[-1]

        config = {
            'temperature': self.parameters.temperature,
            'system_instruction': systemprompt
        }

        if len(tools) > 0:
            config['tools'] = [{'function_declarations': [self._toolToGemini(t) for t in tools]}]
            config['automatic_function_calling'] = { 'disable': True}
            mode = 'AUTO'
            if tool_mode == ToolMode.FORCE:
                mode = 'ANY'
            if tool_mode == ToolMode.NONE:
                mode = 'NONE'
            config['tool_config'] = {'function_calling_config': {'mode': mode}}
        
        chat = client.chats.create(model=self.model, config=config, history=[self._messageToGemini(m) for m in history])
        response = chat.send_message(self._messageToGemini(request).parts)
        
        usage = response.usage_metadata.model_dump()
        self.tracker.logTokens(usage['prompt_token_count'], usage['candidates_token_count'], usage['total_token_count'])
        self.tracker.logMetadata({'model': self.model, 'temperature': self.parameters.temperature, 'tool_mode': tool_mode})

        if response.function_calls:
            return ToolMessage(role=MessageRole.MODEL, tool_calls=[ToolCall(**fc.model_dump(exclude_none=True)) for fc in response.function_calls])
        if response.text:
            return TextMessage(role=MessageRole.MODEL, text=response.text)
        
        raise ValueError(f"Unexpected response from gemini: {response}")