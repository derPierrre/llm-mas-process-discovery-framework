import os
import vertexai
from vertexai.generative_models import GenerativeModel, ResponseValidationError
from vertexai.generative_models import Content, FunctionDeclaration, Tool as VertexTool, Part, FunctionCall, GenerationConfig, ToolConfig
from google.api_core.exceptions import GoogleAPIError

from agentFramework.llm import LLM, LLMParameters, Message, ToolCall, MessageRole, TextMessage, ToolMessage, ToolMode, RetryLater
from agentFramework.tool import Tool
from typing import Any, List, Sequence

from pydantic import Field

TYPE_MAPPING_STRINGS = {
    'array': 'array',
    'boolean': 'boolean',
    'integer': 'integer',
    'number': 'number',
    'object': 'object',
    'string': 'string',
    'int': 'integer',
    'float': 'number',
    'double': 'number',
    'str': 'string',
    'dict': 'object',
    'list': 'array',
    'tuple': 'array',
    'bool': 'boolean',
}

class VertexParameters(LLMParameters):
    model: str = 'gemini-2.0-flash'
    location: str = Field(default=os.getenv("GOOGLE_CLOUD_LOCATION", ''))
    project: str = Field(default=os.getenv("GOOGLE_CLOUD_PROJECT", ''))
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.location:
            raise ValueError("Provide a location or set the environment variable: GOOGLE_CLOUD_LOCATION")
        if not self.project:
            raise ValueError("Provide a project or set the environment variable: GOOGLE_CLOUD_PROJECT")


class Vertex(LLM[VertexParameters]):
    parameters_class = VertexParameters
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vertexai.init(location=self.parameters.location, project=self.parameters.project)

    def _messageToPart(self, message: Message):
        if message.type == 'text':
            return [Part.from_text(message.text)]
        if message.type == 'tool':
            if len(message.tool_results) > 0:
                return [Part.from_function_response(name=tool_result.name, response=tool_result.result.model_dump(exclude_none=True, exclude=set('success'))) for tool_result in message.tool_results]
            if len(message.tool_calls) > 0:
                return [Part.from_dict({'functionCall': {'name': obj.name, 'args': dict(obj.args)}}) for obj in message.tool_calls]                
                return [FunctionCall(name=tool_call.name, args=tool_call.args) for tool_call in message.tool_calls]
        
        raise ValueError(f"{message.model_dump()} is not supported yet")

    def _messageToVertex(self, message: Message) -> Content:
        return Content(parts=self._messageToPart(message), role=message.role.value)

    def _toolToVertex(self, tool: Tool) -> FunctionDeclaration:
        return FunctionDeclaration(**tool.to_schema(TYPE_MAPPING_STRINGS, 'type_unspecified'))
        
        paramter_schema = tool.parameters.model_json_schema()
        properties = dict()
        for key in paramter_schema['properties'].keys():
            property = paramter_schema['properties'][key]
            properties[key] = {
                'type': TYPE_MAPPING_STRINGS.get(property['type'].lower(), 'type_unspecified'),
                'description': property.get('description', '')
            }
        
        return FunctionDeclaration(
            name=tool.name,
            description=tool.description, 
            parameters={
                'type': 'object',
                'properties': properties,
                'required': tool.required,
            }
        )


    def _inferModel(self, systemprompt: str, messages: Sequence[Message], tools: Sequence[Tool], tool_mode: ToolMode) -> Message:
        history = messages[0:-1]
        request = messages[-1]
        config = {
            'generation_config': GenerationConfig(temperature=self.parameters.temperature),
            'system_instruction': systemprompt
        }

        if len(tools) > 0:
            config['tools'] = [VertexTool(function_declarations= [self._toolToVertex(t) for t in tools])]
            # config['automatic_function_calling'] = { 'disable': True}
            mode = ToolConfig.FunctionCallingConfig.Mode.AUTO
            if tool_mode == ToolMode.FORCE:
                mode = ToolConfig.FunctionCallingConfig.Mode.ANY
            if tool_mode == ToolMode.NONE:
                mode = ToolConfig.FunctionCallingConfig.Mode.NONE
            config['tool_config'] = ToolConfig(function_calling_config=ToolConfig.FunctionCallingConfig(mode=mode))
        
        model = GenerativeModel(self.parameters.model, **config)

        chat = model.start_chat(history=[self._messageToVertex(m) for m in history])
        try:
            response = chat.send_message(self._messageToPart(request))
        except GoogleAPIError as err:
            if err.code == 429:
                raise RetryLater("Rate limit exceeded", err)
            else:
                raise err
        except ResponseValidationError as err:
            raise RetryLater("Invalid resonse generation", err)
        
        self.tracker.logTokens(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count, response.usage_metadata.total_token_count )
        self.tracker.logMetadata({'model': self.parameters.model, 'temperature': self.parameters.temperature, 'tool_mode': tool_mode})
        
        try: 
            return TextMessage(role=MessageRole.MODEL, text=response.text)
        except Exception:
            pass
        
        try: 
            function_calls = [fc for c in response.candidates for fc in c.function_calls]
            return ToolMessage(role=MessageRole.MODEL, tool_calls=[create_tool_call_from_structured_object(fc) for fc in function_calls])
        except Exception:
            pass
        
        raise ValueError(f"Unexpected response from vertex: {response}")
    

def create_tool_call_from_structured_object(obj: Any) -> ToolCall:
    # Extract name
    if isinstance(obj, dict):
        name = obj.get('name')
    else:
        name = getattr(obj, 'name', None)
        
    if not name:
        raise ValueError("Input object must contain a 'name' field")
    
    # Extract and convert args
    args_dict = {}
    
    # Get the args structure
    if isinstance(obj, dict):
        args_struct = obj.get('args', {})
    else:
        args_struct = getattr(obj, 'args', {})
    
    # Handle protobuf Struct format with 'fields'
    if hasattr(args_struct, 'fields'):
        # Handle protobuf Struct
        for key, value_struct in args_struct.fields.items():
            # Extract value based on which field is set
            if hasattr(value_struct, 'number_value') and value_struct.number_value:
                args_dict[key] = value_struct.number_value
            elif hasattr(value_struct, 'string_value') and value_struct.string_value:
                args_dict[key] = value_struct.string_value
            elif hasattr(value_struct, 'bool_value'):  # Need to check existence, not value
                args_dict[key] = value_struct.bool_value
            elif hasattr(value_struct, 'struct_value') and value_struct.struct_value:
                # Recursively process nested structs
                args_dict[key] = struct_to_dict(value_struct.struct_value)
            elif hasattr(value_struct, 'list_value') and value_struct.list_value:
                args_dict[key] = [get_value_from_struct(item) for item in value_struct.list_value.values]
    
    # If it's a dictionary with 'fields'
    elif isinstance(args_struct, dict) and 'fields' in args_struct:
        for field in args_struct.get('fields', []):
            key = field.get('key')
            value_obj = field.get('value', {})
            
            # Extract the value based on which field is set
            if 'number_value' in value_obj:
                args_dict[key] = value_obj['number_value']
            elif 'string_value' in value_obj:
                args_dict[key] = value_obj['string_value']
            elif 'bool_value' in value_obj:
                args_dict[key] = value_obj['bool_value']
            elif 'struct_value' in value_obj:
                # Recursively process nested structs
                args_dict[key] = process_struct_value(value_obj['struct_value'])
            elif 'list_value' in value_obj:
                args_dict[key] = process_list_value(value_obj['list_value'])
    
    # If args_struct is already a dictionary, use it directly
    elif isinstance(args_struct, dict):
        args_dict = args_struct
    
    return ToolCall(name=name, args=args_dict)


# Helper functions for processing complex structures
def get_value_from_struct(value_struct):
    """Extract the appropriate value from a protobuf Value."""
    if hasattr(value_struct, 'number_value') and value_struct.number_value:
        return value_struct.number_value
    elif hasattr(value_struct, 'string_value') and value_struct.string_value:
        return value_struct.string_value
    elif hasattr(value_struct, 'bool_value'):  # Need to check existence, not value
        return value_struct.bool_value
    elif hasattr(value_struct, 'struct_value') and value_struct.struct_value:
        return struct_to_dict(value_struct.struct_value)
    elif hasattr(value_struct, 'list_value') and value_struct.list_value:
        return [get_value_from_struct(item) for item in value_struct.list_value.values]
    return None

def struct_to_dict(struct):
    """Convert a protobuf Struct to a Python dictionary."""
    result = {}
    for key, value in struct.fields.items():
        result[key] = get_value_from_struct(value)
    return result

def process_struct_value(struct_value):
    """Process a struct value from a dictionary representation."""
    if isinstance(struct_value, dict) and 'fields' in struct_value:
        result = {}
        for field in struct_value['fields']:
            result[field['key']] = get_value_from_dict(field['value'])
        return result
    return struct_value

def process_list_value(list_value):
    """Process a list value from a dictionary representation."""
    if isinstance(list_value, dict) and 'values' in list_value:
        return [get_value_from_dict(item) for item in list_value['values']]
    return list_value

def get_value_from_dict(value_dict):
    """Extract the appropriate value from a dictionary representing a Value."""
    if 'number_value' in value_dict:
        return value_dict['number_value']
    elif 'string_value' in value_dict:
        return value_dict['string_value']
    elif 'bool_value' in value_dict:
        return value_dict['bool_value']
    elif 'struct_value' in value_dict:
        return process_struct_value(value_dict['struct_value'])
    elif 'list_value' in value_dict:
        return process_list_value(value_dict['list_value'])
    return None

