import inspect
import functools
from pydantic import BaseModel, Field, create_model
from typing import Callable, TypedDict, Union, Optional, Any, Type, List, Dict, Annotated, TypeVar, cast, get_type_hints, get_origin, get_args

class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    name: str
    description: str
    system_prompt: Optional[str] = None
    parameters: Type[BaseModel]
    return_type: Optional[Any] = None
    function: Callable
    required: List[str] = []  # List of required parameter names

    def to_schema(self, typeMapping: Optional[Dict[str, str]] = None, defaultType: Optional[str] = None) -> ToolSchema:
        parameters = self.parameters.model_json_schema()
        del parameters["title"]
        for key in parameters["properties"]:
            del parameters["properties"][key]["title"]
            if typeMapping:
                old_type = parameters["properties"][key]["type"]
                parameters["properties"][key]["type"] = typeMapping.get(old_type, defaultType)
                if parameters["properties"][key]["type"] is None:
                    raise ValueError(f"Type {old_type} is not supported, and no default type is provided.")

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }


def extract_tool(func: Callable) -> Tool:
    """
    Create a Tool object by analyzing a function's signature, including Annotated type hints.
    
    Args:
        func: The function to analyze
    
    Returns:
        Tool: A Tool instance with the function's metadata
    """
    # Get function name
    name = func.__name__
    
    # Get function docstring
    description = inspect.getdoc(func) or ""
    
    # Get type hints including Annotated
    type_hints = get_type_hints(func, include_extras=True)
    
    # Separate return type from parameter types
    return_type = type_hints.pop("return", Any)
    
    # Prepare fields for parameter model
    param_fields = {}
    
    # Keep track of required parameters
    required_params = []
    
    # Process each parameter
    signature = inspect.signature(func)
    for param_name, param in signature.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
            
        # Get the type hint for this parameter
        param_type = type_hints.get(param_name, Any)
        param_description = ""
        
        # Process Annotated types
        if get_origin(param_type) is Annotated:
            args = get_args(param_type)
            if len(args) >= 2:
                param_type = args[0]  # First arg is the actual type
                param_description = args[1] if isinstance(args[1], str) else ""
        
        # Check if the parameter is Optional
        is_optional = False
        original_type = param_type
        
        # Handle both Optional[X] and Union[X, None]
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            # Check if None or NoneType is in the Union
            if type(None) in args or None in args:
                is_optional = True
                # Extract the original type (removing None from the Union)
                types = [arg for arg in args if arg is not type(None) and arg is not None]
                if len(types) == 1:
                    original_type = types[0]
                else:
                    # Handle Union with multiple types plus None
                    original_type = Union[tuple(types)]
        
        # Default value
        if param.default is not inspect.Parameter.empty:
            # Parameter has a default value
            default_value = param.default
        elif is_optional:
            # Parameter is Optional and has no default value, set to None
            default_value = None
        else:
            # Required parameter
            default_value = ...
            # Add to required parameters list
            required_params.append(param_name)
        
        # Create field with description and extracted type information
        param_fields[param_name] = (original_type, Field(default=default_value, description=param_description))
    
    # Create the parameters model
    parameters_model = create_model(f"{name}_parameters", **param_fields)
    
    system_message = getattr(func, 'system_message', None)

    # Create and return the Tool
    return Tool(
        name=name,
        description=description,
        system_prompt=system_message,
        parameters=parameters_model,
        return_type=return_type,
        function=func,
        required=required_params  # Add the required parameters list
    )


F = TypeVar('F', bound=Callable)

def add_system_message(message: str) -> Callable[[F], F]:
    """
    Decorator that attaches a system message to a function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        # Attach the message as an attribute
        setattr(wrapper, 'system_message', message)
        
        # Cast to preserve the original function's type
        return cast(F, wrapper)
    
    return decorator

