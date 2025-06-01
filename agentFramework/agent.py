from agentFramework.llm import *
from agentFramework.tool import Tool, extract_tool, add_system_message

from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TypeVar

from pydantic import BaseModel, create_model
import re

import mlflow
import mlflow.tracing
from mlflow.types.chat import ChatMessage as MLFlowChatMessage

T = TypeVar('T', bound=BaseModel)

class SubAgentTool(Tool):
    model_config = {
        "arbitrary_types_allowed": True
    }
    agent: 'Agent'

SubAgent = Union['Agent', SubAgentTool]

PreCallable = Callable[[ToolCall], Optional[ToolCall]]
PostCallable = Callable[[ToolCall, ExecutionResult], Optional[ExecutionResult]]

PreHookTuple = Tuple[Literal["pre"], PreCallable]
PostHookTuple = Tuple[Literal["post"], PostCallable]

class Agent:
    def __init__(self, llm: LLM, agent_name: str, system_prompt: str, *, tools: List[Union[Tool, Callable]] = [], max_iterations: int = 10, default_tool_mode: ToolMode = ToolMode.AUTO, sub_agents: List[SubAgent] = [], injected_system_message: Optional[str] = None):
        self.llm = llm
        self.name = agent_name
        self.sanatized_name = re.sub(r'[^a-zA-Z0-9_.-]', '', agent_name.replace(' ', '_'))
        
        self.system_prompt = system_prompt
        
        self.tool_map: Dict[str, Tool] = {}
        for tool in tools:
            if isinstance(tool, Callable):
                self.add_function_as_tool(tool)
            else: 
                self.add_tool(tool)
    
        self.max_iterations = max_iterations
        self.allow_poking = True
        self.default_tool_mode = default_tool_mode

        self.history: List[Message] = []

        self.sub_agents: List[SubAgentTool] = []
        for sub_agent in sub_agents:
            self.add_sub_agent(sub_agent)

        self.hook_map: Dict[str, List[Union[PreHookTuple, PostHookTuple]]] = {}

        self.injected_system_message = injected_system_message

        self.llm.setAgent(self)

    def add_function_as_tool(self, function: Callable) -> Callable:
        """Add a function as tool to the agent, returns a function to remove the tool"""
        tool = extract_tool(function)
        return self.add_tool(tool)

    def add_tool(self, tool: Tool) -> Callable:
        """Add a tool to the agent, returns a function to remove the tool"""
        self.tool_map[tool.name] = tool
        return lambda: self.tool_map.pop(tool.name, None)

    def add_sub_agent(self, agent: SubAgent):
        if isinstance(agent, Agent):
            tool = agent.as_subagent()
        else:
            tool = agent
        
        self.sub_agents.append(tool)
        self.tool_map[tool.name] = tool

    def get_all_subagents(self) -> List['Agent']:
        """Returns all subagents including this"""
        agents: List['Agent'] = [self]
        for subagent in self.sub_agents:
            agents.extend(subagent.agent.get_all_subagents())
        return agents
    
    def as_subagent(self, *, description: Optional[str] = None, system_message: Optional[str] = None, call: Optional[Callable] = None) -> SubAgentTool:
        """Formats the agent as tool to be used as subagent in other agents"""
        if description is None:
            description = f"Deligate a task to {self.name} and get back the result"
        
        if call is None:
            def deligate(task: str) -> str:
                return self.chat(task)
            call = deligate

        agent_tool = extract_tool(call)
        agent_tool = SubAgentTool(agent=self, **agent_tool.model_dump())
        
        agent_tool.name = self.sanatized_name
        agent_tool.description = description
        
        if system_message is not None:
            agent_tool.system_prompt = system_message
        elif self.injected_system_message is not None:
            agent_tool.system_prompt = self.injected_system_message

        return agent_tool

    def listen_to_all_deligations(self, listener: Callable[[str, str, str], None]) -> Callable:
        remove: List[Callable] = []
        for subagent in self.sub_agents:
            removeCall = self.add_tool_hook(subagent.name, 
                                            pre=lambda tool_call, name=subagent.name: listener(self.name, name, tool_call.args.get('task', '-- No Task --')),
                                            post=lambda tool_call, result, name=subagent.name: listener(name, self.name, result.toString()))

            remove.append(removeCall)
        
        return lambda: [call() for call in remove]


    def add_tool_hook(self, tool_name: str, *, pre: Optional[PreCallable] = None, post: Optional[PostCallable] = None) -> Callable:
        """Add a listener to the agent, returns a function to remove the listener, doesn't validate if the tool exists, as the tools are runtime dependent"""
        
        if tool_name not in self.hook_map:
            self.hook_map[tool_name] = []
        
        hooks = []
        if pre is not None:
            hooks.append(('pre', pre))
        if post is not None:
            hooks.append(('post', post))

        self.hook_map[tool_name].extend(hooks)
        return lambda: [self.hook_map[tool_name].remove(hook) for hook in hooks]
    
    def clear_history(self):
        self.history = []

    def poke(self) -> str:
        """Use this method, to let the agent generate a message without any input"""
        if not self.allow_poking:
            raise RuntimeError(f"Poking is not allowed for {self.name}")
        
        return self.chat('Hey, I am ready, interact with me!')

    def chat(self, user_input: str, tool_mode: Optional[ToolMode] = None) -> str:
        if tool_mode is None:
            tool_mode = self.default_tool_mode
        message = TextMessage(role=MessageRole.USER, text=user_input)
        response = self._invoke(message, tool_mode)
        return response.text
    
    def _invoke(self, message: TextMessage, tool_mode: ToolMode) -> TextMessage:        
        with mlflow.start_span(name=self.name, span_type='AGENT') as span:
            span.set_inputs(message.text)
            self.history.append(message)
            self._track_mlflow(span) # do this first to have it in case of an error
            self.history = self._run(self.max_iterations, self.history, tool_mode, self.tool_map, lambda last_message: last_message.type == 'text')
            response = self.history[-1]  # type: TextMessage # type: ignore
            self._track_mlflow(span) # do again to include new messages
            span.set_outputs(response.text)
            return response

    def _track_mlflow(self, span: mlflow.entities.LiveSpan):
        mlflow.tracing.set_span_chat_messages(span, [MLFlowChatMessage(role='system', content=self.system_prompt)]) # type: ignore
        mlflow.tracing.set_span_chat_messages(span, [m.getMLFlowMessage() for m in self.history], True) # type: ignore

    def structuredAnswer(self, user_input: str, desired_output: Type[T]) -> T:
        tool = Tool(
            name="set_output",
            description="use this function to reply in the desired format",
            parameters=desired_output,
            function=lambda **kwargs: desired_output(**kwargs),
            required=desired_output.model_json_schema().get('required', [])
            )
        messages = self.history.copy()
        messages.append(TextMessage(role=MessageRole.USER, text=user_input))
        messages = self._run(1, messages, ToolMode.FORCE, {tool.name: tool}, lambda last_message: last_message.type == 'tool' and last_message.get('set_output') is not None)
        result = messages[-1].get('set_output') # type: ExecutionResult # type: ignore
        return result.output # type: ignore

    def _run(self, iterations: int, messages: List[Message], tool_mode: ToolMode, tool_map: Dict[str, Tool], stop_criteria: Callable[[Message], bool]) -> List[Message]:
        if iterations <= 0:
            raise RuntimeError(f"Agent reached max iterations while processing user input")
        
        systemprompt = self.system_prompt
        original_tool_map = tool_map
        next_tool_mode = tool_mode
        if tool_mode == ToolMode.FORCE:
            if self.llm.parameters.end_tool_mode:
                # systemprompt += "\n\nYou have to use a tool, in case you don't need one use 'end_tool_mode' to indicate that you are done with the tools"

                # def end_tool_mode():
                #     """Use this tool to indicate that you are done with using tools and don't need it anymore"""
                #     nonlocal tool_mode
                #     print("Ending tool mode")
                #     tool_mode = ToolMode.AUTO
                
                tool_map = tool_map.copy()
                #tool_map['end_tool_mode'] = extract_tool(end_tool_mode)
                tool_map['end_tool_mode'] = Tool(name='end_tool_mode', description="Use this tool to indicate that you are done with using tools and don't need it anymore", parameters=create_model('Empty'), function=lambda: None)
            else:
                next_tool_mode = ToolMode.AUTO

        
        response = self.llm.generate(systemprompt=systemprompt, messages=messages, tools=list(tool_map.values()), tool_mode=tool_mode)
        if isinstance(response, ToolMessage):
            if any([tc.name == 'end_tool_mode' for tc in response.tool_calls]):
                print(f"{self.name} ending tool mode")
                return self._run(iterations, messages, ToolMode.AUTO, original_tool_map, stop_criteria)

        messages.append(response)
        if stop_criteria(response):
            return messages

        if response.type == 'tool':
            tool_responses = self._execute_all_tools(response.tool_calls, tool_map)
            response = ToolMessage(role=MessageRole.USER, tool_results=tool_responses)
            messages.append(response)

        if stop_criteria(response):
            return messages
        
        return self._run(iterations - 1, messages, next_tool_mode, original_tool_map, stop_criteria)

    def _execute_all_tools(self, tool_calls: List[ToolCall], tool_map: Dict[str, Tool]) -> List[ToolResult]:
        return [ToolResult(id=tool_call.id, name=tool_call.name, result=self._execute_tool(tool_call, tool_map)) for tool_call in tool_calls]

    def _execute_tool(self, tool_call: ToolCall, tool_map: Dict[str, Tool]) -> ExecutionResult:
        hooks = self.hook_map.get(tool_call.name, [])
        for hook in hooks:
            if hook[0] == 'pre':
                potentialy_transformed = hook[1](tool_call)
                if potentialy_transformed is not None:
                    tool_call = potentialy_transformed
        try:
            result = ExecutionResult(success=True, output=self._execute_tool_raw(tool_call, tool_map))
        except Exception as e:
            result = ExecutionResult(success=False, error=str(e))

        for hook in hooks:
            if hook[0] == 'post':
                potentialy_transformed = hook[1](tool_call, result)
                if potentialy_transformed is not None:
                    result = potentialy_transformed

        return result

    def _execute_tool_raw(self, tool_call: ToolCall, tool_map: Dict[str, Tool]) -> Any:
        with mlflow.start_span(name=tool_call.name, span_type='TOOL') as span:
            span.set_inputs({'name': tool_call.name, 'args': tool_call.args})

            if tool_call.name not in tool_map:
                raise ValueError(f"Tool '{tool_call.name}' not found, available tools: {', '.join(tool_map.keys())}")
            
            tool = tool_map[tool_call.name]
            result = tool.function(**tool_call.args)
        
            span.set_outputs(result)
            return result


@add_system_message("Use internal_thinking to break down any situation and plan your next action")
def internal_thinking(thought: str) -> str:
    """Secret scratchpad to think about the next action"""
    return thought