import mlflow
from mlflow.types.chat import ChatMessage as MLFlowChatMessage
from agentFramework.agent import Agent
from typing import Optional, List, Callable, Tuple
from agentFramework.tool import Tool, extract_tool


def link_agents(actor: Agent, partners: List[Agent], *, interactionName: str = 'ask', anonymizePrefix: Optional[str] = None, callback: Optional[Callable[[str, str, str], None]] = None):
    """Links the agent with the partners, so that the agent can ask questions to the users and the users can answer them"""
    if anonymizePrefix:
        partnerMap = {f"{anonymizePrefix} {i}": user for i, user in enumerate(partners)}
    else: 
        partnerMap = {user.name: user for user in partners}

    def interact(user: str, request: str):
        nonlocal partnerMap
        nonlocal actor
        try:
            agent = partnerMap[user]
        except KeyError:
            raise KeyError(f'User "{user}" not found, available users: {", ".join(partnerMap.keys())}')

        if callback:
            callback(actor.name, agent.name, request)

        response = agent.chat(request)
        
        if callback:
            callback(agent.name, actor.name, response)
            #callback(f'"{actor.name}" @ "{name}":\n\n{request}', f'"{name}" @ "{actor.name}":\n\n{response}')
        
        return f"{user}: {response}"

    def single_interact(request: str):
        return interact(list(partnerMap.keys())[0], request)


    if len(partners) == 1:
        tool = extract_tool(single_interact)
    else:
        tool = extract_tool(interact)
        
    tool.name = interactionName
    users = ', '.join(list([f'"{name}"' for name in partnerMap.keys()]))
    
    if len(partners) > 1:
        tool.description = f"Interact with one of the users in this conversation {users}"
    else:
        tool.description = f"Interact with the user in this conversation using this tool"

    remove = actor.add_tool(tool)
