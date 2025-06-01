from agentFramework import (
    Agent,
    internal_thinking,
    add_system_message,
    extract_tool,
    ToolMode,
    link_agents,
)
from .tools import generate_powl_model
from agentFramework.models import BaseLLM, Vertex, Mistral, Deepseek
from typing import List
import yaml
import os
from .prompts import get_prompt
from agentFramework.tracker import MLFlowTracker

MODEL = "vertex"


def get_model() -> BaseLLM:
    print(f"Using model: {MODEL}")
    if MODEL == "vertex":
        return Vertex(
            tracker=MLFlowTracker(),
            model="gemini-2.5-pro-preview-05-06",
            location="us-central1",
            end_tool_mode=True,
        )
    if MODEL == "vertex-flash":
        return Vertex(tracker=MLFlowTracker())
    if MODEL == "mistral":
        return Mistral(tracker=MLFlowTracker(), end_tool_mode=True)
    if MODEL == "deepseek":
        return Deepseek(
            tracker=MLFlowTracker(), model="deepseek-chat", end_tool_mode=True
        )
    if MODEL == "deepseek-reasoning":
        return Deepseek(tracker=MLFlowTracker(), model="deepseek-reasoner")

    raise ValueError(f"Unknown model: {MODEL}")


def powl() -> Agent:
    return Agent(
        llm=get_model(),
        agent_name="Process Modeler",
        system_prompt=get_prompt("process_modeler"),
        tools=[generate_powl_model],
        default_tool_mode=ToolMode.FORCE,
        injected_system_message="Whenever you feel you have enough information you can deligate the task to the Process Modeler, he will try to model the process and will give you feedback if he needs more inofrmation about sth specific.",
    )


def duo() -> Agent:
    return Agent(
        llm=get_model(),
        agent_name="Manager_Interviewer",
        max_iterations=100,
        system_prompt=get_prompt("manager_and_interviewer"),
        tools=[internal_thinking],
        default_tool_mode=ToolMode.FORCE,
        sub_agents=[powl()],
    )


def manager() -> Agent:
    interviewer = Agent(
        llm=get_model(),
        agent_name="Knowledge Gatherer",
        max_iterations=100,
        system_prompt=get_prompt("interviewer"),
        tools=[internal_thinking],
        default_tool_mode=ToolMode.FORCE,
    )
    return Agent(
        llm=get_model(),
        agent_name="Manager",
        max_iterations=100,
        system_prompt=get_prompt("manager"),
        tools=[internal_thinking],
        default_tool_mode=ToolMode.FORCE,
        sub_agents=[powl(), interviewer],
    )


def monolithic() -> Agent:
    return Agent(
        llm=get_model(),
        agent_name="Process Consultant",
        max_iterations=100,
        system_prompt=get_prompt("monolithic"),
        tools=[generate_powl_model, internal_thinking],
        default_tool_mode=ToolMode.FORCE,
    )


def team(process_owners: List[Agent], log) -> Agent:
    manager = Agent(
        llm=get_model(),
        agent_name="Manager",
        system_prompt=get_prompt("manager"),
        tools=[internal_thinking],
        default_tool_mode=ToolMode.FORCE,
        max_iterations=50,
        sub_agents=[powl()],
    )
    knowledgeGatherer = Agent(
        llm=get_model(),
        agent_name="Knowledge Gatherer",
        system_prompt=get_prompt("knowledgeGatherer"),
        tools=[],
        injected_system_message="The Interviewer can gather information from the process owners",
        default_tool_mode=ToolMode.FORCE,  # tools will be added later
        max_iterations=50,
    )

    partners: List[Agent] = []

    for i, po in enumerate(process_owners):
        partner = Agent(
            llm=get_model(),
            agent_name=f"Interview Partner {i}",
            system_prompt=get_prompt("interviewer_partner"),
            tools=[],
            default_tool_mode=ToolMode.FORCE,  # tools will be added later
            max_iterations=50,
        )
        link_agents(
            partner,
            [po],
            interactionName="ask_process_owner",
            anonymizePrefix="Process Owner",
            callback=log,
        )
        knowledgeGatherer.add_sub_agent(partner)
        partners.append(partner)

    @add_system_message(
        "Use to_all to send a message to all interviewers and get their answers"
    )
    def to_all(question: str) -> str:
        """Send a message to all interview partners and return their answers"""
        nonlocal partners
        nonlocal log
        nonlocal knowledgeGatherer
        log(knowledgeGatherer.name, "all", question)
        responses = []

        for partner in partners:
            try:
                response = partner.chat(question)
                responses.append(f"{partner.name}: {response}")
            except Exception as e:
                responses.append(f"{partner.name} had an error")

        log("all", knowledgeGatherer.name, "\n".join(responses))
        return "\n".join(responses)

    knowledgeGatherer.add_function_as_tool(to_all)
    manager.add_sub_agent(knowledgeGatherer)
    return manager


def get_process_roles(process: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, f"process.yaml")

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data.get(process)


def get_multiple_process_agents(process: str) -> List[Agent]:
    process_data = get_process_roles(process)

    return [
        Agent(
            llm=Vertex(),
            agent_name=obj["role"],
            system_prompt=get_prompt("process_owner", obj),
            tools=[],
        )
        for obj in process_data
    ]
