from enum import Enum
from agentFramework.agent import Agent
from typing import List

class Connection(str, Enum):
    DELIGATION = "delegation"

class Team:
    def __init__(self):
        self.members: List[Agent] = []
        self.connections = []
        self.history = []


    def add_member(self, member: Agent):
        self.members.append(member)

    def add_connection(self, member1: Agent, connection: Connection, member2: Agent):
        if member1 not in self.members:
            self.add_member(member1)
        if member2 not in self.members:
            self.add_member(member2)
        
        self.connections.append((member1, connection, member2))

    def kickoff(self, start: Agent, prompt: str):
        """Starts the team with the given entry agent and prompt"""
        if start not in self.members:
            raise ValueError("Entry agent is not a member of the team")
        
        start.chat(prompt)
