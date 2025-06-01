from typing import Callable, Literal, Union, Optional, Any, Type, List, Dict, Annotated, get_type_hints, get_origin, get_args
from abc import ABC

class Tracker(ABC):
    _all_trackers: List["Tracker"] = []
    
    @classmethod
    def collectTokenStats(cls) -> Dict[str, int]:
        stats = {
            "total_invocations": 0,
            "total_cumulative_completion_tokens": 0,
            "total_cumulative_prompt_tokens": 0
        }
        for tracker in cls._all_trackers:
            stats["total_invocations"] += tracker._invocation
            stats["total_cumulative_completion_tokens"] += tracker._cumulative_completion_tokens
            stats["total_cumulative_prompt_tokens"] += tracker._cumulative_prompt_tokens
        return stats

    def __init__(self):
        from agentFramework.agent import Agent
        self.agent: Optional['Agent'] = None

        self._invocation = 0
        self._cumulative_completion_tokens = 0
        self._cumulative_prompt_tokens = 0

        Tracker._all_trackers.append(self)

    def setAgent(self, agent):
        self.agent = agent

    def _agentName(self) -> str:
        if self.agent is None:
            return "NoAgentSet"
        return self.agent.name

    def logTokens(self, prompt: int, completion: int, total: int): 
        self._invocation += 1
        self._cumulative_completion_tokens += completion
        self._cumulative_prompt_tokens += prompt

    def logMetadata(self, metadata: Dict[str, Any]):
        pass

import mlflow

class MLFlowTracker(Tracker):
    def logTokens(self, prompt: int, completion: int, total: int):
        super().logTokens(prompt, completion, total)
        span = mlflow.get_current_active_span()
        if span:
            span.set_attributes({
                'prompt_token_count': prompt,
                'completion_token_count': completion,
                'total_token_count': total
            })

        agent_name = self._agentName()
        mlflow.log_metric(f"{agent_name}:completion_tokens", completion, step=self._invocation)
        mlflow.log_metric(f"{agent_name}:cumulative_completion_tokens", self._cumulative_completion_tokens, step=self._invocation)
        mlflow.log_metric(f"{agent_name}:prompt_tokens", prompt, step=self._invocation)
        mlflow.log_metric(f"{agent_name}:cumulative_prompt_tokens", self._cumulative_prompt_tokens, step=self._invocation)

    def logMetadata(self, metadata: Dict[str, Any]):
        span = mlflow.get_current_active_span()
        if span:
            span.set_attributes(metadata)
    
    @classmethod
    def collectTokenStats(cls) -> Dict[str, int]:
        stats = super().collectTokenStats()
        mlflow.log_metrics(stats)
        return stats

class LoggingTracker(Tracker):
    def __init__(self, log_tokens: bool = True, log_metadata: bool = False):
        super().__init__()
        self.log_tokens = log_tokens
        self.log_metadata = log_metadata

    def logTokens(self, prompt: int, completion: int, total: int):
        super().logTokens(prompt, completion, total)
        if self.log_tokens:
            print(f"Prompt tokens: {prompt}, Completion tokens: {completion}, Total tokens: {total}")

    def logMetadata(self, metadata: Dict[str, Any]):
        if self.log_metadata:
            print(f"Metadata: {metadata}")
