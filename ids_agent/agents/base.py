"""Abstract base class shared by every agent in the system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ids_agent.rag.retriever import RAGRetriever
from ids_agent.tools.registry import ToolRegistry


@dataclass
class AgentContext:
    """Shared mutable state passed between agents within one pipeline run."""

    run_id: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    All agents inherit from this class.

    Subclasses must implement `run(ctx)`.  They receive a shared `AgentContext`
    and return an updated copy of its `data` dict.  Agents should not mutate
    ctx.data directly — return a new dict and let the orchestrator merge.
    """

    #: Override in subclass to declare which named RAG indices this agent reads.
    rag_indices: list[str] = []

    def __init__(
        self,
        tool_registry: ToolRegistry,
        rag_retriever: RAGRetriever,
    ) -> None:
        self.tools = tool_registry
        self.rag = rag_retriever
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        """Execute the agent's task and return result data."""

    async def retrieve(self, query: str, index: str, k: int = 5) -> list[dict]:
        """Convenience wrapper for RAG retrieval."""
        return await self.rag.retrieve(query=query, index=index, k=k)
