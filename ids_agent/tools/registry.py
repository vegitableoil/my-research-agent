"""Tool registry — decorator-based registration and lookup."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


class ToolRegistry:
    """
    Central registry for all agent tools.

    Usage
    -----
    Register:
        @tool(name="my_tool", tags=["literature"])
        async def my_tool(arg: str) -> dict: ...

    Call from an agent:
        result = await self.tools.call("my_tool", arg="value")
    """

    _global_registry: dict[str, Callable] = {}

    def __init__(self, tools: dict[str, Callable] | None = None) -> None:
        self._tools: dict[str, Callable] = dict(self._global_registry)
        if tools:
            self._tools.update(tools)

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    async def call(self, name: str, **kwargs: Any) -> Any:
        fn = self._tools.get(name)
        if fn is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, lambda: fn(**kwargs))

    @classmethod
    def _register_global(cls, name: str, fn: Callable) -> None:
        cls._global_registry[name] = fn


def tool(name: str, tags: list[str] | None = None) -> Callable:
    """Decorator to register a function as a globally available tool."""
    def decorator(fn: Callable) -> Callable:
        ToolRegistry._register_global(name, fn)
        fn._tool_name = name
        fn._tool_tags = tags or []
        return fn
    return decorator


def register_agent(name: str) -> Callable:
    """Decorator to register an agent class in the global agent registry."""
    from ids_agent.orchestrator import AgentRegistry  # lazy import
    def decorator(cls: type) -> type:
        AgentRegistry.register(name, cls)
        return cls
    return decorator
