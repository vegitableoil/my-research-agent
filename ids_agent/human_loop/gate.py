"""Human-in-the-Loop gate abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Decision:
    choice: str                    # one of the supplied `options`
    comment: str = ""
    metadata: dict[str, Any] | None = None


class HumanGate(ABC):
    """
    Blocks the pipeline until a human makes a decision.

    Subclasses implement `request()` using different I/O backends
    (CLI, webhook, Slack, etc.) — the rest of the system only
    depends on this interface.
    """

    @abstractmethod
    async def request(
        self,
        prompt: str,
        options: list[str],
        **context: Any,
    ) -> Decision:
        """Present `prompt` to the human and return their Decision."""
