"""CLI-based human review gate using Rich for formatting."""

from __future__ import annotations

import asyncio
from typing import Any

from ids_agent.human_loop.gate import Decision, HumanGate


class CLIGate(HumanGate):
    """
    Blocks until the operator types a choice in the terminal.
    Uses asyncio.to_thread so it doesn't block the event loop.
    """

    async def request(
        self,
        prompt: str,
        options: list[str],
        **context: Any,
    ) -> Decision:
        choice = await asyncio.to_thread(self._blocking_prompt, prompt, options)
        comment = await asyncio.to_thread(
            input, "Optional comment (press Enter to skip): "
        )
        return Decision(choice=choice, comment=comment, metadata=context)

    @staticmethod
    def _blocking_prompt(prompt: str, options: list[str]) -> str:
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel(prompt, title="[bold yellow]Human Review Required[/]"))
        except ImportError:
            print("\n" + "=" * 60)
            print(prompt)
            print("=" * 60)

        option_str = " / ".join(f"[{o}]" for o in options)
        while True:
            raw = input(f"\nChoose {option_str}: ").strip().lower()
            if raw in options:
                return raw
            print(f"  Invalid choice. Please enter one of: {options}")
