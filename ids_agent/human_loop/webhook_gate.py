"""HTTP webhook gate — posts a review request and polls for a response."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ids_agent.human_loop.gate import Decision, HumanGate


class WebhookGate(HumanGate):
    """
    Posts the review prompt to an external HTTP endpoint and polls for
    a decision.  Suitable for a web dashboard or CI/CD integration.
    """

    def __init__(self, endpoint_url: str, poll_interval: float = 5.0) -> None:
        self.endpoint_url = endpoint_url
        self.poll_interval = poll_interval

    async def request(
        self,
        prompt: str,
        options: list[str],
        **context: Any,
    ) -> Decision:
        import aiohttp

        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "prompt": prompt,
            "options": options,
            "context": context,
        }

        async with aiohttp.ClientSession() as session:
            await session.post(f"{self.endpoint_url}/submit", json=payload)

            while True:
                await asyncio.sleep(self.poll_interval)
                async with session.get(
                    f"{self.endpoint_url}/decision/{request_id}"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return Decision(
                            choice=data["choice"],
                            comment=data.get("comment", ""),
                            metadata=data.get("metadata"),
                        )
