import asyncio
import os
from typing import Any

from .config import OPENAI_API_KEY

class LLMResponse:
    def __init__(self, content: str):
        self.content = content

class LLMClient:
    async def ainvoke(self, prompt: Any) -> LLMResponse:
        raise NotImplementedError()

    def bind_tools(self, tools: list):
        return self

class MockLLM(LLMClient):
    """Deterministic mock used for local testing."""
    async def ainvoke(self, prompt):
        text = ""
        if isinstance(prompt, list):
            try:
                first = prompt[0]
                text = first.content if hasattr(first, "content") else str(prompt)
            except:
                text = str(prompt)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
        await asyncio.sleep(0.05)
        lower = text.lower()

        # routing decision
        if "which agent is the most appropriate" in lower or "decision:" in lower:
            if any(k in lower for k in ["flight", "call the api", "go to the link", "token"]):
                return LLMResponse("set_goal_agent")
            return LLMResponse("rag_agent")

        # json plan generator
        if "create a json plan" in lower or "json plan" in lower:
            sample = {
                "goal": "Find the flight number.",
                "steps": [
                    {"step": 1, "description": "Call the API at https://register.hackrx.in/submissions/myFavouriteCity to find the city name."},
                    {"step": 2, "description": "Find the landmark for that city in the document."},
                    {"step": 3, "description": "Based on the landmark, call the appropriate flight endpoint described in the document."}
                ]
            }
            import json as _json
            return LLMResponse(_json.dumps(sample))

        # perform_action heuristic
        if "execute this task" in lower or "if the task requires calling an api" in lower:
            import re
            m = re.search(r"https?://[^\s'\",]+", text)
            if m:
                url = m.group(0)
                return LLMResponse(f"TOOL_CALL: {url}")
            return LLMResponse('Found the flight number is "eb465a".')

        # final synthesis
        if "synthesize final" in lower or "final answer" in lower:
            return LLMResponse('Your flight number is "eb465a".')

        return LLMResponse("I cannot handle that request in mock mode.")

# Optional: placeholder OpenAI client (fill in per provider docs)
class OpenAIClient(LLMClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        # Implement initialization as needed (openai package or new api)
    async def ainvoke(self, prompt):
        # Implement your provider call here. This is intentionally a stub.
        raise NotImplementedError("OpenAIClient isn't implemented in this template. Replace with your adapter.")
