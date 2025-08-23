#!/usr/bin/env python3
"""
Autonomous Agentic AI Workflow for Solving Dynamic Web Challenges
- ReAct-like agent using LangGraph.
- Robust Challenge ID extraction (from DOM/scripts/session/local storage).
- Uses a Custom OpenAI-compatible LLM endpoint (tool calling via OpenAI "tools" spec).
- No bind_tools() needed; we pass tools each invoke.

Run:
  pip install playwright beautifulsoup4 python-dotenv langchain-openai langchain-core langgraph requests
  playwright install
  # Custom LLM (default ON):
  set / export USE_CUSTOM_LLM=1
  set / export CUSTOM_LLM_ENDPOINT="https://register.hackrx.in/llm/openai"
  set / export CUSTOM_LLM_SUBSCRIPTION_KEY="YOUR_SUBSCRIPTION_KEY"
  set / export CUSTOM_LLM_MODEL="gpt-5"
  python just_new.py
"""

import os
import asyncio
import json
import re
from typing import TypedDict, Annotated, List, Optional, Dict

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from playwright.async_api import async_playwright, Page, Browser

# ------------------------- State -------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str

# ------------------------- Heuristics -------------------------
JWT_PATTERN = r"eyJ[\w-]+\.[\w-]+\.[\w-]+"  # generic JWT

def _extract_text_from_scripts(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    blocks = []
    for s in soup.find_all("script"):
        if s.string:
            blocks.append(s.string)
    return "\n".join(blocks)

def _looks_like_token(s: str) -> bool:
    if not s or len(s) < 6:
        return False
    if re.fullmatch(JWT_PATTERN, s):
        return True
    if re.fullmatch(r"[A-Za-z0-9_\-]{12,}\=?\=?", s):
        return True
    return False

def _extract_from_storage_dict(d: Dict[str, str]) -> List[str]:
    candidates = []
    for k, v in (d or {}).items():
        key = (k or "").lower()
        if _looks_like_token(v) or any(x in key for x in ["challenge", "token", "key", "jwt", "id"]):
            if v and len(v) >= 6:
                candidates.append(v)
    return candidates

def heuristic_extract_challenge_id(html: str, session_data: Dict[str, str], local_data: Dict[str, str]) -> Optional[str]:
    html = html or ""
    jwt_html = re.findall(JWT_PATTERN, html)
    if jwt_html:
        return jwt_html[0]
    scripts_text = _extract_text_from_scripts(html)
    jwt_scripts = re.findall(JWT_PATTERN, scripts_text)
    if jwt_scripts:
        return jwt_scripts[0]
    for src in (_extract_from_storage_dict(session_data) + _extract_from_storage_dict(local_data)):
        if re.fullmatch(JWT_PATTERN, src):
            return src
    opaque = (_extract_from_storage_dict(session_data) + _extract_from_storage_dict(local_data))
    if opaque:
        return opaque[0]
    assign_matches = re.findall(r"setItem\(\s*['\"]([^'\"]+)['\"]\s*,\s*(['\"])(.+?)\2\s*\)", html)
    for _key, _q, _val in assign_matches:
        if _looks_like_token(_val):
            return _val
    window = re.findall(r"challenge[\w\s\-:]*?([A-Za-z0-9_\-]{10,})", html, flags=re.IGNORECASE)
    for w in window:
        if _looks_like_token(w):
            return w
    return None

# ------------------------- Browser -------------------------
class WebBrowser:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright = None

    async def initialize(self):
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        print("✅ Browser initialized.")

    async def close(self):
        try:
            if self.browser:
                await self.browser.close()
            if self._playwright:
                await self._playwright.stop()
            print("✅ Browser closed.")
        except Exception:
            pass

    async def navigate(self, url: str) -> str:
        if not self.page:
            return "Browser not initialized."
        try:
            await self.page.goto(url, timeout=45000, wait_until='networkidle')
            return f"Successfully navigated to {url}."
        except Exception as e:
            return f"Navigation failed: {str(e)}"

    async def get_page_snapshot(self) -> dict:
        if not self.page:
            return {"error": "Browser not initialized."}
        try:
            await self.page.evaluate("""() => {
                const interactiveElements = document.querySelectorAll('a, button, input, textarea, select');
                interactiveElements.forEach((el, index) => {
                    el.setAttribute('data-agent-id', `agent-${index}`);
                });
            }""")
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup.find_all(['script', 'style']):
                tag.decompose()
            interactive_elements = []
            elements = await self.page.query_selector_all('a, button, input, textarea, select')
            for el in elements:
                agent_id = await el.get_attribute('data-agent-id')
                tag_name = await el.evaluate('e => e.tagName.toLowerCase()')
                text = await el.text_content()
                attrs = await el.evaluate('e => { const a = {}; for (const at of e.attributes) { a[at.name] = at.value; } return a; }')
                interactive_elements.append({
                    "id": agent_id,
                    "tag": tag_name,
                    "text": (text or "").strip(),
                    "attributes": {k: v for k, v in (attrs or {}).items() if k in ['type', 'placeholder', 'name', 'value', 'aria-label']}
                })
            return {
                "simplified_html": str(soup.body)[:8000],
                "interactive_elements": interactive_elements
            }
        except Exception as e:
            return {"error": f"Failed to get page snapshot: {str(e)}"}

    async def get_raw_html(self) -> str:
        if not self.page:
            return ""
        try:
            return await self.page.content()
        except Exception:
            return ""

    async def get_session_storage(self) -> Dict[str, str]:
        if not self.page:
            return {}
        try:
            return await self.page.evaluate("""() => {
                const s = {}; for (let i=0;i<sessionStorage.length;i++){ const k=sessionStorage.key(i); s[k]=sessionStorage.getItem(k);} return s;
            }""")
        except Exception:
            return {}

    async def get_local_storage(self) -> Dict[str, str]:
        if not self.page:
            return {}
        try:
            return await self.page.evaluate("""() => {
                const s = {}; for (let i=0;i<localStorage.length;i++){ const k=localStorage.key(i); s[k]=localStorage.getItem(k);} return s;
            }""")
        except Exception:
            return {}

    async def click_element(self, element_id: str) -> str:
        if not self.page:
            return "Browser not initialized."
        try:
            selector = f'[data-agent-id="{element_id}"]'
            await self.page.click(selector, timeout=5000)
            await self.page.wait_for_load_state('networkidle', timeout=10000)
            return f"Successfully clicked element '{element_id}'."
        except Exception as e:
            return f"Failed to click element '{element_id}': {str(e)}"

    async def type_text(self, element_id: str, text: str) -> str:
        if not self.page:
            return "Browser not initialized."
        try:
            selector = f'[data-agent-id="{element_id}"]'
            await self.page.fill(selector, text, timeout=5000)
            return f"Successfully typed '{text}' into element '{element_id}'."
        except Exception as e:
            return f"Failed to type into element '{element_id}': {str(e)}"

browser_manager = WebBrowser()

# ------------------------- Tools (LangChain) -------------------------
@tool
async def get_page_snapshot() -> str:
    """Take a snapshot of the current page, including simplified HTML and interactive elements."""
    snapshot = await browser_manager.get_page_snapshot()
    return json.dumps(snapshot)

@tool
async def click_element(element_id: str) -> str:
    """Click an interactive element by its assigned agent-id."""
    return await browser_manager.click_element(element_id)

@tool
async def type_text(element_id: str, text: str) -> str:
    """Type the provided text into the specified input element by agent-id."""
    return await browser_manager.type_text(element_id, text)

@tool
def submit_completion_code(completion_code: str) -> str:
    """Submit the discovered completion code back to the system."""
    return json.dumps({"completion_code": completion_code})

TOOLS = [get_page_snapshot, click_element, type_text, submit_completion_code]

# OpenAI-style tool specs we pass to the model for tool calling
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_page_snapshot",
            "description": "Take a snapshot of the current page, including simplified HTML and interactive elements.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click an interactive element by its assigned agent-id.",
            "parameters": {
                "type": "object",
                "properties": {"element_id": {"type": "string"}},
                "required": ["element_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type the provided text into the specified input element by agent-id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["element_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_completion_code",
            "description": "Submit the discovered completion code back to the system.",
            "parameters": {
                "type": "object",
                "properties": {"completion_code": {"type": "string"}},
                "required": ["completion_code"],
            },
        },
    },
]

# ------------------------- Custom OpenAI-Compatible ChatModel -------------------------
class CustomOpenAICompatibleLLM(BaseChatModel):
    """
    Minimal OpenAI-compatible chat model that posts to a single endpoint.
    Passes 'tools' and 'tool_choice' straight through.
    """
    endpoint: str
    subscription_key: str
    model: str = "gpt-4.1-mini"
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        return "custom_openai_compatible"

    def _to_openai_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        out = []
        for m in messages:
            if isinstance(m, SystemMessage):
                out.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                out.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                item: Dict = {"role": "assistant", "content": m.content}
                if getattr(m, "tool_calls", None):
                    tc_list = []
                    for tc in m.tool_calls:
                        fn_args = tc.get("args", {})
                        if isinstance(fn_args, dict):
                            fn_args = json.dumps(fn_args)
                        tc_list.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {"name": tc.get("name"), "arguments": fn_args}
                        })
                    item["tool_calls"] = tc_list
                out.append(item)
            elif isinstance(m, ToolMessage):
                out.append({"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def _post(self, payload: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "x-subscription-key": self.subscription_key,
        }
        resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _parse_choice(self, data: Dict) -> AIMessage:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if content is None:
            content = ""
        elif isinstance(content, list):
            try:
                content = "".join(
                    [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
                )
            except Exception:
                content = str(content)

        tool_calls = msg.get("tool_calls") or []
        ai = AIMessage(content=content)
        if tool_calls:
            parsed = []
            for tc in tool_calls:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {"_raw": args_raw}
                parsed.append({"name": name, "args": args, "id": tc.get("id")})
            ai.tool_calls = parsed
        return ai

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        payload: Dict = {"messages": self._to_openai_messages(messages), "model": self.model}
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        if stop:
            payload["stop"] = stop
        data = self._post(payload)
        ai_message = self._parse_choice(data)
        gen = ChatGeneration(message=ai_message, generation_info={"finish_reason": (data.get("choices") or [{}])[0].get("finish_reason")})
        return ChatResult(generations=[gen])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._generate(messages, stop, **kwargs))

# ------------------------- Agent -------------------------
class WebChallengeAgent:
    def __init__(self, llm, tools, openai_tools_spec):
        self.llm = llm
        self.tools = tools
        self.openai_tools_spec = openai_tools_spec  # what we pass to the model

    def _get_system_prompt(self, task: str) -> str:
        return f"""
You are an expert web automation agent. Your goal is to solve an interactive web challenge to find a single piece of information: the 'completion_code'.

THE OVERALL TASK IS: "{task}"

Workflow:
1) Call get_page_snapshot to understand the page.
2) Find and click a button like "Start Challenge" via click_element.
3) After the challenge starts, use get_page_snapshot again. A secret value may appear in DOM.
4) Locate a text input and a submit button. Enter the secret using type_text and then click_element.
5) After submitting, use get_page_snapshot once more to view the success screen.
6) Extract the final Completion Code from the success screen.
7) When you have it, call submit_completion_code to finish.

Constraints:
- Do not guess element ids. Use ids provided by get_page_snapshot.
- Only aim to retrieve and submit the completion code. Do not chat.
"""

    async def run_agent(self, state: AgentState):
        messages = state['messages']
        task = state['task']
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self._get_system_prompt(task))] + messages

        # IMPORTANT: pass tools every time (no bind_tools)
        response = await self.llm.ainvoke(messages, tools=self.openai_tools_spec, tool_choice="auto")
        return {"messages": [response]}

    def should_continue(self, state: AgentState) -> str:
        last_message = state['messages'][-1]
        if getattr(last_message, "tool_calls", None):
            if any(call['name'] == 'submit_completion_code' for call in last_message.tool_calls):
                return "end"
            return "continue"
        return "end"

# ------------------------- Workflow -------------------------
async def create_workflow(url: str, questions: List[str]) -> dict:
    print("🔑 Searching for Challenge ID (JWT) in the URL...")
    jwt_in_url = None
    m = re.search(JWT_PATTERN, url)
    if m:
        jwt_in_url = m.group(0)
        print("   ✅ Found Challenge ID (JWT) in URL.")
    else:
        print("   ℹ️  No JWT-like token in URL; will derive it from page after completion.")

    await browser_manager.initialize()

    llm = CustomOpenAICompatibleLLM(
        endpoint=os.getenv("CUSTOM_LLM_ENDPOINT", "https://register.hackrx.in/llm/openai"),
        subscription_key="sk-spgw-api01-b75c065b5d321f07d09e035766fb9330",
        model="gpt-4.1-mini",
        timeout=int(os.getenv("CUSTOM_LLM_TIMEOUT", "60")),
    )

    agent = WebChallengeAgent(llm, TOOLS, OPENAI_TOOLS)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent.run_agent)
    workflow.add_node("action", ToolNode(TOOLS))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", agent.should_continue, {"continue": "action", "end": END})
    workflow.add_edge("action", "agent")
    app = workflow.compile()

    task_description = (
        "Go to the website, start the challenge, complete all steps, and find the final 'Completion Code'. "
        f"The initial URL is: {url}"
    )

    initial_state: AgentState = {
        "task": task_description,
        "messages": [HumanMessage(content=f"Let's begin. The starting URL is {url}. Please navigate there and find the completion code.")],
    }

    nav_result = await browser_manager.navigate(url)
    print(f"➡️  Initial Navigation: {nav_result}")

    final_state = None
    try:
        print("🚀 Starting autonomous agent workflow...")
        async for output in app.astream(initial_state, {"recursion_limit": 20}):
            for key, value in output.items():
                if key == "agent" and getattr(value['messages'][-1], "tool_calls", None):
                    for call in value['messages'][-1].tool_calls:
                        print(f"🧠 Agent wants to run: {call['name']}({json.dumps(call['args'])})")
                elif key == "action":
                    print(f"🛠️  Tool Result: {value['messages'][-1].content}")
            final_state = output
        print("✅ Workflow finished.")

        completion_code = None
        last_msg = final_state['agent']['messages'][-1]
        if getattr(last_msg, "tool_calls", None) and last_msg.tool_calls[0]['name'] == 'submit_completion_code':
            completion_code = last_msg.tool_calls[0]['args'].get('completion_code')

        challenge_id = jwt_in_url
        if not challenge_id:
            raw_html = await browser_manager.get_raw_html()
            sess = await browser_manager.get_session_storage()
            loc = await browser_manager.get_local_storage()
            challenge_id = heuristic_extract_challenge_id(raw_html, sess, loc) or None

        answers = {}

        for idx, question in enumerate(questions):
            key = f"question_{idx + 1}"
            if idx == 0:
                answers[key] = {
                    "question": question,
                    "answer": challenge_id,
                }
            elif idx == 1:
                answers[key] = {
                    "question": question,
                    "answer": completion_code,
                }
        return answers

    finally:
        await browser_manager.close()

# ------------------------- Main -------------------------
async def web():

    payload ={"url": "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ==",
        "questions": [
            "Go to the website and start the challenge. Complete the challenge and return the answers for the following question? What is the challenge ID?",
            "Go to the website and start the challenge. Complete the challenge and return the answers for the following question? What is the completion code?",]}

    print("=" * 80)
    print("🤖 AUTONOMOUS AGENTIC WEB CHALLENGE SOLVER")
    print("=" * 80)

    loop = asyncio.get_event_loop()
    start_time = loop.time()

    try:
        results = await create_workflow(url=payload["url"], questions=payload["questions"])
        execution_time = loop.time() - start_time

        print("\n" + "=" * 80)
        print("🎉 FINAL RESULTS:")
        print("=" * 80)

        for key, result in results.items():
            print(f"\n{key.upper()}:")
            print(f"  Question: {result['question']}")
            print(f"  Answer:   {result['answer']}")

        print(f"\n⏱️  Total Execution time: {execution_time:.2f} seconds")
        print("=" * 80)
        return results

    except Exception as e:
        import traceback
        print("❌ Main execution error:")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Silence Windows "Event loop is closed" noise during shutdown.
    try:
        asyncio.run(web())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            pass
        else:
            raise
