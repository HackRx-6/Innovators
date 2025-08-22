#!/usr/bin/env python3
# new_problem_agentic_final_dynamic_refactor.py
import os
import json
import time
import asyncio
from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
import re
import logging
import sys
import warnings

# --- WEB & SCRAPING IMPORTS ---
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# --- LANGGRAPH & ORCHESTRATION IMPORTS ---
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- CONFIGURATION & SAFE DEFAULTS ---
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.setrecursionlimit(10**5)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Config (still from env)
AZURE_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_OPENAI_CHAT_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")

# Global driver instance (main expects this)
_GLOBAL_DRIVER = None

# Config path override: set CONFIG_PATH environment variable to point to a JSON file to override defaults.
CONFIG_PATH = os.getenv("CONFIG_PATH", "")

# Default config (selectors & behavior). Add or override via JSON file if desired.
DEFAULT_CONFIG: Dict[str, Any] = {
    "timeouts": {
        "element_wait": 10,
        "short_wait": 2,
        "long_wait": 20
    },
    "selectors": {
        # Provide ordered fallbacks for start buttons
        "start_buttons": [
            {"css": "button.btn.primary"},
            {"text": "Start Challenge"},
            {"text": "Start"},
            {"text": "Begin"},
            {"css": "button.primary"},
            {"xpath": "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'start')]"}
        ],
        # Hidden/secret element fallbacks
        "secret_element": [
            {"css": "div[data-secret='true']"},
            {"css": "[data-secret='true']"},
            {"xpath": "//div[contains(@data-secret, 'true')]"},
            {"xpath": "//div[contains(., 'secret') or contains(., 'SECRET') or contains(., 'token')]"}
        ],
        # Generic input & submit fallbacks
        "input_field": [
            {"css": "input.input"},
            {"css": "input[type='text']"},
            {"xpath": "//input[contains(@class,'input') or contains(@name,'answer') or contains(@placeholder,'answer')]"}
        ],
        "submit_button": [
            {"css": "button.btn.success"},
            {"text": "Submit"},
            {"text": "Send"},
            {"css": "button[type='submit']"},
            {"xpath": "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'submit')]"}
        ]
    },
    # Extraction options
    "extraction": {
        "page_text_truncate": 4000,  # how many chars to return in get_page_content
        "keyword_regex_template": r"{keyword}[ :\n\r]*([a-zA-Z0-9\-_]+)"
    },
    # Retries/backoff
    "retries": {
        "click_retry": 2,
        "extract_retry": 1
    }
}


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file if provided, otherwise return DEFAULT_CONFIG."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            # shallow merge for simplicity
            config.update(user_cfg)
            logger.info(f"Loaded config from {CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load config from {CONFIG_PATH}: {e}. Falling back to defaults.")
    return config


CONFIG = load_config()


# --- Helper functions (generic, robust find/click/extract) ---

def _resolve_driver(driver_arg: Optional[webdriver.Chrome]) -> webdriver.Chrome:
    """Return provided driver or fallback to global. Raises if none available."""
    if driver_arg:
        return driver_arg
    if _GLOBAL_DRIVER:
        return _GLOBAL_DRIVER
    raise RuntimeError("WebDriver not initialized. Provide a driver or set global _GLOBAL_DRIVER.")


def find_element_by_methods(driver: webdriver.Chrome, methods: List[Dict[str, str]], wait: int) -> Optional[webdriver.remote.webelement.WebElement]:
    """
    Try multiple find strategies described in methods list.
    Each method is dict with one of: 'css', 'xpath', 'text' (text searches buttons/elements).
    Returns the first found WebElement or None.
    """
    for method in methods:
        try:
            if "css" in method:
                selector = method["css"]
                el = WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                return el
            if "xpath" in method:
                selector = method["xpath"]
                el = WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.XPATH, selector)))
                return el
            if "text" in method:
                txt = method["text"]
                # try buttons or clickable elements containing that normalized text (case-insensitive)
                xpath = f"//*[self::button or @role='button' or name()='a'][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{txt.lower()}')]"
                el = WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.XPATH, xpath)))
                return el
        except (TimeoutException, NoSuchElementException, StaleElementReferenceException):
            continue
        except Exception as e:
            logger.debug(f"find_element_by_methods unexpected error for {method}: {e}")
            continue
    return None


def click_element_with_fallbacks(driver: webdriver.Chrome, methods: List[Dict[str, str]], wait: int, click_retry: int = 1) -> str:
    """
    Attempt to find and click element from fallback methods.
    Returns descriptive success/failure string.
    """
    last_err = ""
    for attempt in range(max(1, click_retry)):
        for method in methods:
            try:
                el = find_element_by_methods(driver, [method], wait)
                if not el:
                    continue
                # try to click via standard click; if not clickable, use JS click
                try:
                    WebDriverWait(driver, wait).until(EC.element_to_be_clickable((By.XPATH, get_xpath_from_element(driver, el))))
                    el.click()
                except Exception:
                    # JS fallback
                    driver.execute_script("arguments[0].click();", el)
                return f"Clicked element (method={method})"
            except Exception as e:
                last_err = str(e)
                continue
    return f"Failed to click element after retries. last_error: {last_err}"


def get_xpath_from_element(driver: webdriver.Chrome, element) -> str:
    """
    Helper: Try to compute an XPath for EC.element_to_be_clickable.
    If not possible, return a broad XPath that matches buttons.
    """
    try:
        # Many drivers expose get_attribute or we can use JS to compute path
        return driver.execute_script(
            "function absoluteXPath(element){var comp, comps = []; var parent = null; var xpath = ''; var getPos = function(element){var position = 1, curNode; if (element.nodeType == Node.ATTRIBUTE_NODE) { return null; } for (curNode = element.previousSibling; curNode; curNode = curNode.previousSibling) { if (curNode.nodeName == element.nodeName) { ++position; } } return position; }; if (element instanceof Document) { return '/'; } for (; element && !(element instanceof Document); element = element.nodeType == Node.ATTRIBUTE_NODE ? element.ownerElement : element.parentNode) { comp = ''; switch (element.nodeType) { case Node.TEXT_NODE: comp = 'text()'; break; case Node.ATTRIBUTE_NODE: comp = '@' + element.nodeName; break; case Node.PROCESSING_INSTRUCTION_NODE: comp = 'processing-instruction()'; break; case Node.COMMENT_NODE: comp = 'comment()'; break; case Node.ELEMENT_NODE: comp = element.nodeName; break; } var pos = getPos(element); if (pos > 1) { comp += '[' + pos + ']'; } comps.unshift(comp); } xpath = '/' + comps.join('/'); return xpath; } return absoluteXPath(arguments[0]);",
            element
        )
    except Exception:
        return "//button"  # fallback


def extract_text_with_fallbacks(driver: webdriver.Chrome, methods: List[Dict[str, str]], wait: int, truncate: Optional[int] = None) -> str:
    """
    Try to extract element text by multiple methods. Returns text or descriptive error.
    """
    for method in methods:
        try:
            el = find_element_by_methods(driver, [method], wait)
            if not el:
                continue
            # use JS to get textContent (works for hidden elements too)
            text = driver.execute_script("return arguments[0].textContent;", el)
            if text is None:
                text = el.text
            if not text:
                continue
            text = text.strip()
            if truncate and len(text) > truncate:
                return text[:truncate] + "... (truncated)"
            return text
        except Exception as e:
            logger.debug(f"extract_text_with_fallbacks error for {method}: {e}")
            continue
    return "Error: Could not extract text using provided methods."


# --- AGENT TOOLS (Tool signatures preserved, but logic is dynamic/config-driven) ---

@tool
def navigate_to_url(url: str, driver: Optional[webdriver.Chrome] = None) -> str:
    """Navigates the browser to the specified URL (robust ready-state wait)."""
    logger.info(f"--- 🛠 TOOL: navigate_to_url -> {url} ---")
    try:
        driver = _resolve_driver(driver)
        driver.get(url)
        # wait for ready state or for body to appear
        wait = CONFIG["timeouts"]["element_wait"]
        try:
            WebDriverWait(driver, wait).until(lambda d: d.execute_script("return document.readyState") == "complete")
        except Exception:
            # fallback small sleep if readyState didn't become 'complete' in time
            time.sleep(CONFIG["timeouts"].get("short_wait", 2))
        return f"Successfully navigated to {url}."
    except Exception as e:
        logger.exception("navigate_to_url failed")
        return f"Error navigating to {url}: {e}"


@tool
def get_page_content(driver: Optional[webdriver.Chrome] = None) -> str:
    """Scrapes the entire text content from the current webpage for observation with truncation."""
    logger.info("--- 🛠 TOOL: get_page_content ---")
    try:
        driver = _resolve_driver(driver)
        # Use JS to capture visible + hidden text
        page_content = driver.execute_script("return document.body ? document.body.innerText : document.documentElement.innerText;")
        if not page_content:
            page_content = "The page appears to be empty or content could not be retrieved."
        truncate = CONFIG["extraction"].get("page_text_truncate", 4000)
        return page_content[:truncate] + "... (truncated)" if len(page_content) > truncate else page_content
    except Exception as e:
        logger.exception("get_page_content failed")
        return f"Error getting page content: {e}"


@tool
def find_and_click_start_challenge_button(driver: Optional[webdriver.Chrome] = None) -> str:
    """
    Attempts to find and click a 'Start' button using multiple heuristics from config.
    """
    logger.info("--- 🛠 TOOL: find_and_click_start_challenge_button ---")
    try:
        driver = _resolve_driver(driver)
        methods = CONFIG["selectors"].get("start_buttons", [])
        wait = CONFIG["timeouts"].get("element_wait", 10)
        retries = CONFIG.get("retries", {}).get("click_retry", 2)
        result = click_element_with_fallbacks(driver, methods, wait, click_retry=retries)
        return result
    except Exception as e:
        logger.exception("find_and_click_start_challenge_button failed")
        return f"Error clicking start button: {e}"


@tool
def solve_hidden_element_challenge(driver: Optional[webdriver.Chrome] = None) -> str:
    """
    Attempts to find a hidden/secret element, extract its text, input it into a guess field
    and click submit, using config-driven selectors/fallbacks.
    """
    logger.info("--- 🛠 TOOL: solve_hidden_element_challenge ---")
    try:
        driver = _resolve_driver(driver)
        wait = CONFIG["timeouts"].get("element_wait", 10)
        secret_methods = CONFIG["selectors"].get("secret_element", [])
        input_methods = CONFIG["selectors"].get("input_field", [])
        submit_methods = CONFIG["selectors"].get("submit_button", [])
        # Extract secret text
        secret_text = extract_text_with_fallbacks(driver, secret_methods, wait)
        if secret_text.startswith("Error:"):
            return f"Error: Secret element not found or empty. Detail: {secret_text}"
        # Find input field and send the secret_text
        input_el = find_element_by_methods(driver, input_methods, wait)
        if not input_el:
            return "Error: Could not find input field to submit the secret text."
        try:
            input_el.clear()
            input_el.send_keys(secret_text)
        except Exception:
            try:
                # fallback: set value via JS
                driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input'));", input_el, secret_text)
            except Exception as e:
                logger.debug(f"Failed to send keys or set value via JS: {e}")
                # continue, attempt submit anyway
        # Click submit
        click_result = click_element_with_fallbacks(driver, submit_methods, wait, click_retry=CONFIG.get("retries", {}).get("click_retry", 2))
        # Return success and the secret text (trim small length)
        return f"Success. Secret data was '{secret_text}'. Submit result: {click_result}"
    except Exception as e:
        logger.exception("solve_hidden_element_challenge failed")
        return f"Error during hidden element challenge: {e}"


@tool
def extract_info_from_page(keyword: str, driver: Optional[webdriver.Chrome] = None) -> str:
    """
    (General Purpose Tool) Scrapes the page for a keyword and extracts the following value.
    Uses regex template from config; robust to multi-line.
    """
    logger.info(f"--- 🛠 TOOL: extract_info_from_page(keyword='{keyword}') ---")
    try:
        driver = _resolve_driver(driver)
        page_text = driver.find_element(By.TAG_NAME, 'body').text
        template = CONFIG["extraction"].get("keyword_regex_template", r"{keyword}[ :\n\r]*([a-zA-Z0-9\-_]+)")
        regex = template.replace("{keyword}", re.escape(keyword))
        match = re.search(regex, page_text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"Success! Found '{keyword}': {match.group(1)}"
        else:
            # fallback: find lines that contain keyword and pick the next token
            for line in page_text.splitlines():
                if keyword.lower() in line.lower():
                    # attempt to extract token-like segment
                    tokens = re.findall(r"([A-Za-z0-9\-\_]{3,})", line)
                    if len(tokens) >= 2:
                        # heuristically assume the token is just after the keyword
                        idxs = [i for i, t in enumerate(tokens) if keyword.lower() in t.lower()]
                        if idxs:
                            idx = idxs[0]
                            if idx + 1 < len(tokens):
                                return f"Success! Found '{keyword}': {tokens[idx+1]}"
            return f"Error: Could not find the keyword '{keyword}' on the page."
    except Exception as e:
        logger.exception("extract_info_from_page failed")
        return f"Error extracting info: {e}"


@tool
def answer_question(answer: str) -> str:
    """Provides the final answer to the user's question when the task is complete."""
    logger.info(f"--- ✅ FINAL ANSWER: {answer} ---")
    return answer


# --- CORE AGENT LOGIC (mostly preserved) ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class WebAgent:
    def __init__(self):
        # Use Azure LLM - defaults to env values above
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT_NAME,
            api_version="2024-05-01-preview",
            azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            temperature=0, request_timeout=120
        )
        # Tools preserved (names must match what graph expects)
        self.tools = [
            navigate_to_url,
            get_page_content,
            find_and_click_start_challenge_button,
            solve_hidden_element_challenge,
            extract_info_from_page,
            answer_question
        ]
        # Bind tools to LLM
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        except Exception:
            # graceful fallback if bind_tools isn't available
            self.llm_with_tools = self.llm
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_agent)
        workflow.add_node("tool_executor", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", tools_condition, {"tools": "tool_executor", END: END})
        workflow.add_edge("tool_executor", "agent")
        return workflow.compile()

    def call_agent(self, state: AgentState):
        logger.info("--- AGENT: Thinking... ---")
        system_prompt = """You are an intelligent web automation agent. Your goal is to fulfill the user's request efficiently and safely.

**Your Thought Process:**
-   **Analyze the Goal:** What does the user want me to do? (e.g., "get secret token," "solve challenge").
-   **Plan the Next Step:** Based on my goal and the last tool's output, what single tool call will get me closer to the solution?
    -   If the goal is to get a token, and I've just navigated, my next step is to observe the page content.
    -   If the goal is to solve a challenge, and the page shows a "Start Challenge" button, my next step is to click that button.
    -   If I've just clicked a button and the page content has changed, my next step is to observe the new content.
    -   If the page content contains the information I need, my next step is to use the correct tool to extract it.
-   **Avoid Unnecessary Actions:** Do not call tools like `find_and_click_start_challenge_button` or `solve_hidden_element_challenge` unless the page content clearly indicates that these steps are necessary.
-   **Final Answer:** Once the required information is found, use `answer_question` with the specific piece of data the user requested.
"""
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        # Use LLM with tools if available; otherwise call plain LLM
        try:
            response = self.llm_with_tools.invoke(messages)
        except Exception:
            response = self.llm.invoke(messages)
        return {"messages": [response]}

    async def run(self, query: str):
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_answer = "The agent finished its work, but a final answer was not extracted."

        async for event in self.graph.astream(initial_state, {"recursion_limit": 15}):
            for key, value in event.items():
                if key != "__end__":
                    print(f"--- Executing Node: {key} ---")
                    if key == "tool_executor" and 'messages' in value:
                        if isinstance(value['messages'][-1], ToolMessage):
                            print(f"--- TOOL RESULT ---\n{value['messages'][-1].content}")
                    print("-" * 60)

            if "agent" in event:
                last_message = event["agent"]['messages'][-1]
                if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                    # last tool call might be answer_question
                    if last_message.tool_calls and last_message.tool_calls[-1]['name'] == 'answer_question':
                        final_answer = last_message.tool_calls[-1]['args']['answer']
        return final_answer


# --- Driver setup (kept similar but with logging) ---
def setup_driver() -> webdriver.Chrome:
    print("🚗 Setting up Selenium WebDriver...")
    options = Options()
    # Headless by default; allow override
    headless_env = os.getenv("HEADLESS", "true").lower()
    if headless_env in ("0", "false", "no"):
        pass  # keep headful
    else:
        options.add_argument("--headless=new")  # newer headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Optional user-agent override
    ua = os.getenv("BROWSER_USER_AGENT")
    if ua:
        options.add_argument(f"--user-agent={ua}")
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        return driver
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        raise


# --- Keep `main()` exactly as in your original file (no changes) ---
async def main():
    # --- CHOOSE WHICH QUERY TO RUN ---
    
    # Query 1: The original multi-step challenge
    TARGET_URL = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    QUESTION = f"Please go to {TARGET_URL}, solve the challenge, and tell me the completion code."

    # Query 2: The new, direct token extraction task
    # TARGET_URL = "https://register.hackrx.in/utils/get-secret-token?hackTeam=8070"
    # QUESTION = f"Please go to {TARGET_URL} and get the Secret Token"

    # Query 3: A multi-step challenge asking for the secret data itself
    # TARGET_URL = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    # QUESTION = f"Please go to {TARGET_URL}, solve the challenge, and tell me the secret data."
    
    # Query 4: A multi-step challenge asking for the secret data itself
    # TARGET_URL = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    # QUESTION = f"Please go to {TARGET_URL}, solve the challenge, and tell me the secret data."
    
    global _GLOBAL_DRIVER
    driver = None
    try:
        driver = setup_driver()
        _GLOBAL_DRIVER = driver
        
        agent = WebAgent()
        
        print("\n" + "#"*80 + f"\n❓ User Query: {QUESTION}\n" + "#"*80)
        final_result = await agent.run(QUESTION)
        print("\n" + "="*80 + f"\n✅ Final Result from Agent: {final_result}\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        print("\nCleaning up...")
        if driver:
            driver.quit()


if __name__ == "__main__":
    asyncio.run(main())
