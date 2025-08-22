import os
import json
import time
import asyncio
from typing import Annotated, Sequence, TypedDict
import re
import logging
import base64

# --- WEB & SCRAPING IMPORTS ---
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- LANGGRAPH & ORCHESTRATION IMPORTS ---
import sys
import warnings
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- CONFIGURATION ---
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.setrecursionlimit(10**5)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Config
AZURE_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_OPENAI_CHAT_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")

# Global driver instance for tools to access
_GLOBAL_DRIVER = None

# --- AUTONOMOUS AGENT TOOLS ---

@tool
def navigate_to_url(url: str) -> str:
    """Navigates the browser to the specified URL."""
    print(f"--- 🛠 TOOL: Navigating to {url} ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER:
        return "Error: WebDriver is not initialized."
    try:
        _GLOBAL_DRIVER.get(url)
        time.sleep(2) # Allow time for the page to load
        return f"Successfully navigated to {url}. Now at {_GLOBAL_DRIVER.current_url}"
    except Exception as e:
        return f"Error navigating to {url}: {e}"

@tool
def scrape_page_content() -> str:
    """Scrapes and returns the visible text content and interactive elements of the current webpage."""
    print("--- 🛠 TOOL: scrape_page_content ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER:
        return "Error: No webdriver available"
    try:
        page_source = _GLOBAL_DRIVER.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        for element in soup(["script", "style", "nav", "footer", "aside", "header"]):
            element.decompose()
        
        clean_text = soup.get_text(separator='\n', strip=True)
        
        if len(clean_text) > 8000:
            clean_text = clean_text[:8000] + "\n... [Content truncated]"
        
        return clean_text
    except Exception as e:
        return f"Error scraping page: {str(e)}"

@tool
def extract_tokens() -> str:
    """Extracts potential tokens, codes, or unique identifiers (including JWTs) from the page source."""
    print("--- 🛠 TOOL: extract_tokens ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER:
        return "Error: No webdriver available"
    try:
        page_source = _GLOBAL_DRIVER.page_source
        token_patterns = [
            r'ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
            r'[A-Fa-f0-9]{32}', r'[A-Fa-f0-9]{40}',
            r'[A-Za-z0-9_.-]{20,}', r'[A-Z0-9]{8,}',
            r'\b[A-Z]{2,}[0-9]{4,}\b'
        ]
        found_tokens = []
        for pattern in token_patterns:
            matches = re.findall(pattern, page_source)
            found_tokens.extend(matches)
        
        unique_tokens = list(set(found_tokens))
        if unique_tokens:
            return f"Found potential tokens: {unique_tokens[:10]}"
        else:
            return "No potential tokens found on the page."
    except Exception as e:
        return f"Error extracting tokens: {str(e)}"

@tool
def decode_jwt(token: str) -> str:
    """Decodes the payload of a JSON Web Token (JWT) to inspect its contents."""
    print(f"--- 🛠 TOOL: decode_jwt ---")
    try:
        _, payload, _ = token.split('.')
        decoded_payload = base64.urlsafe_b64decode(payload + '==')
        payload_json = json.loads(decoded_payload.decode('utf-8'))
        return f"Successfully decoded JWT payload: {json.dumps(payload_json, indent=2)}"
    except Exception as e:
        return f"Error: Invalid token format or decoding failed. Reason: {str(e)}"

@tool
def input_text(selector: str, value: str) -> str:
    """Inputs text into a field on the current page using its CSS selector."""
    print(f"--- 🛠 TOOL: input_text(selector='{selector}', value='{value}') ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER:
        return "Error: No webdriver available"
    try:
        element = WebDriverWait(_GLOBAL_DRIVER, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        element.clear()
        element.send_keys(value)
        return f"Successfully input text into field with selector '{selector}'."
    except Exception as e:
        return f"Error: Failed to input text into '{selector}'. Reason: {e}"

@tool
def click_element(selector: str) -> str:
    """Clicks an element on the current page using a CSS selector."""
    print(f"--- 🛠 TOOL: click_element(selector='{selector}') ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER:
        return "Error: No webdriver available"
    try:
        element = WebDriverWait(_GLOBAL_DRIVER, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
        )
        _GLOBAL_DRIVER.execute_script("arguments[0].click();", element)
        time.sleep(3)
        return f"Successfully clicked element with selector '{selector}'."
    except Exception as e:
        return f"Error: Failed to click element '{selector}'. Reason: {e}"

@tool
def answer_question(answer: str) -> str:
    """Provides the final answer to the user's question when the task is complete."""
    print(f"--- ✅ FINAL ANSWER: {answer} ---")
    return answer

# --- CORE AGENT LOGIC ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class WebAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT_NAME,
            api_version="2024-05-01-preview",
            azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            temperature=0,
            request_timeout=120
        )
        self.tools = [
            navigate_to_url,
            scrape_page_content,
            extract_tokens,
            decode_jwt,
            input_text,
            click_element,
            answer_question
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_agent)
        workflow.add_node("tool_executor", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tool_executor", END: END},
        )
        workflow.add_edge("tool_executor", "agent")
        return workflow.compile()

    def call_agent(self, state: AgentState):
        print("--- AGENT: Thinking... ---")
        system_prompt = """You are a highly skilled web automation assistant.
Your goal is to solve the user's query by using tools to interact with a webpage.

**Strategy:**
1.  **Start with the URL**: If the user provides a URL, use `navigate_to_url` first.
2.  **Scrape and Understand**: Always use `scrape_page_content` to understand the current state of the page.
3.  **Analyze and Plan**: Based on the scraped content and the user's goal, decide the next logical step.
4.  **Execute**: Use the appropriate tool (`click_element`, `input_text`, etc.) to perform the action.
5.  **Look for Tokens**: If the goal is to find a code or secret, use `extract_tokens`.
6.  **Loop**: Repeat steps 2-4 until the task is complete.
7.  **Answer**: When you have found the final answer, you **MUST** use the `answer_question` tool to provide it. This is your final step.
Run Look for Tokens last time after whole task has been completed to make sure the no hidden code or secret or token is present

**Important Rules:**
-   Be methodical. Do not guess.
-   Always scrape after navigating or clicking to see the result of your action.
-   Provide clear and simple selectors for elements (e.g., `button.primary`, `input[name='email']`).
-   Your final output must be a call to the `answer_question` tool."""
        
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    async def run(self, query: str):
        initial_state = {"messages": [HumanMessage(content=query)]}
        final_answer = "The agent finished its work, but a final answer was not extracted."
        
        async for event in self.graph.astream(initial_state, {"recursion_limit": 50}):
            for key, value in event.items():
                if key != "end":
                    print(f"--- Executing Node: {key} ---")
                    if key == "tool_executor" and 'messages' in value:
                        last_message = value['messages'][-1]
                        if isinstance(last_message, ToolMessage):
                            print(f"--- TOOL RESULT ---\n{last_message.content}")
                    print("-" * 60)
            
            if "agent" in event:
                last_message = event["agent"]['messages'][-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    # Check if the last tool call is answer_question
                    if last_message.tool_calls[-1]['name'] == 'answer_question':
                        final_answer = last_message.tool_calls[-1]['args']['answer']
                        break # End the loop as we have the final answer
        
        return final_answer

def setup_driver():
    print("🚗 Setting up Selenium WebDriver...")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    try:
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), 
            options=options
        )
        return driver
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        raise

async def main():
    TARGET_URL = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    QUESTION = f"First, go to the URL '{TARGET_URL}'. Then, start the challenge, complete it, and find the challenge ID."

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