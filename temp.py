#!/usr/bin/env python3
# new_problem_agentic_final_dynamic.py
import os
import json
import time
import asyncio
from typing import Annotated, Sequence, TypedDict
import re
import logging

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
from langchain_core.tools import tool
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

# --- AGENT TOOLS (Refined for Dynamic Use) ---

@tool
def navigate_to_url(url: str) -> str:
    """Navigates the browser to the specified URL."""
    print(f"--- 🛠 TOOL: Navigating to {url} ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER: return "Error: WebDriver is not initialized."
    try:
        _GLOBAL_DRIVER.get(url)
        time.sleep(2)
        return f"Successfully navigated to {url}."
    except Exception as e:
        return f"Error navigating to {url}: {e}"

@tool
def get_page_content() -> str:
    """Scrapes the entire text content from the current webpage for observation."""
    print("--- 🛠 TOOL: get_page_content ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER: return "Error: WebDriver is not initialized."
    try:
        # Use innerHTML to capture hidden content as well
        page_content = _GLOBAL_DRIVER.execute_script("return document.body.innerText;")
        if not page_content:
            page_content = "The page appears to be empty or content could not be retrieved."
        return page_content[:4000] + "... (truncated)" if len(page_content) > 4000 else page_content
    except Exception as e:
        return f"Error getting page content: {e}"

@tool
def find_and_click_start_challenge_button() -> str:
    """
    (Specialized Tool) Finds and clicks the 'Start Challenge' button on the page.
    Should only be used when the page content explicitly mentions a button to start a challenge.
    """
    print("--- 🛠 TOOL: find_and_click_start_challenge_button ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER: return "Error: WebDriver is not initialized."
    try:
        selector = "button.btn.primary"
        element = WebDriverWait(_GLOBAL_DRIVER, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
        )
        element.click()
        time.sleep(2)
        return "Successfully clicked the 'Start Challenge' button."
    except Exception as e:
        return f"Error clicking start button: Could not find button with selector '{selector}'. Reason: {e}"

@tool
def solve_hidden_element_challenge() -> str:
    """
    (Specialized Tool) Solves the 'Hidden Element' challenge. It finds the hidden
    element, inputs its text into a field, clicks 'Submit', and returns the
    text of the hidden element.
    """
    print("--- 🛠 TOOL: solve_hidden_element_challenge ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER: return "Error: WebDriver is not initialized."
    try:
        secret_element = WebDriverWait(_GLOBAL_DRIVER, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-secret='true']"))
        )
        
        # Use JavaScript to get text content, which works for hidden elements
        secret_text = _GLOBAL_DRIVER.execute_script("return arguments[0].textContent;", secret_element)
        if not secret_text:
            return "Error: Found the secret element, but it contains no text via JavaScript."

        input_field = WebDriverWait(_GLOBAL_DRIVER, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input.input"))
        )
        input_field.send_keys(secret_text)
        
        submit_button = WebDriverWait(_GLOBAL_DRIVER, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.success"))
        )
        submit_button.click()
        time.sleep(2)
        
        return f"Success. Secret data was '{secret_text}'."
    except Exception as e:
        return f"Error during hidden element challenge: {e}"

@tool
def extract_info_from_page(keyword: str) -> str:
    """
    (General Purpose Tool) Scrapes the page for a keyword and extracts the following value.
    This is best for simple pages with key-value pairs.
    """
    print(f"--- 🛠 TOOL: extract_info_from_page(keyword='{keyword}') ---")
    global _GLOBAL_DRIVER
    if not _GLOBAL_DRIVER: return "Error: WebDriver is not initialized."
    try:
        page_text = _GLOBAL_DRIVER.find_element(By.TAG_NAME, 'body').text
        # Regex to find the keyword followed by a colon or space, then capture the value
        match = re.search(f"{re.escape(keyword)}[ :\\n\\r]*([a-zA-Z0-9\\-_]+)", page_text, re.IGNORECASE | re.DOTALL)
        if match:
            return f"Success! Found '{keyword}': {match.group(1)}"
        else:
            return f"Error: Could not find the keyword '{keyword}' on the page."
    except Exception as e:
        return f"Error extracting info: {e}"

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
            temperature=0, request_timeout=120
        )
        self.tools = [
            navigate_to_url,
            get_page_content,
            find_and_click_start_challenge_button,
            solve_hidden_element_challenge,
            extract_info_from_page,
            answer_question
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
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
        print("--- AGENT: Thinking... ---")
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
        response = self.llm_with_tools.invoke(messages)
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
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    if last_message.tool_calls[-1]['name'] == 'answer_question':
                        final_answer = last_message.tool_calls[-1]['args']['answer']
        return final_answer

def setup_driver():
    print("🚗 Setting up Selenium WebDriver...")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        return driver
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        raise

async def main():
    # --- CHOOSE WHICH QUERY TO RUN ---
    
    # Query 1: The original multi-step challenge
    # TARGET_URL = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    # QUESTION = f"Please go to {TARGET_URL}, solve the challenge, and tell me the completion code."

    # Query 2: The new, direct token extraction task
    # TARGET_URL = "https://register.hackrx.in/utils/get-secret-token?hackTeam=8070"
    # QUESTION = f"Please go to {TARGET_URL} and get the Secret Token"

    # Query 3: A multi-step challenge asking for the secret data itself
    TARGET_URL = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    QUESTION = f"Please go to {TARGET_URL}, solve the challenge, and tell me the secret data."
    

    
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