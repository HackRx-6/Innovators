#!/usr/bin/env python3
"""
agent_solver.py - Autonomous Web Agent

A fully autonomous, general-purpose web agent capable of navigating websites,
extracting information, and performing interactions using LLM-driven decision making.

Required pip packages:
    pip install playwright beautifulsoup4 requests python-dotenv pydantic tqdm tenacity lxml pillow opencv-python pytesseract

Environment variables (create .env file):
    AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
    AZURE_OPENAI_API_KEY=YOUR_KEY
    AZURE_OPENAI_DEPLOYMENT=gpt-4o
    AZURE_OPENAI_API_VERSION=2024-05-01-preview

Note: For OCR functionality, pytesseract requires Tesseract OCR binary to be installed.
If not available, OCR will gracefully degrade to return empty results.
"""

import asyncio
import json
import logging
import os
import re
import hashlib
import base64
import urllib.parse
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time

# Third-party imports
import requests
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

# Optional OCR support
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_STEPS = 40
MAX_HTML_CHARS = 15000
DEFAULT_TIMEOUT_MS = 3000


class AzureChatLLM:
    """Azure OpenAI Chat Completions API wrapper with function calling support."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        
        if not all([self.endpoint, self.api_key]):
            raise ValueError("Missing required Azure OpenAI environment variables")
        
        self.url = f"{self.endpoint}openai/deployments/{self.deployment}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call(self, messages: List[Dict], tools: Optional[List[Dict]] = None, temperature: float = 0.1) -> Dict:
        """Call Azure OpenAI Chat Completions API with function calling support."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = requests.post(
                f"{self.url}?api-version={self.api_version}",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            raise


class BrowserManager:
    """Manages Playwright browser instance and provides web automation tools."""
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.tabs = {}
        self.current_tab_id = 0
        
    async def start(self):
        """Initialize Playwright browser."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--no-sandbox",
                    "--disable-setuid-sandbox"
                ]
            )
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            self.page = await self.context.new_page()
            self.tabs[0] = self.page
            logger.info("Browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def close(self):
        """Clean up browser resources."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
    
    def _add_agent_ids(self, html: str) -> str:
        """Add stable agent_id attributes to interactive elements."""
        soup = BeautifulSoup(html, 'html.parser')
        agent_id = 0
        
        interactive_tags = ['a', 'button', 'input', 'select', 'textarea', 'form', 'img', 'iframe']
        for tag in soup.find_all(interactive_tags):
            if not tag.get('agent_id'):
                tag['agent_id'] = f"agent_{agent_id}"
                agent_id += 1
        
        # Also add to elements with onclick, role, or other interactive attributes
        for tag in soup.find_all(attrs={"onclick": True}):
            if not tag.get('agent_id'):
                tag['agent_id'] = f"agent_{agent_id}"
                agent_id += 1
                
        for tag in soup.find_all(attrs={"role": True}):
            if tag.get('role') in ['button', 'link', 'tab', 'menuitem']:
                if not tag.get('agent_id'):
                    tag['agent_id'] = f"agent_{agent_id}"
                    agent_id += 1
        
        return str(soup)
    
    # Tool implementations
    async def navigate(self, url: str) -> Dict:
        """Navigate to a URL."""
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            current_url = self.page.url
            logger.info(f"Navigated to: {current_url}")
            return {"status": "success", "current_url": current_url}
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return {"error": f"Navigation failed: {str(e)}"}
    
    async def reload(self) -> Dict:
        """Reload the current page."""
        try:
            await self.page.reload(wait_until="domcontentloaded", timeout=30000)
            return {"status": "success", "current_url": self.page.url}
        except Exception as e:
            return {"error": f"Reload failed: {str(e)}"}
    
    async def wait_for_selector(self, selector: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Dict:
        """Wait for an element to appear."""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout_ms)
            return {"found": True}
        except PlaywrightTimeoutError:
            return {"found": False}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_page_snapshot(self, max_chars: int = MAX_HTML_CHARS) -> Dict:
        """Get a simplified snapshot of the current page."""
        try:
            # Get page title and URL
            title = await self.page.title()
            url = self.page.url
            
            # Get HTML and simplify
            html = await self.page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(["script", "style", "meta", "link", "noscript"]):
                element.decompose()
            
            # Add agent IDs to interactive elements
            html_with_ids = self._add_agent_ids(str(soup))
            
            # Truncate if too long
            simplified_html = html_with_ids[:max_chars] if len(html_with_ids) > max_chars else html_with_ids
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            text_summary = text_content[:max_chars // 2] if len(text_content) > max_chars // 2 else text_content
            
            # Extract interactive elements
            interactive_elements = []
            soup_with_ids = BeautifulSoup(html_with_ids, 'html.parser')
            
            for element in soup_with_ids.find_all(attrs={"agent_id": True}):
                elem_data = {
                    "agent_id": element.get("agent_id"),
                    "tag": element.name,
                    "type": element.get("type"),
                    "text": element.get_text(strip=True)[:100],  # Limit text length
                    "href": element.get("href"),
                    "src": element.get("src"),
                    "value": element.get("value"),
                    "placeholder": element.get("placeholder"),
                    "name": element.get("name"),
                    "role": element.get("role"),
                    "aria_label": element.get("aria-label")
                }
                # Remove None values
                elem_data = {k: v for k, v in elem_data.items() if v is not None}
                interactive_elements.append(elem_data)
            
            return {
                "simplified_html": simplified_html,
                "text_summary": text_summary,
                "interactive_elements": interactive_elements[:50],  # Limit number of elements
                "title": title,
                "url": url
            }
        except Exception as e:
            return {"error": f"Failed to get page snapshot: {str(e)}"}
    
    async def query_selectors(self, selectors: List[str]) -> Dict:
        """Query multiple selectors and return information about matching elements."""
        results = []
        for selector in selectors:
            try:
                elements = await self.page.query_selector_all(selector)
                if elements:
                    element_info = []
                    for elem in elements[:5]:  # Limit to first 5 matches
                        inner_text = await elem.inner_text() if await elem.is_visible() else ""
                        all_text = await elem.text_content() or ""
                        # Get key attributes
                        attrs = {}
                        for attr in ['id', 'class', 'name', 'type', 'value', 'href', 'src']:
                            val = await elem.get_attribute(attr)
                            if val:
                                attrs[attr] = val
                        
                        element_info.append({
                            "inner_text": inner_text[:200],
                            "all_text": all_text[:200],
                            "attributes": attrs
                        })
                    
                    results.append({
                        "selector": selector,
                        "count": len(elements),
                        "elements": element_info
                    })
                else:
                    results.append({
                        "selector": selector,
                        "count": 0,
                        "elements": []
                    })
            except Exception as e:
                results.append({
                    "selector": selector,
                    "error": str(e)
                })
        
        return {"results": results}
    
    async def click(self, selector_or_id: str, by: str = "css") -> Dict:
        """Click on an element."""
        try:
            if by == "agent_id":
                selector = f"[agent_id='{selector_or_id}']"
            elif by == "xpath":
                selector = selector_or_id
            elif by == "text":
                # Use text locator
                await self.page.get_by_text(selector_or_id).first.click(timeout=DEFAULT_TIMEOUT_MS)
                return {"status": "success"}
            else:
                selector = selector_or_id
            
            await self.page.click(selector, timeout=DEFAULT_TIMEOUT_MS)
            # Wait a bit for any resulting page changes
            await self.page.wait_for_timeout(500)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Click failed: {str(e)}"}
    
    async def hover(self, selector: str) -> Dict:
        """Hover over an element."""
        try:
            await self.page.hover(selector, timeout=DEFAULT_TIMEOUT_MS)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Hover failed: {str(e)}"}
    
    async def scroll(self, to: str, value: int = 0) -> Dict:
        """Scroll the page."""
        try:
            if to == "top":
                await self.page.evaluate("window.scrollTo(0, 0)")
            elif to == "bottom":
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif to == "pixels":
                await self.page.evaluate(f"window.scrollBy(0, {value})")
            
            scroll_y = await self.page.evaluate("window.pageYOffset")
            return {"status": "success", "scroll_y": scroll_y}
        except Exception as e:
            return {"error": f"Scroll failed: {str(e)}"}
    
    async def type_text(self, selector_or_id: str, text: str, by: str = "css", delay_ms: int = 0) -> Dict:
        """Type text into an element."""
        try:
            if by == "agent_id":
                selector = f"[agent_id='{selector_or_id}']"
            else:
                selector = selector_or_id
            
            # Clear the field first
            await self.page.fill(selector, "")
            # Type the text
            if delay_ms > 0:
                await self.page.type(selector, text, delay=delay_ms)
            else:
                await self.page.fill(selector, text)
            
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Type text failed: {str(e)}"}
    
    async def press(self, selector: str, key: str) -> Dict:
        """Press a key on an element."""
        try:
            await self.page.press(selector, key, timeout=DEFAULT_TIMEOUT_MS)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Press key failed: {str(e)}"}
    
    async def select_option(self, selector: str, value: str) -> Dict:
        """Select an option from a dropdown."""
        try:
            await self.page.select_option(selector, value, timeout=DEFAULT_TIMEOUT_MS)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Select option failed: {str(e)}"}
    
    async def drag_and_drop(self, source_selector: str, target_selector: str) -> Dict:
        """Drag and drop from source to target."""
        try:
            await self.page.drag_and_drop(source_selector, target_selector, timeout=DEFAULT_TIMEOUT_MS)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Drag and drop failed: {str(e)}"}
    
    async def upload_file(self, selector: str, file_path: str) -> Dict:
        """Upload a file to a file input."""
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            await self.page.set_input_files(selector, file_path, timeout=DEFAULT_TIMEOUT_MS)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"File upload failed: {str(e)}"}
    
    async def open_new_tab(self, url: str) -> Dict:
        """Open a new tab and navigate to URL."""
        try:
            new_page = await self.context.new_page()
            await new_page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            tab_id = len(self.tabs)
            self.tabs[tab_id] = new_page
            
            return {"tab_id": tab_id}
        except Exception as e:
            return {"error": f"Open new tab failed: {str(e)}"}
    
    async def switch_tab(self, tab_id: int) -> Dict:
        """Switch to a specific tab."""
        try:
            if tab_id not in self.tabs:
                return {"error": f"Tab {tab_id} not found"}
            
            self.page = self.tabs[tab_id]
            self.current_tab_id = tab_id
            return {"status": "success", "current_url": self.page.url}
        except Exception as e:
            return {"error": f"Switch tab failed: {str(e)}"}
    
    async def current_url(self) -> Dict:
        """Get current URL."""
        try:
            return {"url": self.page.url}
        except Exception as e:
            return {"error": str(e)}
    
    async def handle_dialog(self, action: str, prompt_text: Optional[str] = None) -> Dict:
        """Handle browser dialogs (alert, confirm, prompt)."""
        try:
            # Set up dialog handler
            def handle_dialog_event(dialog):
                if action == "accept":
                    if dialog.type == "prompt" and prompt_text:
                        dialog.accept(prompt_text)
                    else:
                        dialog.accept()
                else:
                    dialog.dismiss()
            
            self.page.on("dialog", handle_dialog_event)
            return {"status": "success"}
        except Exception as e:
            return {"error": f"Handle dialog failed: {str(e)}"}
    
    async def get_local_storage(self) -> Dict:
        """Get local storage items."""
        try:
            items = await self.page.evaluate("() => Object.assign({}, window.localStorage)")
            return {"items": items}
        except Exception as e:
            return {"error": f"Get local storage failed: {str(e)}"}
    
    async def get_session_storage(self) -> Dict:
        """Get session storage items."""
        try:
            items = await self.page.evaluate("() => Object.assign({}, window.sessionStorage)")
            return {"items": items}
        except Exception as e:
            return {"error": f"Get session storage failed: {str(e)}"}
    
    async def get_cookies(self) -> Dict:
        """Get page cookies."""
        try:
            cookies = await self.context.cookies()
            return {"cookies": cookies}
        except Exception as e:
            return {"error": f"Get cookies failed: {str(e)}"}
    
    async def evaluate_js(self, expression: str) -> Dict:
        """Evaluate JavaScript expression."""
        try:
            result = await self.page.evaluate(expression)
            # Ensure result is JSON serializable
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                return {"result": result}
            else:
                return {"result": str(result)}
        except Exception as e:
            return {"error": f"JS evaluation failed: {str(e)}"}
    
    async def screenshot(self, full_page: bool = True, selector: Optional[str] = None) -> Dict:
        """Take a screenshot."""
        try:
            if selector:
                element = await self.page.query_selector(selector)
                if not element:
                    return {"error": f"Element not found: {selector}"}
                screenshot_bytes = await element.screenshot()
            else:
                screenshot_bytes = await self.page.screenshot(full_page=full_page)
            
            png_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            return {"png_base64": png_base64}
        except Exception as e:
            return {"error": f"Screenshot failed: {str(e)}"}
    
    async def ocr_image(self, png_base64: str) -> Dict:
        """Extract text from image using OCR."""
        if not OCR_AVAILABLE:
            return {"text": "", "note": "ocr_unavailable"}
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(png_base64)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to PIL Image for pytesseract
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Extract text
            text = pytesseract.image_to_string(pil_img)
            
            return {"text": text.strip()}
        except Exception as e:
            return {"error": f"OCR failed: {str(e)}"}
    
    async def extract_links(self) -> Dict:
        """Extract all links from the page."""
        try:
            links = await self.page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        text: link.textContent.trim(),
                        href: link.href
                    })).filter(link => link.text && link.href);
                }
            """)
            return {"links": links[:100]}  # Limit to first 100 links
        except Exception as e:
            return {"error": f"Extract links failed: {str(e)}"}
    
    async def extract_tables(self) -> Dict:
        """Extract table data from the page."""
        try:
            tables = await self.page.evaluate("""
                () => {
                    const tables = Array.from(document.querySelectorAll('table'));
                    return tables.map(table => {
                        const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                        const rows = Array.from(table.querySelectorAll('tr')).slice(headers.length > 0 ? 1 : 0).map(tr => {
                            return Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim());
                        });
                        return { headers, rows };
                    });
                }
            """)
            return {"tables": tables[:10]}  # Limit to first 10 tables
        except Exception as e:
            return {"error": f"Extract tables failed: {str(e)}"}


class UtilityTools:
    """Utility functions for string manipulation and data processing."""
    
    @staticmethod
    def regex_extract(text: str, pattern: str, flags: List[str] = None) -> Dict:
        """Extract matches using regex."""
        try:
            flag_map = {
                'IGNORECASE': re.IGNORECASE,
                'MULTILINE': re.MULTILINE,
                'DOTALL': re.DOTALL
            }
            
            regex_flags = 0
            if flags:
                for flag in flags:
                    if flag in flag_map:
                        regex_flags |= flag_map[flag]
            
            matches = re.findall(pattern, text, regex_flags)
            return {"matches": matches}
        except Exception as e:
            return {"error": f"Regex extraction failed: {str(e)}"}
    
    @staticmethod
    def base64_encode(text: str) -> Dict:
        """Base64 encode text."""
        try:
            encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            return {"encoded": encoded}
        except Exception as e:
            return {"error": f"Base64 encoding failed: {str(e)}"}
    
    @staticmethod
    def base64_decode(text: str) -> Dict:
        """Base64 decode text."""
        try:
            decoded = base64.b64decode(text).decode('utf-8')
            return {"decoded": decoded}
        except Exception as e:
            return {"error": f"Base64 decoding failed: {str(e)}"}
    
    @staticmethod
    def url_encode(text: str) -> Dict:
        """URL encode text."""
        try:
            encoded = urllib.parse.quote(text)
            return {"encoded": encoded}
        except Exception as e:
            return {"error": f"URL encoding failed: {str(e)}"}
    
    @staticmethod
    def url_decode(text: str) -> Dict:
        """URL decode text."""
        try:
            decoded = urllib.parse.unquote(text)
            return {"decoded": decoded}
        except Exception as e:
            return {"error": f"URL decoding failed: {str(e)}"}
    
    @staticmethod
    def hash_text(text: str, algo: str) -> Dict:
        """Hash text using specified algorithm."""
        try:
            if algo == "md5":
                hash_obj = hashlib.md5()
            elif algo == "sha1":
                hash_obj = hashlib.sha1()
            elif algo == "sha256":
                hash_obj = hashlib.sha256()
            else:
                return {"error": f"Unsupported hash algorithm: {algo}"}
            
            hash_obj.update(text.encode('utf-8'))
            return {"hexdigest": hash_obj.hexdigest()}
        except Exception as e:
            return {"error": f"Hashing failed: {str(e)}"}


class WebAgent:
    """Main web agent orchestrator."""
    
    def __init__(self):
        self.llm = AzureChatLLM()
        self.browser = BrowserManager()
        self.utils = UtilityTools()
        self.answers = {}
        self.messages = []
        self.step_count = 0
        self.finished = False
        
        # Tool registry
        self.tools = self._build_tool_registry()
    
    def _build_tool_registry(self) -> Dict:
        """Build the tool registry with function schemas for Azure OpenAI."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "navigate",
                    "description": "Navigate to a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to navigate to"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reload",
                    "description": "Reload the current page",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "wait_for_selector",
                    "description": "Wait for an element to appear on the page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector to wait for"},
                            "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds", "default": 10000}
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_snapshot",
                    "description": "Get a simplified snapshot of the current page with interactive elements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_chars": {"type": "integer", "description": "Maximum characters in response", "default": 20000}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_selectors",
                    "description": "Query multiple CSS selectors and get element information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selectors": {"type": "array", "items": {"type": "string"}, "description": "List of CSS selectors to query"}
                        },
                        "required": ["selectors"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click on an element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector_or_id": {"type": "string", "description": "CSS selector, XPath, text, or agent_id"},
                            "by": {"type": "string", "enum": ["css", "xpath", "text", "agent_id"], "default": "css"}
                        },
                        "required": ["selector_or_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "hover",
                    "description": "Hover over an element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"}
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "scroll",
                    "description": "Scroll the page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "enum": ["top", "bottom", "pixels"]},
                            "value": {"type": "integer", "description": "Pixels to scroll (for 'pixels' mode)", "default": 0}
                        },
                        "required": ["to"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text into an element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector_or_id": {"type": "string", "description": "CSS selector or agent_id"},
                            "text": {"type": "string", "description": "Text to type"},
                            "by": {"type": "string", "enum": ["css", "agent_id"], "default": "css"},
                            "delay_ms": {"type": "integer", "description": "Delay between keystrokes", "default": 0}
                        },
                        "required": ["selector_or_id", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "press",
                    "description": "Press a key on an element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"},
                            "key": {"type": "string", "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape')"}
                        },
                        "required": ["selector", "key"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_option",
                    "description": "Select an option from a dropdown",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector of the select element"},
                            "value": {"type": "string", "description": "Value to select"}
                        },
                        "required": ["selector", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "drag_and_drop",
                    "description": "Drag and drop from source to target",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_selector": {"type": "string", "description": "CSS selector of source element"},
                            "target_selector": {"type": "string", "description": "CSS selector of target element"}
                        },
                        "required": ["source_selector", "target_selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "upload_file",
                    "description": "Upload a file to a file input",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector of file input"},
                            "file_path": {"type": "string", "description": "Path to file to upload"}
                        },
                        "required": ["selector", "file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "open_new_tab",
                    "description": "Open a new tab and navigate to URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to navigate to in new tab"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "switch_tab",
                    "description": "Switch to a specific tab",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tab_id": {"type": "integer", "description": "ID of tab to switch to"}
                        },
                        "required": ["tab_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "current_url",
                    "description": "Get the current URL",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "handle_dialog",
                    "description": "Handle browser dialogs (alert, confirm, prompt)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["accept", "dismiss"]},
                            "prompt_text": {"type": "string", "description": "Text to enter in prompt dialog"}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_local_storage",
                    "description": "Get local storage items",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_session_storage",
                    "description": "Get session storage items",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cookies",
                    "description": "Get page cookies",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_js",
                    "description": "Evaluate JavaScript expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "JavaScript expression to evaluate"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "screenshot",
                    "description": "Take a screenshot of the page or element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "full_page": {"type": "boolean", "description": "Whether to capture full page", "default": True},
                            "selector": {"type": "string", "description": "CSS selector of element to capture"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ocr_image",
                    "description": "Extract text from image using OCR",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "png_base64": {"type": "string", "description": "Base64 encoded PNG image"}
                        },
                        "required": ["png_base64"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_links",
                    "description": "Extract all links from the current page",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_tables",
                    "description": "Extract table data from the current page",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "regex_extract",
                    "description": "Extract matches using regular expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to search in"},
                            "pattern": {"type": "string", "description": "Regular expression pattern"},
                            "flags": {"type": "array", "items": {"type": "string"}, "description": "Regex flags"}
                        },
                        "required": ["text", "pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "base64_encode",
                    "description": "Encode text as base64",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to encode"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "base64_decode",
                    "description": "Decode base64 text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Base64 text to decode"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "url_encode",
                    "description": "URL encode text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to encode"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "url_decode",
                    "description": "URL decode text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to decode"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "hash_text",
                    "description": "Hash text using specified algorithm",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to hash"},
                            "algo": {"type": "string", "enum": ["md5", "sha1", "sha256"], "description": "Hash algorithm"}
                        },
                        "required": ["text", "algo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "record_answer",
                    "description": "Record an answer to a specific question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "description": "0-based index of the question"},
                            "answer": {"type": "string", "description": "The answer to record"}
                        },
                        "required": ["index", "answer"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Indicate that all answers have been recorded and the task is complete",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
        
        return tools
    
    async def _call_tool(self, tool_name: str, args: Dict) -> Dict:
        """Execute a tool call."""
        try:
            # Browser tools
            if hasattr(self.browser, tool_name):
                method = getattr(self.browser, tool_name)
                if asyncio.iscoroutinefunction(method):
                    return await method(**args)
                else:
                    return method(**args)
            
            # Utility tools
            elif hasattr(self.utils, tool_name):
                method = getattr(self.utils, tool_name)
                return method(**args)
            
            # Agent lifecycle tools
            elif tool_name == "record_answer":
                index = args["index"]
                answer = args["answer"]
                self.answers[index] = answer
                logger.info(f"Recorded answer {index}: {answer[:100]}...")
                return {"ok": True}
            
            elif tool_name == "finish":
                self.finished = True
                logger.info("Agent indicated completion")
                return {"ok": True}
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    async def run(self, url: str, questions: List[str]) -> Dict:
        """Run the web agent to answer questions about a website."""
        try:
            await self.browser.start()
            
            # Initialize conversation
            system_prompt = """You are an autonomous web agent capable of navigating websites, extracting information, and performing interactions. Your goal is to answer the provided questions by exploring and interacting with the website.

Available Tools: You have access to comprehensive browser automation tools, data extraction tools, and utility functions. Use them strategically to gather information and complete tasks.

Instructions:
1. Start by navigating to the given URL and getting a page snapshot
2. Analyze the page structure and plan your approach
3. Use appropriate tools to find information, interact with elements, and explore the site
4. For each question, gather relevant information and record your answer using record_answer(index, answer)
5. Use stable selectors and verify your actions when possible
6. If text is not selectable, use screenshots and OCR for extraction
7. When all questions are answered, call finish() to complete the task

Success Criteria:
- Answer all questions accurately based on the website content
- Use robust automation practices (wait for elements, verify actions)
- Handle errors gracefully and try alternative approaches
- Record answers in the correct order using 0-based indexing

Remember: You must call record_answer() for each question and finish() when done."""

            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Navigate to {url} and answer these questions:\n" + 
                 "\n".join([f"{i}: {q}" for i, q in enumerate(questions)])}
            ]
            
            logger.info(f"Starting agent with {len(questions)} questions")
            
            # Main agent loop
            while self.step_count < MAX_STEPS and not self.finished:
                self.step_count += 1
                logger.info(f"Step {self.step_count}/{MAX_STEPS}")
                
                try:
                    # Call LLM
                    response = self.llm.call(self.messages, self.tools)
                    
                    if "choices" not in response or not response["choices"]:
                        logger.error("No choices in LLM response")
                        break
                    
                    choice = response["choices"][0]
                    message = choice["message"]
                    
                    # Add assistant message
                    self.messages.append(message)
                    
                    # Handle tool calls
                    if message.get("tool_calls"):
                        for tool_call in message["tool_calls"]:
                            function = tool_call["function"]
                            tool_name = function["name"]
                            args = json.loads(function["arguments"])
                            
                            logger.info(f"Calling tool: {tool_name} with args: {args}")
                            
                            # Execute tool
                            result = await self._call_tool(tool_name, args)
                            
                            # Add tool result to conversation
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(result)
                            })
                            
                            logger.info(f"Tool {tool_name} result: {str(result)[:200]}...")
                    
                    # If no tool calls, the assistant might be done or stuck
                    elif message.get("content"):
                        logger.info(f"Assistant message: {message['content']}")
                        # Continue the conversation
                        if not self.finished:
                            self.messages.append({
                                "role": "user",
                                "content": "Continue with your task. If you have answered all questions, call finish()."
                            })
                    
                except Exception as e:
                    logger.error(f"Error in agent step {self.step_count}: {e}")
                    # Add error to conversation to let the LLM adapt
                    self.messages.append({
                        "role": "user",
                        "content": f"An error occurred: {str(e)}. Please try a different approach."
                    })
            
            # Prepare final results
            results = {}
            for i, question in enumerate(questions):
                results[i] = {
                    "question": question,
                    "answer": self.answers.get(i, "(not found)")
                }
            
            logger.info(f"Agent completed after {self.step_count} steps")
            logger.info(f"Answered {len(self.answers)}/{len(questions)} questions")
            
            return results
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {"error": str(e)}
        finally:
            await self.browser.close()


def print_results(results: Dict, questions: List[str]):
    """Print results in the required format."""
    print("\n" + "="*80)
    print("🎉 FINAL RESULTS")
    print("="*80)
    
    for i, question in enumerate(questions):
        answer = results.get(i, {}).get("answer", "(not found)")
        print(f"QUESTION_{i+1}:")
        print(f"  Question: {question}")
        print(f"  Answer:   {answer}")
        print()
    
    print("="*80)


async def main(url, questions):
    """Main entry point."""
    # Example usage - replace with your actual URL and questions
    # url = "https://register.hackrx.in/showdown/startChallenge/ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmpiMjlzUjNWNUlqb2lUVlZCV2xwQlRTSXNJbU5vWVd4c1pXNW5aVWxFSWpvaWFHbGtaR1Z1SWl3aWRYTmxja2xrSWpvaWRYTmxjbDl0ZFdGNmVtRnRJaXdpWlcxaGFXd2lPaUp0ZFdGNmVtRnRRR0poYW1GcVptbHVjMlZ5ZG1obFlXeDBhQzVwYmlJc0luSnZiR1VpT2lKamIyOXNYMmQxZVNJc0ltbGhkQ0k2TVRjMU5UZzFPRE01TlN3aVpYaHdJam94TnpVMU9UUTBOemsxZlEuUXRkdmVGWmhnVDVLNEtYcFdpbWRNbTQ5MW1SZThoTjY2cC1jSjFCU2lzTQ=="
    # questions =  [
    #         "Go to the website and start the challenge. Complete the challenge and return the answers for the following question?  What is the challenge ID? ",
    #         "Go to the website and start the challenge. Complete the challenge and return the answers for the following question? What is the completion code? "]


    
    print(f"🚀 Starting Web Agent")
    print(f"URL: {url}")
    print(f"Questions: {len(questions)}")
    
    agent = WebAgent()
    results = await agent.run(url, questions)
    
    if "error" in results:
        print(f"❌ Agent failed: {results['error']}")
    else:
        return results


if __name__ == "__main__":
    asyncio.run(main())