"""
graph.py — Enhanced pipeline using LangGraph for state management and workflow orchestration
Keep this file in your project root and ensure llm_client.py is available.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import os
import json
import traceback
import time
import re

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Use the provided LLM config module
from llm_client import LLM

# Message classes for constructing calls if needed
from langchain.schema import SystemMessage, HumanMessage, BaseMessage

# Try to import user's PlaywrightController; otherwise we'll provide a fallback implementation below.
try:
    from tools.playwright_tool import PlaywrightController  # type: ignore
except Exception:
    PlaywrightController = None  # we'll define a fallback later

# Parser helpers from your project
from utils.parser import expand_url_with_tokens, safe_parse_plan  # keep using project's parser.safe_parse_plan


# Define the state schema for our graph
class GraphState(TypedDict, total=False):
    """State schema for the web automation graph"""
    input_payload: Dict[str, Any]
    url: str
    questions: List[str]
    plan: List[Dict[str, Any]]
    browser_result: Dict[str, Any]
    answers: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    error: Optional[str]
    debug_info: Dict[str, Any]


# ---- Plan validation helpers ----
ALLOWED_STEP_TYPES = {"navigate", "click", "extract", "run_js", "wait", "extract_full"}


def validate_plan(plan: List[Dict[str, Any]], default_url: str = "") -> List[Dict[str, Any]]:
    """
    Ensure that plan is a list of dicts with allowed types and required fields.
    Invalid steps are skipped.
    """
    safe_plan: List[Dict[str, Any]] = []
    for step in (plan or []):
        if not isinstance(step, dict):
            continue
        t = step.get("type")
        if t not in ALLOWED_STEP_TYPES:
            # skip unknown types
            continue

        # minimal validations per type
        if t == "navigate":
            url = step.get("url") or default_url
            if not url or not isinstance(url, str):
                # skip navigation without url
                continue
            safe_plan.append({"type": "navigate", "url": url})
        elif t == "click":
            sel = step.get("selector")
            if not sel or not isinstance(sel, str):
                continue
            safe_plan.append({"type": "click", "selector": sel})
        elif t == "extract":
            sel = step.get("selector")
            if not sel or not isinstance(sel, str):
                continue
            safe_plan.append({"type": "extract", "selector": sel})
        elif t == "wait":
            try:
                secs = float(step.get("secs", 1.0))
            except Exception:
                secs = 1.0
            # clamp waits (prevent long sleeps)
            if secs < 0:
                secs = 0.1
            if secs > 10.0:
                secs = 10.0
            safe_plan.append({"type": "wait", "secs": secs})
        elif t == "run_js":
            # allow only short scripts; reject very large scripts from untrusted LLM output
            script = step.get("script", "")
            if not isinstance(script, str) or len(script) > 20000:
                # skip unsafe run_js
                continue
            safe_plan.append({"type": "run_js", "script": script})
        elif t == "extract_full":
            safe_plan.append({"type": "extract_full"})
    return safe_plan


# ---- LLM helpers ----
def _extract_text_from_llm_result(res: Any) -> str:
    """
    Attempt to extract text from various response shapes returned by LLM.invoke(...)
    """
    try:
        if res is None:
            return ""

        # If res is a list-like of messages
        if isinstance(res, (list, tuple)) and len(res) > 0:
            first = res[0]
            if hasattr(first, "content"):
                return getattr(first, "content") or ""
            if isinstance(first, dict) and "content" in first:
                return first["content"]

        # Single message object
        if hasattr(res, "content"):
            return getattr(res, "content") or ""
        if hasattr(res, "text"):
            return getattr(res, "text") or ""

        # If the object has a choices list (like openai)
        if hasattr(res, "choices"):
            try:
                ch = res.choices
                if isinstance(ch, (list, tuple)) and len(ch) > 0:
                    c0 = ch[0]
                    if hasattr(c0, "message") and hasattr(c0.message, "content"):
                        return c0.message.content or ""
                    if isinstance(c0, dict) and "text" in c0:
                        return c0["text"] or ""
            except Exception:
                pass

        # fall back to string representation
        return str(res)
    except Exception:
        return ""


def _llm_get_text(prompt: str) -> str:
    """
    Use the provided LLM with exactly the required invocation shapes
    """
    try:
        if hasattr(LLM, "invoke"):
            try:
                res = LLM.invoke(prompt)
                txt = _extract_text_from_llm_result(res)
                if txt and txt.strip():
                    return txt
            except Exception:
                pass

            # Try the list-of-SystemMessage shape
            try:
                res2 = LLM.invoke([SystemMessage(content=prompt)])
                txt2 = _extract_text_from_llm_result(res2)
                if txt2 and txt2.strip():
                    return txt2
            except Exception:
                pass

    except Exception:
        pass

    return ""


# ---- Graph nodes ----
def parse_input_node(state: GraphState) -> GraphState:
    """Parse and expand input payload"""
    try:
        inp = state.get("input_payload", {}) or {}
        url = inp.get("url", "")
        url = expand_url_with_tokens(url, inp)
        questions = inp.get("questions", []) or []

        new_state: GraphState = {
            **state,
            "url": url,
            "questions": questions,
            "debug_info": {
                **state.get("debug_info", {}),
                "parsed_url": url,
                "questions_count": len(questions)
            }
        }
        return new_state
    except Exception as e:
        return {
            **state,
            "error": f"Input parsing failed: {str(e)}",
            "debug_info": {**state.get("debug_info", {}), "parse_error": str(e)}
        }


def plan_actions_node(state: GraphState) -> GraphState:
    """Generate action plan using LLM"""
    try:
        questions = state.get("questions", []) or []
        url = state.get("url", "") or ""

        # Create planning prompt with focus on hidden elements and correct format
        prompt = (
            "You are an automation planner for web scraping challenges that often contain HIDDEN elements. "
            "Given a target URL and questions, create a JSON action plan to find hidden content. "
            "CRITICAL: Each action must use 'type' field, not 'action' field!\n"
            "Format: [{\"type\":\"navigate\",\"url\":\"...\"}, {\"type\":\"click\",\"selector\":\"...\"}, etc.]\n"
            "Available types: navigate, click, extract, run_js, wait, extract_full\n"
            "The challenge likely contains elements hidden via CSS. Focus on revealing hidden elements.\n\n"
            f"URL: {url}\nQuestions: {questions}\n"
            "Example: [{\"type\":\"navigate\",\"url\":\"https://...\"}]\n"
            "Return only valid JSON array with 'type' field for each action!"
        )

        planning_message = HumanMessage(content=f"Planning actions for URL: {url}")

        # Ask the LLM for a plan
        llm_response_text = _llm_get_text(prompt)

        plan: List[Dict[str, Any]] = []
        if llm_response_text:
            try:
                # Use project's safe_parse_plan if available
                plan = safe_parse_plan(llm_response_text)
            except Exception:
                plan = []

        # If parsing failed or returned nothing, provide a robust fallback plan
        if not isinstance(plan, list) or len(plan) == 0:
            plan = [
                {"type": "navigate", "url": url},
                {"type": "wait", "secs": 2.0},
                {"type": "run_js", "script": """
                    // Comprehensive hidden element reveal
                    document.querySelectorAll('*').forEach(el => {
                        try {
                            el.style.display = '';
                            el.style.visibility = 'visible';
                            el.style.opacity = '1';
                            el.style.position = 'static';
                            el.style.left = 'auto';
                            el.style.top = 'auto';
                            el.hidden = false;
                            el.removeAttribute('aria-hidden');
                            if (el.style.fontSize === '0px') el.style.fontSize = '';
                        } catch(e) {}
                    });
                    return 'Elements revealed';
                """},
                {"type": "wait", "secs": 1.0},
                {"type": "extract_full", "extract_full": True}
            ]

        # Validate & sanitize plan before storing
        plan = validate_plan(plan, default_url=url)

        return {
            **state,
            "plan": plan,
            "messages": state.get("messages", []) + [planning_message],
            "debug_info": {
                **state.get("debug_info", {}),
                "plan_steps": len(plan),
                "llm_response_length": len(llm_response_text) if llm_response_text else 0
            }
        }
    except Exception as e:
        return {
            **state,
            "error": f"Action planning failed: {str(e)}",
            "plan": [],
            "debug_info": {**state.get("debug_info", {}), "planning_error": str(e)}
        }


def execute_actions_node(state: GraphState) -> GraphState:
    """Execute browser actions"""
    try:
        plan = state.get("plan", []) or []
        url = state.get("url", "") or ""

        browser = PlaywrightController(headless=True)
        try:
            browser_result = browser.run_plan(plan, url)
            execution_message = HumanMessage(content=f"Executed {len(plan)} browser actions")

            return {
                **state,
                "browser_result": browser_result or {},
                "messages": state.get("messages", []) + [execution_message],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "execution_log_entries": len(browser_result.get("log", [])) if browser_result else 0,
                    "extracted_text_length": len(browser_result.get("combined_text", "")) if browser_result else 0
                }
            }
        finally:
            try:
                browser.close()
            except Exception:
                pass

    except Exception as e:
        return {
            **state,
            "error": f"Action execution failed: {str(e)}",
            "browser_result": {},
            "debug_info": {**state.get("debug_info", {}), "execution_error": str(e)}
        }


def extract_answers_node(state: GraphState) -> GraphState:
    """Extract answers from browser results with enhanced pattern matching"""
    try:
        questions = state.get("questions", []) or []
        browser_result = state.get("browser_result", {}) or {}
        text = browser_result.get("combined_text", "") or ""
        answers: List[str] = []

        # Debug: log the extracted text for analysis
        debug_patterns = []

        for i, q in enumerate(questions):
            found_answer = ""

            # Enhanced pattern matching strategies
            patterns_to_try = [
                # Direct challenge name patterns
                (r"challenge name[:\s-]*([^\n\r<>{}]{2,100})", "challenge_name_colon"),
                (r"Challenge Name[:\s-]*([^\n\r<>{}]{2,100})", "challenge_name_caps"),
                (r"challenge[:\s-]*([^\n\r<>{}]{2,100})", "challenge_generic"),

                # Hidden element patterns
                (r'HIDDEN\s+\w+[^}]*text[:\s]*([^\n\r]{2,100})', "hidden_text"),
                (r'class[="][^"]*challenge[^"]*"[^>]*>([^<]{2,100})', "challenge_class"),
                (r'id[="][^"]*challenge[^"]*"[^>]*>([^<]{2,100})', "challenge_id"),

                # Common challenge indicators
                (r'title[:\s]*([A-Z][^\n\r<>{}]{5,80})', "title_pattern"),
                (r'"name"[:\s]*"([^"]{2,80})"', "json_name"),
                (r"'name'[:\s]*'([^']{2,80})'", "json_name_single"),

                # Element-specific patterns
                (r'<h[1-6][^>]*>([^<]{2,100})</h[1-6]>', "heading_tags"),
                (r'<title[^>]*>([^<]{2,100})</title>', "title_tag"),
                (r'<div[^>]*challenge[^>]*>([^<]{2,100})</div>', "challenge_div"),

                # JavaScript/JSON patterns
                (r'challengeName[:\s]*["\']([^"\']{2,80})["\']', "js_challenge_name"),
                (r'challenge[:\s]*["\']([^"\']{2,80})["\']', "js_challenge"),

                # Fallback patterns
                (r'["\']([A-Z][a-zA-Z\s]{5,50})["\']', "quoted_caps"),
                (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "title_case_words")
            ]

            for pattern, pattern_name in patterns_to_try:
                matches = re.findall(pattern, text, flags=re.I | re.MULTILINE)
                if matches:
                    # Clean and validate matches
                    for match in matches:
                        cleaned = re.sub(r'^\s*[:=-]\s*', '', str(match)).strip()
                        if len(cleaned) > 2 and not cleaned.lower() in ['the', 'and', 'for', 'with', 'this', 'that']:
                            found_answer = cleaned
                            debug_patterns.append(f"Q{i+1}: Found '{cleaned}' using {pattern_name}")
                            break
                    if found_answer:
                        break

            # If no specific pattern matched, try extracting from common locations
            if not found_answer:
                # Look for any meaningful text in hidden elements
                hidden_matches = re.findall(r'HIDDEN[^}]*text[:\s]*([^\n\r,]{5,80})', text)
                if hidden_matches:
                    found_answer = hidden_matches[0].strip()
                    debug_patterns.append(f"Q{i+1}: Found '{found_answer}' in hidden elements")

            # Final fallback - look for any reasonable text
            if not found_answer and text.strip():
                # Extract first meaningful sentence or phrase
                sentences = re.split(r'[.!?\n\r]', text.strip())
                for sentence in sentences[:10]:  # Check first 10 sentences
                    clean_sentence = re.sub(r'[<>{}]', '', sentence).strip()
                    if 5 <= len(clean_sentence) <= 200 and any(c.isalpha() for c in clean_sentence):
                        found_answer = clean_sentence
                        debug_patterns.append(f"Q{i+1}: Using fallback sentence: '{found_answer}'")
                        break

            answers.append(found_answer if found_answer else "")

        extraction_message = HumanMessage(content=f"Extracted {len(answers)} answers from browser results")

        return {
            **state,
            "answers": answers,
            "messages": state.get("messages", []) + [extraction_message],
            "debug_info": {
                **state.get("debug_info", {}),
                "answers_extracted": len(answers),
                "source_text_length": len(text),
                "extraction_patterns": debug_patterns,
                "text_sample": text[:500] if text else "No text extracted"
            }
        }
    except Exception as e:
        return {
            **state,
            "error": f"Answer extraction failed: {str(e)}",
            "answers": [],
            "debug_info": {**state.get("debug_info", {}), "extraction_error": str(e)}
        }


def format_output_node(state: GraphState) -> GraphState:
    """Format final output"""
    try:
        input_payload = state.get("input_payload", {}) or {}
        plan = state.get("plan", []) or []
        browser_result = state.get("browser_result", {}) or {}
        answers = state.get("answers", []) or []
        debug_info = state.get("debug_info", {}) or {}

        # Enhanced debug information
        final_debug = {
            "url": input_payload.get("url"),
            "plan": plan,
            "browser_summary": browser_result.get("summary", ""),
            "browser_log": browser_result.get("log", [])[:20],
            "workflow_debug": debug_info,
            "message_count": len(state.get("messages", []) or [])
        }

        if state.get("error"):
            final_debug["error"] = state["error"]

        completion_message = HumanMessage(content=f"Workflow completed with {len(answers)} answers")

        return {
            **state,
            "messages": state.get("messages", []) + [completion_message],
            "debug_info": final_debug
        }
    except Exception as e:
        return {
            **state,
            "error": f"Output formatting failed: {str(e)}",
            "debug_info": {**state.get("debug_info", {}), "formatting_error": str(e)}
        }


def should_continue(state: GraphState) -> str:
    """Determine if workflow should continue based on state"""
    if state.get("error"):
        return "format_output"  # Skip to output formatting if there's an error
    return "continue"


def create_automation_graph() -> StateGraph:
    """Create and configure the LangGraph workflow"""

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("plan_actions", plan_actions_node)
    workflow.add_node("execute_actions", execute_actions_node)
    workflow.add_node("extract_answers", extract_answers_node)
    workflow.add_node("format_output", format_output_node)

    # Define the workflow edges
    workflow.set_entry_point("parse_input")

    # Sequential flow with error handling
    workflow.add_edge("parse_input", "plan_actions")
    workflow.add_edge("plan_actions", "execute_actions")
    workflow.add_edge("execute_actions", "extract_answers")
    workflow.add_edge("extract_answers", "format_output")
    workflow.add_edge("format_output", END)

    # Add memory for state persistence
    memory = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app


def build_and_run_graph(payload: Dict[str, Any], thread_id: str = "default") -> Dict[str, Any]:
    """
    Top-level runner that uses LangGraph to orchestrate the pipeline.
    Returns the final JSON-ready dict.
    """
    try:
        # Create the graph
        app = create_automation_graph()

        # Initialize state as a plain dict matching GraphState
        initial_state: GraphState = {
            "input_payload": payload,
            "url": "",
            "questions": [],
            "plan": [],
            "browser_result": {},
            "answers": [],
            "messages": [],
            "error": None,
            "debug_info": {}
        }

        # Configure thread for state persistence
        config = {"configurable": {"thread_id": thread_id}}

        # Run the graph
        final_state = app.invoke(initial_state, config=config)

        # Extract the final results
        result = {
            "answers": final_state.get("answers", []),
            "debug": final_state.get("debug_info", {})
        }

        if final_state.get("error"):
            result["error"] = final_state["error"]

        return result

    except Exception as e:
        # Return error information in a structured way for debugging
        return {
            "answers": [],
            "error": str(e),
            "traceback": traceback.format_exc(),
            "debug": {
                "graph_execution_failed": True,
                "original_payload": payload
            }
        }


def get_graph_visualization() -> str:
    """Get a visual representation of the graph structure"""
    try:
        app = create_automation_graph()
        # This would require additional dependencies like graphviz
        # For now, return a text representation
        return """
        Web Automation Graph Structure:

        parse_input → plan_actions → execute_actions → extract_answers → format_output → END

        Each node maintains state and can handle errors gracefully.
        Memory is preserved across the execution for debugging and state inspection.
        """
    except Exception as e:
        return f"Graph visualization failed: {str(e)}"


# ---- Legacy compatibility functions ----
def parse_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy compatibility wrapper"""
    state: GraphState = {
        "input_payload": payload,
        "url": "",
        "questions": [],
        "plan": [],
        "browser_result": {},
        "answers": [],
        "messages": [],
        "error": None,
        "debug_info": {}
    }
    result = parse_input_node(state)
    return {"url": result.get("url"), **(result.get("input_payload") or {})}


def plan_actions(input_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Legacy compatibility wrapper"""
    state: GraphState = {
        "input_payload": input_payload,
        "url": input_payload.get("url", ""),
        "questions": input_payload.get("questions", []),
        "plan": [],
        "browser_result": {},
        "answers": [],
        "messages": [],
        "error": None,
        "debug_info": {}
    }
    result = plan_actions_node(state)
    return result.get("plan", [])


def execute_actions(plan: List[Dict[str, Any]], default_url: Optional[str] = None) -> Dict[str, Any]:
    """Legacy compatibility wrapper"""
    state: GraphState = {
        "input_payload": {},
        "url": default_url or "",
        "questions": [],
        "plan": plan,
        "browser_result": {},
        "answers": [],
        "messages": [],
        "error": None,
        "debug_info": {}
    }
    result = execute_actions_node(state)
    return result.get("browser_result", {})


def extract_answers(input_payload: Dict[str, Any], browser_result: Dict[str, Any]) -> List[str]:
    """Legacy compatibility wrapper"""
    state: GraphState = {
        "input_payload": input_payload,
        "url": "",
        "questions": input_payload.get("questions", []),
        "plan": [],
        "browser_result": browser_result,
        "answers": [],
        "messages": [],
        "error": None,
        "debug_info": {}
    }
    result = extract_answers_node(state)
    return result.get("answers", [])


def format_output(input_payload: Dict[str, Any], plan: List[Dict[str, Any]],
                  browser_result: Dict[str, Any], answers: List[str]) -> Dict[str, Any]:
    """Legacy compatibility wrapper"""
    return {
        "answers": answers,
        "debug": {
            "url": input_payload.get("url"),
            "plan": plan,
            "browser_summary": browser_result.get("summary", ""),
            "browser_log": browser_result.get("log", [])[:20]
        }
    }


# ---- Local fallback PlaywrightController (only used if user's tool not available) ----
if PlaywrightController is None:
    # Provide fallback implementation that uses Playwright sync API
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        sync_playwright = None  # If Playwright isn't available, calling code will error at runtime.

    class _LocalPlaywrightController:
        def __init__(self, headless: bool = True):
            if sync_playwright is None:
                raise RuntimeError("Playwright is not installed or not available in this environment.")
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=headless)
            self._context = self._browser.new_context()
            self._page = self._context.new_page()

        def close(self):
            try:
                self._page.close()
            except:
                pass
            try:
                self._context.close()
            except:
                pass
            try:
                self._browser.close()
            except:
                pass
            try:
                self._pw.stop()
            except:
                pass

        def run_plan(self, plan: List[Dict[str, Any]], default_url: Optional[str] = None) -> Dict[str, Any]:
            log = []
            extracted_texts = []
            for step in (plan or []):
                t = step.get("type")
                # navigate
                if t == "navigate":
                    url = step.get("url") or default_url
                    if not url:
                        log.append({"error": "no url provided for navigate step", "step": step})
                        continue
                    try:
                        resp = self._page.goto(url, wait_until="networkidle", timeout=30000)
                        try:
                            self._page.wait_for_load_state("networkidle", timeout=5000)
                        except:
                            pass
                        log.append({"navigate": url, "status": getattr(resp, "status", None)})
                    except Exception as e:
                        log.append({"navigate_error": str(e), "url": url})
                        continue

                elif t == "click":
                    sel = step.get("selector")
                    if not sel:
                        log.append({"click_skip": "missing selector", "step": step})
                        continue
                    try:
                        try:
                            self._page.wait_for_selector(sel, timeout=3000)
                        except:
                            pass
                        self._page.click(sel, timeout=5000)
                        log.append({"click": sel})
                        try:
                            self._page.wait_for_load_state("networkidle", timeout=1500)
                        except:
                            time.sleep(0.4)
                    except Exception as e:
                        log.append({"click_error": str(e), "selector": sel})

                elif t == "wait":
                    secs = float(step.get("secs", 1.0))
                    if secs > 0:
                        time.sleep(secs)
                    log.append({"wait": secs})

                elif t == "run_js":
                    script = step.get("script", "")[:20000]  # ensure bounded length
                    try:
                        res = self._page.evaluate(script)
                        log.append({"run_js": script[:120], "result_preview": str(res)[:500]})
                    except Exception as e:
                        log.append({"run_js_error": str(e)})

                elif t == "extract":
                    sel = step.get("selector")
                    if not sel:
                        log.append({"extract_skip": "missing selector", "step": step})
                        continue
                    try:
                        elements = self._page.locator(sel)
                        count = 0
                        try:
                            count = elements.count()
                        except:
                            count = 0
                        if count > 0:
                            texts = []
                            for i in range(min(count, 10)):
                                try:
                                    el_text = elements.nth(i).inner_text(timeout=3000)
                                    if el_text and el_text.strip():
                                        texts.append(el_text.strip())
                                except:
                                    try:
                                        el_text = elements.nth(i).text_content(timeout=3000)
                                        if el_text and el_text.strip():
                                            texts.append(el_text.strip())
                                    except:
                                        pass
                            combined = "\n".join(texts)
                            extracted_texts.append(combined)
                            log.append({"extract": sel, "elements": count, "length": len(combined)})
                        else:
                            log.append({"extract_error": "no elements found", "selector": sel})
                    except Exception as e:
                        log.append({"extract_error": str(e), "selector": sel})

                elif t == "extract_full":
                    texts = []
                    # try the enhanced hidden-text detection
                    try:
                        found = self._find_hidden_texts()
                        if found:
                            texts.extend(found)
                    except Exception as e:
                        log.append({"hidden_detect_error": str(e)})

                    # Visible content
                    try:
                        body_text = self._page.locator("body").inner_text(timeout=5000)
                        if body_text and body_text.strip():
                            texts.append(f"---VISIBLE-CONTENT---\n{body_text}")
                    except:
                        pass

                    try:
                        page_content = self._page.content()
                        texts.append(f"---RAW-HTML---\n{page_content[:15000]}")
                    except:
                        pass

                    combined = "\n\n".join(texts)
                    extracted_texts.append(combined)
                    log.append({"extract_full": True, "pieces": len(texts), "total_length": len(combined)})
                else:
                    log.append({"unknown_step": step})

            combined_text = "\n\n".join(extracted_texts).strip()
            summary = self._create_enhanced_summary(combined_text)
            return {"summary": summary, "combined_text": combined_text, "log": log}

        def _create_enhanced_summary(self, text: str) -> str:
            """Create an enhanced summary that highlights potential challenge names"""
            if not text:
                return "No content extracted"

            summary_parts = []

            # Look for challenge-related content
            challenge_matches = re.findall(r'challenge[^.]*?([A-Z][^.\n\r]{5,80})', text, flags=re.I)
            if challenge_matches:
                summary_parts.append(f"Potential challenges found: {', '.join(challenge_matches[:3])}")

            # Look for hidden content indicators
            hidden_matches = re.findall(r'HIDDEN[^}]*text[:\s]*([^\n\r,]{5,80})', text)
            if hidden_matches:
                summary_parts.append(f"Hidden content: {', '.join(hidden_matches[:3])}")

            # General summary
            summary_parts.append(f"Content length: {len(text)} chars")
            summary_parts.append(f"Content preview: {text[:400]}...")

            return " | ".join(summary_parts)

        def _find_hidden_texts(self):
            """Enhanced hidden element detection"""
            js = r'''
            (function(){
                const results = {
                    hidden_elements: [],
                    comments: [],
                    scripts: [],
                    meta_data: [],
                    data_attributes: []
                };

                // 1. Find all hidden elements with comprehensive checks
                const all_elements = Array.from(document.querySelectorAll('*'));

                for (const el of all_elements.slice(0, 1000)) { // Increased limit
                    try {
                        const computedStyle = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();

                        const isHidden = 
                            computedStyle.display === 'none' ||
                            computedStyle.visibility === 'hidden' ||
                            parseFloat(computedStyle.opacity) === 0 ||
                            (computedStyle.position === 'absolute' && (
                                parseFloat(computedStyle.left) < -9999 ||
                                parseFloat(computedStyle.top) < -9999
                            )) ||
                            el.hidden === true ||
                            el.getAttribute('aria-hidden') === 'true' ||
                            rect.width === 0 || rect.height === 0 ||
                            el.style.fontSize === '0px';

                        if (isHidden) {
                            const text = (el.innerText || el.textContent || '').trim();
                            const html = el.outerHTML;

                            if (text.length > 0) {
                                results.hidden_elements.push({
                                    tag: el.tagName.toLowerCase(),
                                    id: el.id || null,
                                    className: el.className || null,
                                    text: text.slice(0, 500),
                                    hideReason: computedStyle.display === 'none' ? 'display:none' : 
                                              computedStyle.visibility === 'hidden' ? 'visibility:hidden' :
                                              parseFloat(computedStyle.opacity) === 0 ? 'opacity:0' : 'other'
                                });
                            }

                            if (html.toLowerCase().includes('challenge') || 
                                html.toLowerCase().includes('name') ||
                                html.toLowerCase().includes('title')) {
                                results.hidden_elements.push({
                                    tag: el.tagName.toLowerCase(),
                                    id: el.id || null,
                                    className: el.className || null,
                                    html_snippet: html.slice(0, 800),
                                    text: text.slice(0, 200),
                                    type: 'challenge_related'
                                });
                            }
                        }

                        // Check for data attributes that might contain challenge info
                        for (const attr of el.attributes) {
                            if (attr.name.startsWith('data-') && attr.value.length > 2) {
                                results.data_attributes.push({
                                    element: el.tagName.toLowerCase(),
                                    attribute: attr.name,
                                    value: attr.value.slice(0, 200)
                                });
                            }
                        }

                    } catch(e) {
                        // Continue with next element
                    }
                }

                // 2. Extract comments
                const walker = document.createTreeWalker(
                    document, 
                    NodeFilter.SHOW_COMMENT, 
                    null, 
                    false
                );

                while (walker.nextNode()) {
                    const comment = walker.currentNode.nodeValue.trim();
                    if (comment.length > 5) {
                        results.comments.push(comment.slice(0, 300));
                    }
                }

                // 3. Extract script contents
                const scripts = Array.from(document.querySelectorAll('script'));
                for (const script of scripts.slice(0, 20)) {
                    const content = script.textContent || script.innerHTML || '';
                    if (content.trim().length > 10) {
                        results.scripts.push(content.slice(0, 1500));
                    }
                }

                // 4. Extract meta tags and other metadata
                const metas = Array.from(document.querySelectorAll('meta, title, [data-*]'));
                for (const meta of metas.slice(0, 50)) {
                    const content = meta.getAttribute('content') || 
                                  meta.getAttribute('value') || 
                                  meta.textContent || '';
                    if (content.trim().length > 2) {
                        results.meta_data.push({
                            tag: meta.tagName.toLowerCase(),
                            content: content.slice(0, 200),
                            attributes: Object.fromEntries(
                                Array.from(meta.attributes).map(attr => [attr.name, attr.value])
                            )
                        });
                    }
                }

                return results;
            })();
            '''

            try:
                res = self._page.evaluate(js)
                pieces = []

                # Process hidden elements
                for hidden in res.get("hidden_elements", []):
                    pieces.append(
                        f"HIDDEN {hidden.get('tag', '')} "
                        f"id={hidden.get('id', '')} "
                        f"class={hidden.get('className', '')} "
                        f"reason={hidden.get('hideReason', '')} "
                        f"text: {hidden.get('text', '')}"
                    )

                # Process comments
                for comment in res.get("comments", [])[:15]:
                    if comment.strip():
                        pieces.append(f"COMMENT: {comment}")

                # Process scripts
                for script in res.get("scripts", [])[:15]:
                    if script.strip():
                        pieces.append(f"SCRIPT: {script}")

                # Process metadata
                for meta in res.get("meta_data", [])[:20]:
                    pieces.append(f"META {meta.get('tag', '')}: {meta.get('content', '')}")

                # Process data attributes
                for data_attr in res.get("data_attributes", [])[:30]:
                    pieces.append(f"DATA-ATTR {data_attr.get('attribute', '')}: {data_attr.get('value', '')}")

                return pieces

            except Exception as e:
                # Enhanced fallback
                try:
                    content = self._page.content()

                    # Extract potential hidden content using regex
                    hidden_content = []

                    # Look for hidden divs/spans
                    hidden_matches = re.findall(
                        r'<(?:div|span)[^>]*(?:display:\s*none|visibility:\s*hidden)[^>]*>(.*?)</(?:div|span)>',
                        content, re.DOTALL
                    )
                    for match in hidden_matches[:10]:
                        clean_text = re.sub(r'<[^>]+>', '', match).strip()
                        if clean_text:
                            hidden_content.append(f"REGEX_HIDDEN: {clean_text[:200]}")

                    # Look for comments
                    comment_matches = re.findall(r'<!--(.*?)-->', content, re.DOTALL)
                    for comment in comment_matches[:10]:
                        if comment.strip():
                            hidden_content.append(f"REGEX_COMMENT: {comment.strip()[:200]}")

                    return hidden_content or [f"FALLBACK_CONTENT: {content[:3000]}"]

                except Exception:
                    return [f"ERROR_FINDING_HIDDEN: {str(e)}"]

    PlaywrightController = _LocalPlaywrightController  # expose fallback class


# ---- Utility ----
def text_preview(s: str, n: int = 250):
    if not s:
        return ""
    return s.strip()[:n]


# If you want to run this module directly for a quick test:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graph.py input.json")
        print("Or: python graph.py --visualize (to see graph structure)")
        sys.exit(1)

    if sys.argv[1] == "--visualize":
        print(get_graph_visualization())
        sys.exit(0)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = build_and_run_graph(payload)
    print(json.dumps(result, indent=2))
