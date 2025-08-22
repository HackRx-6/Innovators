# Enhanced version of your PlaywrightController with better hidden element detection
from playwright.sync_api import sync_playwright
from typing import List, Dict, Any, Optional
import time
import re

class PlaywrightController:
    def __init__(self, headless: bool = True):
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
        for step in plan:
            t = step.get("type")
            if t == "navigate":
                url = step.get("url") or default_url
                if not url:
                    log.append({"error":"no url provided for navigate step"})
                    continue
                r = self._page.goto(url, wait_until="networkidle", timeout=30000)
                log.append({"navigate": url, "status": getattr(r, "status", None)})
                # Longer wait for dynamic content
                time.sleep(2.0)
            elif t == "click":
                sel = step.get("selector")
                try:
                    self._page.click(sel, timeout=5000)
                    log.append({"click": sel})
                    time.sleep(0.5)
                except Exception as e:
                    log.append({"click_error": str(e), "selector": sel})
            elif t == "wait":
                secs = float(step.get("secs", 1.0))
                time.sleep(secs)
            elif t == "run_js":
                script = step.get("script", "")
                try:
                    res = self._page.evaluate(script)
                    log.append({"run_js": script[:120], "result": str(res)[:500]})
                except Exception as e:
                    log.append({"run_js_error": str(e)})
            elif t == "extract":
                sel = step.get("selector")
                try:
                    # Try to find elements with longer timeout
                    elements = self._page.locator(sel)
                    count = elements.count()
                    if count > 0:
                        texts = []
                        for i in range(min(count, 10)):  # Limit to first 10 elements
                            try:
                                el_text = elements.nth(i).inner_text(timeout=3000)
                                if el_text.strip():
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
                        log.append({"extract_error": f"No elements found for selector: {sel}", "selector": sel})
                except Exception as e:
                    log.append({"extract_error": str(e), "selector": sel})
            elif t == "extract_full" or step.get("extract_full"):
                # Enhanced full extraction with better hidden element detection
                texts = []
                texts.extend(self._find_hidden_texts())
                
                # Also try to get all visible text
                try:
                    body_text = self._page.locator("body").inner_text(timeout=5000)
                    if body_text.strip():
                        texts.append(f"---VISIBLE-CONTENT---\n{body_text}")
                except:
                    pass
                
                # Get page content as fallback
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
        # Enhanced summary with key information
        summary = self._create_enhanced_summary(combined_text)
        return {"summary": summary, "combined_text": combined_text, "log": log}

    def _create_enhanced_summary(self, text: str) -> str:
        """Create an enhanced summary that highlights potential challenge names"""
        if not text:
            return "No content extracted"
        
        import re
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
        
        # Add beginning of content
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
                        computedStyle.position === 'absolute' && (
                            parseFloat(computedStyle.left) < -9999 ||
                            parseFloat(computedStyle.top) < -9999
                        ) ||
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
                        
                        // Check for challenge-related attributes or classes
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
                # Try basic content extraction
                content = self._page.content()
                
                # Extract potential hidden content using regex
                import re
                hidden_content = []
                
                # Look for hidden divs/spans
                hidden_matches = re.findall(r'<(?:div|span)[^>]*(?:display:\s*none|visibility:\s*hidden)[^>]*>(.*?)</(?:div|span)>', content, re.DOTALL)
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
                
            except:
                return [f"ERROR_FINDING_HIDDEN: {e}"]


def text_preview(s: str, n: int = 250):
    if not s: 
        return ""
    return s.strip()[:n]