import io
import base64
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright

async def annotate_page(url: str, max_elements: int = 80) -> Tuple[str, List[Dict[str, Any]]]:
    """Open URL in Playwright, capture screenshot, and extract interactive elements."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(800)

        # find interactive elements (anchors, buttons, inputs)
        elements = await page.eval_on_selector_all(
            "a,button,input,textarea,select,[role='button'],[onclick]",
            """(els) => els.map(e => {
                const r = e.getBoundingClientRect();
                const text = (e.innerText || e.value || e.placeholder || e.getAttribute('aria-label') || '').toString().slice(0,200);
                return {text, tag: e.tagName, x: r.x, y: r.y, width: r.width, height: r.height};
            })"""
        )

        filtered = [el for el in elements if el["width"] > 4 and el["height"] > 4][:max_elements]
        labels = []
        for i, el in enumerate(filtered, start=1):
            labels.append({
                "id": i,
                "bbox": {"x": el["x"], "y": el["y"], "width": el["width"], "height": el["height"]},
                "text": el["text"],
                "type": el["tag"]
            })

        png = await page.screenshot(full_page=True)
        await browser.close()

        img = Image.open(io.BytesIO(png)).convert("RGBA")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        for lbl in labels:
            x = lbl["bbox"]["x"]; y = lbl["bbox"]["y"]; w = lbl["bbox"]["width"]; h = lbl["bbox"]["height"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0, 255), width=2)
            draw.text((x + 2, max(0, y - 12)), str(lbl["id"]), fill=(255, 255, 255, 255), font=font)

        out = io.BytesIO()
        img.save(out, format="PNG")
        b64 = base64.b64encode(out.getvalue()).decode()
        return b64, labels
