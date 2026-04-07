"""Browser tool: JS-rendered web page fetching via Playwright.

For pages that require JavaScript rendering (SPAs, dynamic content),
this tool launches a headless Chromium browser to load the page fully
before extracting content.

Falls back to simple HTTP fetch (via web_fetch) if Playwright is unavailable.

Dependencies: playwright (pip install playwright && playwright install chromium)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

_BROWSER_TIMEOUT = 30  # seconds
_MAX_CONTENT_LENGTH = 15000  # chars

# Cache Playwright availability
_playwright_available: bool | None = None


async def _check_playwright() -> bool:
    """Check if Playwright is installed and usable."""
    global _playwright_available
    if _playwright_available is not None:
        return _playwright_available
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
        _playwright_available = True
        logger.info("Playwright available (Chromium)")
    except Exception as e:
        _playwright_available = False
        logger.info("Playwright not available: %s", e)
    return _playwright_available


async def _extract_page_content(page: Any) -> str:
    """Extract readable text content from a Playwright page."""
    # Remove script, style, nav, footer elements
    await page.evaluate("""
        () => {
            for (const sel of ['script', 'style', 'nav', 'footer', 'header',
                                'iframe', 'noscript', '.ad', '.sidebar']) {
                document.querySelectorAll(sel).forEach(el => el.remove());
            }
        }
    """)

    # Get text content
    text = await page.evaluate("() => document.body?.innerText || ''")
    return text.strip()


async def _browser_fetch(params: dict[str, Any], **deps: Any) -> str:
    """Fetch a web page with full JS rendering via Playwright.

    Use this for JavaScript-heavy pages (SPAs, dynamic content) where
    simple HTTP fetch returns empty or incomplete content.
    """
    url = params.get("url", "")
    if not url:
        return "Error: url is required"

    wait_for = params.get("wait_for", "")
    screenshot = params.get("screenshot", False)

    # Try Playwright first
    if await _check_playwright():
        try:
            return await _fetch_with_playwright(url, wait_for, screenshot)
        except Exception as e:
            logger.warning("Playwright fetch failed for %s: %s", url, e)
            return f"Error: browser fetch failed: {e}"
    else:
        # Fallback: use web_fetch tool
        tools = deps.get("tools", {})
        web_fetch = tools.get("web_fetch")
        if web_fetch:
            return await web_fetch.fn({"url": url}, **deps)
        return "Error: Playwright not available and no web_fetch fallback"


async def _fetch_with_playwright(
    url: str, wait_for: str = "", screenshot: bool = False
) -> str:
    """Internal: fetch page using Playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            # Navigate with timeout
            await page.goto(url, wait_until="networkidle", timeout=_BROWSER_TIMEOUT * 1000)

            # Optional: wait for specific selector
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=10000)
                except Exception:
                    logger.debug("wait_for selector '%s' timed out", wait_for)

            # Get page title
            title = await page.title()

            # Extract text content
            text = await _extract_page_content(page)

            # Truncate
            if len(text) > _MAX_CONTENT_LENGTH:
                text = text[:_MAX_CONTENT_LENGTH] + "\n... (content truncated)"

            # Optional screenshot
            screenshot_info = ""
            if screenshot:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".png", prefix="_browser_", dir="/tmp", delete=False
                )
                await page.screenshot(path=tmp.name, full_page=False)
                screenshot_info = f"\n[Screenshot saved: {tmp.name}]"

            result = f"# {title}\nURL: {url}\n\n{text}{screenshot_info}"
            return result

        finally:
            await browser.close()


BROWSER_FETCH = ToolDef(
    name="browser_fetch",
    description=(
        "Fetch a web page with full JavaScript rendering using a headless browser. "
        "Use this for JavaScript-heavy pages (SPAs, React apps, dynamic content) where "
        "web_fetch returns empty or incomplete content. Slower than web_fetch but handles "
        "dynamic rendering. Optionally wait for a CSS selector or take a screenshot."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "wait_for": {
                "type": "string",
                "description": "CSS selector to wait for before extracting content (optional)",
            },
            "screenshot": {
                "type": "boolean",
                "description": "Take a screenshot of the page (default: false)",
            },
        },
        "required": ["url"],
    },
    fn=_browser_fetch,
    category="external",
)
