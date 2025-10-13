    
import asyncio
import sys

def setup_windows_event_loop():
    """Set up Windows-specific event loop policy if on Windows platform."""
    if sys.platform.startswith('win'):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except Exception:
            pass