from typing import List, AsyncGenerator
import asyncio
import threading
import ollama  # pip install ollama
# =============================
# Ollama Streaming
# =============================
async def ollama_stream_chat(conversation: List[dict], model: str, max_tokens: int, keep_alive: int | str = 300) -> AsyncGenerator[str, None]:
    """Async generator yielding text chunks from Ollama chat streaming.

    conversation: list of {'role': 'system'|'user'|'assistant', 'content': str}
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            # streaming=True returns incremental responses
            for part in ollama.chat(
                model=model,
                messages=conversation,
                stream=True,
                options={"num_predict": max_tokens},
                keep_alive=keep_alive,
            ):
                try:
                    msg = part.get('message', {})
                    content = msg.get('content')
                    if content:
                        asyncio.run_coroutine_threadsafe(q.put(content), loop)
                except Exception:
                    continue
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(f"[Error: {e} ]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        chunk = await q.get()
        if chunk is None:
            break
        yield chunk