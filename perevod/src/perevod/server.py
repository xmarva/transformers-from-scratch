import argparse
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .translator import MODEL, Translator


class TranslateRequest(BaseModel):
    text: str = Field(max_length=5000, examples=["I think, therefore I am."])


class TranslateResponse(BaseModel):
    translation: str


class Batcher:
    """Translates requests that arrive close together as one batch.

    A batch closes when it holds max_batch requests or max_wait seconds after its first one.
    """

    def __init__(self, translator, max_batch=16, max_wait=0.005):
        self.translator = translator
        self.max_batch, self.max_wait = max_batch, max_wait
        self.queue = asyncio.Queue()
        self.batch_sizes = []

    async def translate(self, text):
        answer = asyncio.get_running_loop().create_future()
        await self.queue.put((text, answer))
        return await answer

    async def collect(self):
        batch = [await self.queue.get()]
        deadline = time.monotonic() + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def run(self):
        while True:
            batch = await self.collect()
            self.batch_sizes.append(len(batch))
            texts = [text for text, _ in batch]
            # in a thread, so that the server keeps accepting requests meanwhile
            job = asyncio.ensure_future(asyncio.to_thread(self.translator.translate, texts))
            await asyncio.wait([job])
            if job.exception() is not None:  # every request of the batch gets the error
                for _, answer in batch:
                    answer.set_exception(job.exception())
                continue
            for (_, answer), translation in zip(batch, job.result(), strict=True):
                answer.set_result(translation)


def create_app(model=MODEL, revision=None, max_batch=16, max_wait=0.005, demo=False):
    @asynccontextmanager
    async def lifespan(app):
        translator = Translator.from_pretrained(model, revision)
        app.state.batcher = Batcher(translator, max_batch, max_wait)
        worker = asyncio.create_task(app.state.batcher.run())
        yield
        worker.cancel()

    app = FastAPI(title="perevod", version=__version__, lifespan=lifespan)

    @app.post("/translate")
    async def translate(request: TranslateRequest) -> TranslateResponse:
        translation = await app.state.batcher.translate(request.text)
        return TranslateResponse(translation=translation)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    if demo:
        import gradio as gr

        from .demo import build_demo

        app = gr.mount_gradio_app(app, build_demo(app), path="/")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="perevod-server",
                                     description="Serve perevod over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=16, help="1 turns batching off")
    parser.add_argument("--max-wait-ms", type=float, default=5.0)
    parser.add_argument("--demo", action="store_true", help="serve a Gradio page at /")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(max_batch=args.max_batch, max_wait=args.max_wait_ms / 1000,
                     demo=args.demo)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
