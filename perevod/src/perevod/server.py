import argparse
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .translator import MODEL, Translator


class TranslateRequest(BaseModel):
    # the model reads at most 256 tokens, about a thousand characters of English
    text: str = Field(max_length=1000, examples=["Tom is afraid of the dark."])


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
        return await answer  # until run() gives the future its result

    async def collect(self):
        batch = [await self.queue.get()]  # waits for the first request, however long
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
            try:  # in a thread, so that the event loop keeps serving requests meanwhile
                translations = await asyncio.to_thread(self.translator.translate, texts)
            except Exception as error:  # noqa: BLE001, every request of the batch gets it
                for _, answer in batch:
                    if not answer.done():
                        answer.set_exception(error)
                continue
            for (_, answer), translation in zip(batch, translations, strict=True):
                if not answer.done():  # done already if its caller stopped waiting
                    answer.set_result(translation)


def create_app(model=MODEL, revision=None, max_batch=16, max_wait=0.005, demo=False):
    @asynccontextmanager
    async def lifespan(app):
        # before yield: when the server starts; after it: when it stops
        translator = Translator.from_pretrained(model, revision)
        app.state.batcher = Batcher(translator, max_batch, max_wait)
        worker = asyncio.create_task(app.state.batcher.run())  # runs next to the handlers
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
        import gradio as gr  # only the demo extra installs it

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

    import uvicorn  # only the serve extra installs it; the tests don't need it

    app = create_app(max_batch=args.max_batch, max_wait=args.max_wait_ms / 1000,
                     demo=args.demo)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
