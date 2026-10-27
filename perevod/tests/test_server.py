import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from perevod.server import create_app


@pytest.fixture
def client(tiny_dir):
    app = create_app(str(tiny_dir), max_wait=0.05)
    with TestClient(app) as client:  # entering it runs the lifespan: model and batcher
        yield client


def test_the_health_check(client):
    assert client.get("/health").json()["status"] == "ok"


def test_the_api_translates_like_the_package(client, translator, pairs):
    english = next(iter(pairs))
    response = client.post("/translate", json={"text": english})
    assert response.json() == {"translation": translator.translate(english)}


def test_a_text_too_long_is_refused(client):
    assert client.post("/translate", json={"text": "tom " * 2000}).status_code == 422


def test_requests_in_one_batch_get_their_own_translations(tiny_dir, pairs):
    app = create_app(str(tiny_dir), max_wait=0.2)

    async def ask_all():
        async with app.router.lifespan_context(app):  # the model and the batcher
            transport = httpx.ASGITransport(app=app)
            base_url = "http://perevod"
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                requests = []
                for english in pairs:
                    requests.append(client.post("/translate", json={"text": english}))
                return await asyncio.gather(*requests)

    responses = asyncio.run(ask_all())
    assert app.state.batcher.batch_sizes == [len(pairs)]  # one batch for all of them
    for response, russian in zip(responses, pairs.values(), strict=True):
        assert response.json()["translation"] == russian
