import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.demo import MODEL_JSON, MODEL_NPZ

pytestmark = pytest.mark.skipif(not (MODEL_NPZ.exists() and MODEL_JSON.exists()),
                                reason="demo model not exported (python -m scripts.train_demo_model)")


@pytest.fixture(scope="module")
def base_url():
    from demo.server import Handler
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(base_url, body):
    request = urllib.request.Request(f"{base_url}/api/predict", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_valid_request_returns_prediction(base_url):
    status, body = post(base_url, json.dumps({"url": "https://www.wikipedia.org"}).encode())
    assert status == 200 and body["host"] == "wikipedia.org" and body["label"] in ("phishing", "legitimate")


@pytest.mark.parametrize("raw", [b"[1, 2]", b'"x.com"', b"{bad", b"\xff\xfe", b'{"url": ""}', b'{"url": "localhost"}'])
def test_malformed_requests_get_400_not_a_dropped_connection(base_url, raw):
    status, body = post(base_url, raw)
    assert status == 400 and "error" in body


def test_oversized_body_is_rejected(base_url):
    status, _ = post(base_url, json.dumps({"url": "a" * 20_000}).encode())
    assert status == 413
