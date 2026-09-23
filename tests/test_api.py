"""Tests for the HTTP-facing serialization layer (pure, no server needed)."""

import json
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from app.server import run_audit


K4 = {
    "nodes": ["A", "B", "C", "D"],
    "edges": [
        {"id": "e1", "u": "A", "v": "B", "length": 1},
        {"id": "e2", "u": "A", "v": "C", "length": 1},
        {"id": "e3", "u": "A", "v": "D", "length": 1},
        {"id": "e4", "u": "B", "v": "C", "length": 1},
        {"id": "e5", "u": "B", "v": "D", "length": 1},
        {"id": "e6", "u": "C", "v": "D", "length": 1},
    ],
    "start": "A",
}

REPORTED = {
    "nodes": ["A", "B", "C", "D", "E", "F"],
    "edges": [
        {"id": "e0", "u": "F", "v": "E", "length": 3},
        {"id": "e1", "u": "E", "v": "C", "length": 1},
        {"id": "e2", "u": "C", "v": "A", "length": 6},
        {"id": "e3", "u": "A", "v": "B", "length": 6},
        {"id": "e4", "u": "B", "v": "D", "length": 2},
        {"id": "e5", "u": "E", "v": "C", "length": 5},
        {"id": "e6", "u": "D", "v": "A", "length": 1},
        {"id": "e7", "u": "A", "v": "D", "length": 5},
        {"id": "e8", "u": "B", "v": "A", "length": 1},
    ],
    "start": "A",
}

REPORTED_CLASSIFICATION = {
    "e0": "required",
    "e1": "never",
    "e2": "required",
    "e3": "never",
    "e4": "optional",
    "e5": "never",
    "e6": "optional",
    "e7": "never",
    "e8": "optional",
}
REPORTED_COPIES = {"e0": 2, "e1": 1, "e2": 2, "e3": 1, "e4": 1,
                   "e5": 1, "e6": 2, "e7": 1, "e8": 2}


def assert_reported_payload(r):
    assert r["ok"] is True
    assert r["totalLength"] == 30
    assert r["addedLength"] == 11
    assert r["optimalCount"] == 2
    assert r["canonicalVector"] == "101000101"
    assert r["canonicalEdges"] == ["e0", "e2", "e6", "e8"]
    assert [e["id"] for e in r["edges"]] == [f"e{i}" for i in range(9)]
    for e in r["edges"]:
        assert e["classification"] == REPORTED_CLASSIFICATION[e["id"]]
        assert e["copies"] == REPORTED_COPIES[e["id"]]
        assert e["duplicated"] is (e["id"] in ("e0", "e2", "e6", "e8"))
    # closed route from A back to A with exact copy counts
    assert r["route"][0]["from"] == "A"
    assert r["route"][-1]["to"] == "A"
    used = {}
    cur = "A"
    for st in r["route"]:
        assert st["from"] == cur
        used[st["edgeId"]] = used.get(st["edgeId"], 0) + 1
        cur = st["to"]
    assert cur == "A"
    assert used == REPORTED_COPIES
    assert len(r["route"]) == 13
    assert sum(st["length"] for st in r["route"]) == 41


def test_reported_network_payload():
    assert_reported_payload(run_audit(REPORTED))


def test_reported_network_payload_order_invariant():
    # edges entered in a deliberately scrambled order; identifiers sort the
    # presentation so the exact payload conclusions must be identical
    payload = json.loads(json.dumps(REPORTED))
    random.Random(123).shuffle(payload["edges"])
    assert_reported_payload(run_audit(payload))


def test_success_payload():
    r = run_audit(K4)
    assert r["ok"] is True
    assert r["optimalCount"] == 3
    assert r["addedLength"] == 2
    assert r["totalLength"] == 6
    assert r["canonicalVector"] == "001100"
    assert r["canonicalEdges"] == ["e3", "e4"]
    assert len(r["route"]) == 8
    # positions cover exactly duplicated canonical edges' extra copies
    counts = {e["id"]: len(r["positions"][str(e["index"])]) for e in r["edges"]}
    duplicated = set(r["canonicalEdges"])
    for eid, n in counts.items():
        assert n == (2 if eid in duplicated else 1)
    # route closes at start
    assert r["route"][0]["from"] == "A"
    assert r["route"][-1]["to"] == "A"


def test_eulerian_payload():
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "a", "u": "A", "v": "B", "length": 3},
            {"id": "b", "u": "B", "v": "C", "length": 4},
            {"id": "c", "u": "C", "v": "A", "length": 5},
        ],
        "start": "B",
    }
    r = run_audit(payload)
    assert r["ok"]
    assert r["eulerian"] is True
    assert r["addedLength"] == 0
    assert r["optimalCount"] == 1
    assert r["canonicalVector"] == "000"
    assert r["canonicalEdges"] == []
    assert all(e["classification"] == "never" for e in r["edges"])


def test_failure_payload_locations():
    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "Z", "length": 1}],
                   "start": "A"})
    assert r["ok"] is False
    assert "edges" in r["fields"]
    assert r["locations"][0]["row"] == 0

    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": -2}],
                   "start": "A"})
    assert not r["ok"] and "正整数" in r["error"]

    r = run_audit({"nodes": ["A", "B"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": 1}],
                   "start": "Q"})
    assert not r["ok"] and r["fields"] == ["start"]

    r = run_audit({"nodes": ["A", "B", "C"],
                   "edges": [{"id": "x", "u": "A", "v": "B", "length": 1}],
                   "start": "A"})
    assert not r["ok"] and "不连通" in r["error"]


def test_malformed_payload_is_safe():
    assert run_audit({})["ok"] is False
    assert run_audit({"nodes": "ab", "edges": None})["ok"] is False


# ---------------------------------------------------------------------------
# Real socket smoke test: boots gunicorn and hits the live HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "gunicorn",
            "-w", "1", "-b", f"127.0.0.1:{port}",
            "--timeout", "30", "app.server:app",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("gunicorn exited during startup")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    else:
        proc.terminate()
        raise RuntimeError("server did not become ready")
    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _post(base, body):
    req = urllib.request.Request(
        base + "/api/audit",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_live_health_and_reported_audit(live_server):
    with urllib.request.urlopen(live_server + "/health", timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read().decode())["status"] == "ok"

    status, body = _post(live_server, REPORTED)
    assert status == 200
    assert_reported_payload(body)

    # order invariance over the real interface as well
    scrambled = json.loads(json.dumps(REPORTED))
    random.Random(99).shuffle(scrambled["edges"])
    status, body = _post(live_server, scrambled)
    assert status == 200
    assert_reported_payload(body)

