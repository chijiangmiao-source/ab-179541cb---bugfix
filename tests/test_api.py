"""Tests for the HTTP-facing serialization layer (pure, no server needed)."""

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


# The reported six-node inspection network (edges intentionally NOT given in
# identifier order): two co-optimal duplicate sets, canonical 101000101.
INSPECTION = {
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


def test_inspection_payload_exact():
    r = run_audit(INSPECTION)
    assert r["ok"] is True
    assert r["totalLength"] == 30
    assert r["addedLength"] == 11
    assert r["optimalCount"] == 2
    assert r["canonicalVector"] == "101000101"
    assert r["canonicalEdges"] == ["e0", "e2", "e6", "e8"]
    by_id = {e["id"]: e for e in r["edges"]}
    assert by_id["e0"]["classification"] == "required"
    assert by_id["e2"]["classification"] == "required"
    for eid in ("e4", "e6", "e8"):
        assert by_id[eid]["classification"] == "optional"
    for eid in ("e1", "e3", "e5", "e7"):
        assert by_id[eid]["classification"] == "never"
    # canonical duplication reflected per edge
    for eid in ("e0", "e2", "e6", "e8"):
        assert by_id[eid]["duplicated"] is True
        assert by_id[eid]["copies"] == 2
    for eid in ("e1", "e3", "e4", "e5", "e7"):
        assert by_id[eid]["duplicated"] is False
        assert by_id[eid]["copies"] == 1
    # closed route starting/ending at A with exact copy usage
    route = r["route"]
    assert len(route) == 13
    assert route[0]["from"] == "A" and route[-1]["to"] == "A"
    usage = {}
    cur = "A"
    for st in route:
        assert st["from"] == cur
        usage[st["edgeId"]] = usage.get(st["edgeId"], 0) + 1
        cur = st["to"]
    assert cur == "A"
    for eid in ("e0", "e2", "e6", "e8"):
        assert usage[eid] == 2
    for eid in ("e1", "e3", "e4", "e5", "e7"):
        assert usage[eid] == 1


def test_inspection_payload_order_independent():
    payload = {
        "nodes": INSPECTION["nodes"],
        "edges": list(reversed(INSPECTION["edges"])),
        "start": "A",
    }
    r = run_audit(payload)
    assert r["optimalCount"] == 2
    assert r["addedLength"] == 11
    assert r["canonicalVector"] == "101000101"
    assert r["canonicalEdges"] == ["e0", "e2", "e6", "e8"]
    by_id = {e["id"]: e["classification"] for e in r["edges"]}
    assert by_id == {
        "e0": "required", "e1": "never", "e2": "required", "e3": "never",
        "e4": "optional", "e5": "never", "e6": "optional",
        "e7": "never", "e8": "optional",
    }


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
