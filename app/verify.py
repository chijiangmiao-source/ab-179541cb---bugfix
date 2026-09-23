"""One-shot verification service.

Runs, in order:
  1. the code test suite (pytest);
  2. domain boundary checks against the solver directly
       - odd-degree network: exact co-optimal count / classification,
       - Eulerian network: zero augmentation boundary;
  3. an HTTP smoke test against the running web service
       - GET /health,
       - POST /api/audit success case,
       - POST /api/audit validation failure case.

BASE_URL may point at an already running instance (compose sets it to the
web service).  When unset, the script starts gunicorn locally on an
ephemeral port and tears it down afterwards.

Exits 0 only when every stage passes; the failed stage is reported.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


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

TRIANGLE = {
    "nodes": ["A", "B", "C"],
    "edges": [
        {"id": "a", "u": "A", "v": "B", "length": 3},
        {"id": "b", "u": "B", "v": "C", "length": 4},
        {"id": "c", "u": "C", "v": "A", "length": 5},
    ],
    "start": "B",
}

BAD = {
    "nodes": ["A", "B"],
    "edges": [{"id": "x", "u": "A", "v": "A", "length": 1}],
    "start": "A",
}

# Six-node inspection network: two co-optimal minimum augmentation sets,
# canonical 0-preferred vector 101000101; edges entered in identifier order.
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

# Same network with the edge rows reversed; audit conclusions must be
# identical because they are keyed by edge identifier.
INSPECTION_REVERSED = {
    "nodes": INSPECTION["nodes"],
    "edges": list(reversed(INSPECTION["edges"])),
    "start": "A",
}


def check_inspection(body):
    check(body.get("totalLength") == 30, "巡检管网原总长 30")
    check(body.get("addedLength") == 11, "巡检管网最小增加长度 11")
    check(body.get("optimalCount") == 2, "巡检管网并列最优数量为 2")
    check(
        body.get("canonicalVector") == "101000101",
        "巡检管网规范位向量 101000101",
    )
    check(
        body.get("canonicalEdges") == ["e0", "e2", "e6", "e8"],
        "巡检管网规范重复边为 e0、e2、e6、e8",
    )
    by_id = {e["id"]: e["classification"] for e in body.get("edges", [])}
    check(
        by_id
        == {
            "e0": "required", "e1": "never", "e2": "required",
            "e3": "never", "e4": "optional", "e5": "never",
            "e6": "optional", "e7": "never", "e8": "optional",
        },
        "巡检管网各管段归属：e0/e2 必重复，e4/e6/e8 可重复，其余从不重复",
    )
    route = body.get("route", [])
    check(
        len(route) == 13
        and route[0].get("from") == "A"
        and route[-1].get("to") == "A",
        "巡检管网路线为 13 步并从 A 闭合回 A",
    )
    usage = {}
    cur = "A"
    contiguous = True
    for st in route:
        if st.get("from") != cur:
            contiguous = False
        usage[st["edgeId"]] = usage.get(st["edgeId"], 0) + 1
        cur = st.get("to")
    check(contiguous and cur == "A", "巡检管网路线逐步连续且终点为 A")
    check(
        all(usage.get(eid) == 2 for eid in ("e0", "e2", "e6", "e8"))
        and all(usage.get(eid) == 1 for eid in ("e1", "e3", "e4", "e5", "e7")),
        "巡检管网路线副本数与规范方案一致（e0/e2/e6/e8 各 2 次，其余各 1 次）",
    )


def stage(name):
    print(f"\n=== {name} ===", flush=True)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


# ---------------------------------------------------------------------------
# Stage 1: pytest
# ---------------------------------------------------------------------------


def run_pytest() -> bool:
    stage("1/3 代码测试 (pytest)")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Stage 2: solver domain boundaries
# ---------------------------------------------------------------------------


def run_domain_checks() -> bool:
    stage("2/3 奇度同优分类 & 欧拉零增程边界")
    from app.solver import audit

    # odd network: K4 unit weights -> 4 odd vertices, 3 distinct optima
    r = audit(K4["nodes"], K4["edges"], K4["start"])
    check(tuple(r.odd_vertices) == ("A", "B", "C", "D"), "K4 有四个奇度节点")
    check(r.added_length == 2, "K4 最小增程为 2")
    check(r.optimal_count == 3, "K4 同优集合数量恰为 3")
    check(r.bit_vector == "001100", "K4 规范位向量为 001100（0 优先）")
    check(
        set(r.classification.values()) == {"optional"},
        "K4 每条边均为可重复（无必重复/从不重复）",
    )
    check(
        len(r.route) == sum(r.multiplicity)
        and r.route[0].frm == "A"
        and r.route[-1].to == "A",
        "K4 规范路线闭合且副本数吻合",
    )

    # optional-vs-required mix: two equal shortest routes + a long edge
    mix_nodes = ["A", "B", "X", "Y"]
    mix_edges = [
        {"id": "p1", "u": "A", "v": "X", "length": 1},
        {"id": "p2", "u": "X", "v": "B", "length": 1},
        {"id": "p3", "u": "A", "v": "Y", "length": 1},
        {"id": "p4", "u": "Y", "v": "B", "length": 1},
        {"id": "long", "u": "A", "v": "B", "length": 3},
    ]
    r2 = audit(mix_nodes, mix_edges, "A")
    check(r2.optimal_count == 2, "双桥结构同优集合数量为 2")
    # edges are identifier-sorted: long(0), p1(1), p2(2), p3(3), p4(4)
    check(all(r2.classification[i] == "optional" for i in (1, 2, 3, 4)),
          "两条等长路径上的边为可重复")
    check(r2.classification[0] == "never", "长边从不重复")
    # candidate sets {p1,p2}=01100 and {p3,p4}=00011 -> 0-pref = 00011
    check(r2.bit_vector == "00011", "双桥规范位向量 00011")

    # Eulerian boundary: triangle
    r3 = audit(TRIANGLE["nodes"], TRIANGLE["edges"], TRIANGLE["start"])
    check(r3.is_eulerian, "三角形为欧拉管网")
    check(r3.added_length == 0, "欧拉管网零增程")
    check(r3.optimal_count == 1, "欧拉管网同优集合数量为 1（空集）")
    check(set(r3.canonical_set) == set(), "规范重复集合为空")
    check(r3.bit_vector == "000", "位向量全 0")
    check(
        all(v == "never" for v in r3.classification.values()),
        "所有边归属为从不重复",
    )
    check(len(r3.route) == 3 and r3.route[-1].to == "B",
          "欧拉回路从检修口出发并返回")

    # six-node inspection network: two distinct optimal sets where one set
    # admits two matching/path decompositions (the original miscount source)
    ri = audit(
        INSPECTION["nodes"], INSPECTION["edges"], INSPECTION["start"]
    )
    check(tuple(ri.odd_vertices) == tuple("ABCDEF"), "巡检管网六个奇度节点")
    check(ri.total_length == 30 and ri.added_length == 11,
          "巡检管网原长 30、最小增程 11（路线总长 41）")
    check(ri.optimal_count == 2, "巡检管网不同最优集合恰为 2 个")
    check(ri.bit_vector == "101000101", "巡检管网规范位向量 101000101")
    by_id = {ri.edges[i].eid: ri.classification[i] for i in range(9)}
    check(
        by_id
        == {
            "e0": "required", "e1": "never", "e2": "required",
            "e3": "never", "e4": "optional", "e5": "never",
            "e6": "optional", "e7": "never", "e8": "optional",
        },
        "巡检管网各边归属：e0/e2 必，e4/e6/e8 可，e1/e3/e5/e7 否",
    )
    check(
        ri.multiplicity == (2, 1, 2, 1, 1, 1, 2, 1, 2),
        "巡检管网规范副本数 (2,1,2,1,1,1,2,1,2)",
    )
    check(
        ri.route[0].frm == "A" and ri.route[-1].to == "A",
        "巡检管网规范路线从 A 闭合回 A",
    )
    seen = [0] * 9
    cur = "A"
    contiguous = True
    for st in ri.route:
        if st.frm != cur:
            contiguous = False
        seen[st.edge_index] += 1
        cur = st.to
    check(
        contiguous and cur == "A" and tuple(seen) == ri.multiplicity,
        "巡检管网规范路线逐步连续且副本使用数与规范方案一致",
    )
    rr = audit(
        INSPECTION_REVERSED["nodes"],
        INSPECTION_REVERSED["edges"],
        "A",
    )
    check(
        rr.bit_vector == "101000101" and rr.optimal_count == 2,
        "改变录入顺序后按标识排列结论不变",
    )
    return True


# ---------------------------------------------------------------------------
# Stage 3: HTTP smoke
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(base: str, proc, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    return False


def _post(base: str, path: str, body: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def run_http_smoke() -> bool:
    stage("3/3 HTTP 冒烟 (health / audit 成功 / audit 失败)")
    base = os.environ.get("BASE_URL", "").rstrip("/")
    proc = None
    if not base:
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        print(f"  启动临时 gunicorn: {base}")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "gunicorn",
                "-w", "1", "-b", f"127.0.0.1:{port}",
                "--timeout", "30", "app.server:app",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        if not _wait_ready(base, proc):
            print("  FAIL: 服务未在限定时间内就绪")
            return False
        print("  PASS: GET /health -> 200")

        status, body = _post(base, "/api/audit", K4)
        check(status == 200 and body.get("ok") is True,
              "POST /api/audit 奇度管网审计成功")
        check(body.get("optimalCount") == 3, "HTTP 返回同优数量为 3")
        check(body.get("addedLength") == 2, "HTTP 返回最小增程为 2")
        check(body.get("canonicalVector") == "001100",
              "HTTP 返回规范位向量 001100")
        check(len(body.get("route", [])) == 8,
              "HTTP 返回 8 步闭合路线（6 原边 + 2 重复副本）")

        # real POST /api/audit for the reported inspection network
        status, body = _post(base, "/api/audit", INSPECTION)
        check(status == 200 and body.get("ok") is True,
              "POST /api/audit 巡检管网审计成功")
        check_inspection(body)

        status, body = _post(base, "/api/audit", INSPECTION_REVERSED)
        check(
            status == 200
            and body.get("optimalCount") == 2
            and body.get("canonicalVector") == "101000101"
            and body.get("addedLength") == 11,
            "HTTP 逆序录入巡检管网返回同一审计结论",
        )

        status, body = _post(base, "/api/audit", TRIANGLE)
        check(body.get("ok") and body.get("addedLength") == 0,
              "HTTP 欧拉管网零增程")

        status, body = _post(base, "/api/audit", BAD)
        check(status == 200 and body.get("ok") is False,
              "非法输入返回 ok=false（HTTP 层仍为 200）")
        check("edges" in body.get("fields", []), "自环错误定位到管段表")
        check(any(l.get("row") == 0 for l in body.get("locations", [])),
              "错误位置精确到第 1 行")
        return True
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    failures = []
    for name, fn in (
        ("代码测试", run_pytest),
        ("同优分类/零增程边界", run_domain_checks),
        ("HTTP 冒烟", run_http_smoke),
    ):
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 - report any failure
            ok = False
            print(f"  FAIL: {name} 抛出异常: {exc!r}")
        if not ok:
            failures.append(name)

    print("\n========================================")
    if failures:
        print("VERIFY 失败：" + "、".join(failures))
        return 1
    print("VERIFY 全部通过：测试 / 奇度同优分类 / 欧拉零增程 / HTTP 冒烟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
