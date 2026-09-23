"""Chinese postman / route inspection core.

Undirected multigraph (parallel edges allowed, self loops forbidden) with
positive integer edge lengths.  Finds the minimum added length needed for all
vertices to have even degree, the exact number of optimal duplicate sets, the
canonical duplicate set (0-preferred bit vector in identifier order),
per-edge classification, and a closed Euler tour for the canonical
augmentation.

Exact counting by meet-in-the-middle over edge subsets
------------------------------------------------------
A duplicate set is a T-join for the set T of originally odd vertices: the
subgraph formed by the duplicated edges has odd degree exactly on T.  The
number of *distinct* minimum-weight T-joins cannot in general be read off
the shortest-path matching DP: a join that is a symmetric difference of
shortest paths can admit several matching/path decompositions (when the
shortest paths of one matching meet at vertices, the pairing re-splits into
another minimum matching yielding the same edge set).

With at most 32 edges the exact answer is obtained by meet-in-the-middle:
split edges into two halves of <=16 edges, enumerate all 2**16 subsets per
half, and aggregate them by (parity bit vector over the vertices, weight).
A full subset is uniquely the disjoint union of its two halves, so joining
the half tables counts distinct edge sets with no over-counting:

    optimum  = min over p of  minW_L[p] + minW_R[T xor p]
    count    = sum over p of  C_L[p][w] * C_R[T xor p][optimum - w]

Per-edge membership among the optimum sets uses the same half tables built
with that edge forced inside the subset; the 0-preferred canonical vector is
fixed greedily by testing whether an optimum join still exists with the
identifier-order prefix pinned (feasibility = min half weights under pins
still sum to the optimum).
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

INF = 10**30
TOKEN_RE = re.compile(r"^[!-~]+$")  # printable non-space ASCII


class AuditError(ValueError):
    """Validation failure with page locations."""

    def __init__(
        self,
        message: str,
        fields: Sequence[str] = (),
        locations: Sequence[dict] = (),
    ):
        super().__init__(message)
        self.message = message
        self.fields = tuple(fields)
        self.locations = list(locations)


@dataclass(frozen=True)
class Edge:
    eid: str
    u: str
    v: str
    length: int
    index: int


@dataclass(frozen=True)
class RouteStep:
    edge_index: int
    edge_id: str
    frm: str
    to: str
    length: int
    duplicate_no: int  # which copy of this edge, 1-based, in traversal order


@dataclass
class AuditResult:
    nodes: List[str]
    edges: List[Edge]
    start: str
    odd_vertices: Tuple[str, ...]
    components: Tuple[Tuple[str, ...], ...]
    total_length: int
    added_length: int
    optimal_count: int
    canonical_set: FrozenSet[int]
    bit_vector: str
    classification: Dict[int, str]  # required | optional | never
    multiplicity: Tuple[int, ...]
    route: Tuple[RouteStep, ...]

    @property
    def is_eulerian(self) -> bool:
        return not self.odd_vertices


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_positive_int(value, eid: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    if isinstance(value, int):
        length = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        length = int(value.strip())
    else:
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    if length <= 0:
        raise AuditError(f"管段 {eid} 长度必须为正整数", ("edges",))
    return length


def validate_input(
    nodes: Sequence[str], raw_edges: Sequence[dict], start: Optional[str]
) -> Tuple[List[str], List[Edge]]:
    clean_nodes: List[str] = []
    node_rows: Dict[str, int] = {}
    for i, raw in enumerate(nodes):
        name = ("" if raw is None else str(raw)).strip()
        if name == "":
            continue
        loc = [{"field": "nodes", "row": i}]
        if not TOKEN_RE.match(name):
            raise AuditError(
                f"节点 {name!r} 必须为非空白 ASCII 字符", ("nodes",), loc
            )
        if name in node_rows:
            raise AuditError(
                f"节点 {name!r} 重复",
                ("nodes",),
                loc + [{"field": "nodes", "row": node_rows[name]}],
            )
        node_rows[name] = i
        clean_nodes.append(name)

    if not (2 <= len(clean_nodes) <= 18):
        raise AuditError(
            f"唯一节点数量为 {len(clean_nodes)}，必须在 2 至 18 之间",
            ("nodes",),
            [{"field": "nodes"}],
        )

    if not raw_edges:
        raise AuditError("至少需要 1 条管段", ("edges",), [{"field": "edges"}])
    if len(raw_edges) > 32:
        raise AuditError(
            f"管段数量为 {len(raw_edges)}，不能超过 32",
            ("edges",),
            [{"field": "edges"}],
        )

    edges: List[Edge] = []
    id_rows: Dict[str, int] = {}
    for i, re_ in enumerate(raw_edges):
        loc = [{"field": "edges", "row": i}]
        eid = str(re_.get("id", "") or "").strip()
        if not eid:
            raise AuditError(
                f"第 {i + 1} 条管段缺少唯一标识", ("edges",), loc
            )
        if not TOKEN_RE.match(eid):
            raise AuditError(
                f"管段标识 {eid!r} 必须为非空白 ASCII 字符", ("edges",), loc
            )
        if eid in id_rows:
            raise AuditError(
                f"管段标识 {eid!r} 重复",
                ("edges",),
                loc + [{"field": "edges", "row": id_rows[eid]}],
            )
        id_rows[eid] = i

        u = str(re_.get("u", "") or "").strip()
        v = str(re_.get("v", "") or "").strip()
        if u not in node_rows:
            raise AuditError(
                f"管段 {eid} 的端点 {u or '(空)'} 不是已声明节点",
                ("edges",),
                loc,
            )
        if v not in node_rows:
            raise AuditError(
                f"管段 {eid} 的端点 {v or '(空)'} 不是已声明节点",
                ("edges",),
                loc,
            )
        if u == v:
            raise AuditError(
                f"管段 {eid} 为自环（{u}），禁止自环", ("edges",), loc
            )

        length = _as_positive_int(re_.get("length", None), eid)
        edges.append(Edge(eid=eid, u=u, v=v, length=length, index=i))

    # The canonical bit vector is ordered by edge *identifier*, so reorder
    # the edges (and their indices) lexicographically now that validation of
    # rows/locations is done.
    edges.sort(key=lambda e: e.eid)
    edges = [
        Edge(eid=e.eid, u=e.u, v=e.v, length=e.length, index=i)
        for i, e in enumerate(edges)
    ]

    start_s = "" if start is None else str(start).strip()
    if not start_s:
        raise AuditError("请选择检修口", ("start",), [{"field": "start"}])
    if start_s not in node_rows:
        raise AuditError(
            f"检修口 {start_s!r} 不存在", ("start",), [{"field": "start"}]
        )

    return clean_nodes, edges


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def build_nadj(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    forbidden: FrozenSet[int] = frozenset(),
):
    nadj: Dict[str, List[Tuple[str, int, int]]] = {n: [] for n in nodes}
    for e in edges:
        if e.index in forbidden:
            continue
        nadj[e.u].append((e.v, e.index, e.length))
        nadj[e.v].append((e.u, e.index, e.length))
    return nadj


def adjacency(nodes: Sequence[str], edges: Sequence[Edge]):
    adj: Dict[str, List[Tuple[str, int]]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.u].append((e.v, e.index))
        adj[e.v].append((e.u, e.index))
    return adj


def connected_components(nodes: Sequence[str], adj) -> List[List[str]]:
    comps: List[List[str]] = []
    unvisited = set(nodes)
    while unvisited:
        seed = next(iter(unvisited))
        stack = [seed]
        unvisited.discard(seed)
        comp: List[str] = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y, _ in adj[x]:
                if y in unvisited:
                    unvisited.discard(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


def dijkstra(nodes: Sequence[str], nadj, src: str):
    """Shortest distances from src."""
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        for w, _, length in nadj[u]:
            nd = du + length
            if w not in dist or nd < dist[w]:
                dist[w] = nd
                heapq.heappush(pq, (nd, w))
    return dist


def shortest_path_masks(
    nadj,
    src: str,
    dst: str,
    dist: dict,
) -> Tuple[int, ...]:
    """All shortest src->dst paths as integer edge-index masks.

    DFS over the shortest-path DAG; positive edge lengths make it acyclic
    (every predecessor has strictly smaller distance).
    """
    if src == dst:
        return (0,)
    pred: Dict[str, List[Tuple[str, int]]] = {}
    for x, dx in dist.items():
        if x == src:
            continue
        ps = []
        for y, ei, length in nadj[x]:
            if y in dist and dist[y] + length == dx:
                ps.append((y, ei))
        pred[x] = ps

    result: List[int] = []
    cur = 0
    seen_v = {dst}

    def dfs(x: str):
        nonlocal cur
        if x == src:
            result.append(cur)
            return
        for y, ei in pred.get(x, ()):
            if y in seen_v:
                continue
            seen_v.add(y)
            cur |= 1 << ei
            dfs(y)
            cur &= ~(1 << ei)
            seen_v.discard(y)

    dfs(dst)
    return tuple(result)


# ---------------------------------------------------------------------------
# Meet-in-the-middle over edge subsets (<= 32 edges, halves of <= 16)
# ---------------------------------------------------------------------------


def _half_enumeration(half_edges: Sequence[Edge], node_index: Dict[str, int]):
    """Every subset of one edge half via Gray-code flipping.

    Returns ``(parities, weights)`` arrays indexed by subset mask: the
    subgraph's odd-vertex bit vector (node indices) and total length.  Each
    Gray step adds/removes exactly one edge, so both values stay O(1).
    """
    h = len(half_edges)
    size = 1 << h
    parities = [0] * size
    weights = [0] * size
    edge_parity = [
        (1 << node_index[e.u]) | (1 << node_index[e.v]) for e in half_edges
    ]
    edge_len = [e.length for e in half_edges]
    cur_p = 0
    cur_w = 0
    prev_gray = 0
    for s in range(1, size):
        gray = s ^ (s >> 1)
        changed = gray ^ prev_gray
        j = changed.bit_length() - 1
        cur_p ^= edge_parity[j]
        if gray & changed:
            cur_w += edge_len[j]
        else:
            cur_w -= edge_len[j]
        parities[gray] = cur_p
        weights[gray] = cur_w
        prev_gray = gray
    return parities, weights


def _aggregate_min(parities, weights, included: int, allowed: int):
    """Minimum weight per parity and number of subsets attaining it.

    Only subsets of ``allowed`` that contain every bit of ``included`` are
    considered; sub-subsets of ``allowed`` are enumerated directly so the
    cost shrinks as more bits get pinned away.
    """
    table: Dict[int, List] = {}  # parity -> [min_weight, count]

    def visit(s: int):
        p = parities[s]
        w = weights[s]
        row = table.get(p)
        if row is None:
            table[p] = [w, 1]
        elif w < row[0]:
            row[0] = w
            row[1] = 1
        elif w == row[0]:
            row[1] += 1

    sub = allowed
    while True:
        if (sub & included) == included:
            visit(sub)
        if sub == 0:
            break
        sub = (sub - 1) & allowed
    return table


def _detailed_table(parities, weights) -> Dict[int, Dict[int, int]]:
    """parity -> {weight -> number of subsets} for every subset of a half."""
    table: Dict[int, Dict[int, int]] = {0: {0: 1}}
    for s in range(1, len(parities)):
        p = parities[s]
        w = weights[s]
        bucket = table.get(p)
        if bucket is None:
            table[p] = {w: 1}
        else:
            bucket[w] = bucket.get(w, 0) + 1
    return table


def _join_min(left: Dict[int, list], right: Dict[int, list], target: int):
    """Min weight and number of distinct half-unions with parity ``target``.

    A full subset is the unique disjoint union of its two halves, so every
    counted pair is one distinct edge set -- no matching/path decomposition
    can inflate the number.
    """
    if len(left) > len(right):
        left, right = right, left
    best = INF
    total = 0
    for p, row_l in left.items():
        row_r = right.get(target ^ p)
        if row_r is None:
            continue
        val = row_l[0] + row_r[0]
        ways = row_l[1] * row_r[1]
        if val < best:
            best = val
            total = ways
        elif val == best:
            total += ways
    return best, total


def _count_with_local_bit(parities, weights, other_detail, target, optimum, bit):
    total = 0
    other_bits = (len(parities) - 1) ^ bit
    sub = other_bits
    while True:
        s = sub | bit
        bucket = other_detail.get(target ^ parities[s])
        if bucket is not None:
            total += bucket.get(optimum - weights[s], 0)
        if sub == 0:
            break
        sub = (sub - 1) & other_bits
    return total


def solve_tjoins(nodes: Sequence[str], edges: Sequence[Edge], odd: Sequence[str]):
    """Exact optimum, distinct-set count, membership and canonical vector.

    Returns (optimum, total_count, in_any_bits, in_all_bits, canonical_mask).
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    m = len(edges)
    cut = m // 2
    left_edges = edges[:cut]
    right_edges = edges[cut:]
    lp, lw = _half_enumeration(left_edges, node_index)
    rp, rw = _half_enumeration(right_edges, node_index)

    l_all = (1 << len(left_edges)) - 1
    r_all = (1 << len(right_edges)) - 1
    left_min = _aggregate_min(lp, lw, 0, l_all)
    right_min = _aggregate_min(rp, rw, 0, r_all)
    target = 0
    for n in odd:
        target |= 1 << node_index[n]

    optimum, total_count = _join_min(left_min, right_min, target)

    # per-edge membership among optimum sets: an edge belongs to some/all
    # optimal joins according to how many optimum sets contain it.
    left_detail = _detailed_table(lp, lw)
    right_detail = _detailed_table(rp, rw)
    in_any = 0
    in_all = 0
    for j, e in enumerate(left_edges):
        c = _count_with_local_bit(
            lp, lw, right_detail, target, optimum, 1 << j
        )
        if c:
            in_any |= 1 << e.index
            if c == total_count:
                in_all |= 1 << e.index
    for j, e in enumerate(right_edges):
        c = _count_with_local_bit(
            rp, rw, left_detail, target, optimum, 1 << j
        )
        if c:
            in_any |= 1 << e.index
            if c == total_count:
                in_all |= 1 << e.index

    # 0-preferred canonical vector: walk edges in identifier order and pin
    # 0 whenever an optimum join consistent with the prefix pins still exists.
    inc_l = exc_l = inc_r = exc_r = 0
    canonical_mask = 0
    for e in edges:
        if e.index < cut:
            j = e.index
            bit = 1 << j
            trial_exc = exc_l | bit
            lm = _aggregate_min(lp, lw, inc_l, l_all & ~trial_exc)
            rm = _aggregate_min(rp, rw, inc_r, r_all & ~exc_r)
            feasible, _ = _join_min(lm, rm, target)
            if feasible == optimum:
                exc_l = trial_exc
                continue
            exc_l &= ~bit
            inc_l |= bit
        else:
            j = e.index - cut
            bit = 1 << j
            trial_exc = exc_r | bit
            lm = _aggregate_min(lp, lw, inc_l, l_all & ~exc_l)
            rm = _aggregate_min(rp, rw, inc_r, r_all & ~trial_exc)
            feasible, _ = _join_min(lm, rm, target)
            if feasible == optimum:
                exc_r = trial_exc
                continue
            exc_r &= ~bit
            inc_r |= bit
        canonical_mask |= 1 << e.index

    return optimum, total_count, in_any, in_all, canonical_mask


# ---------------------------------------------------------------------------
# Euler circuit (Hierholzer) on the expanded multigraph
# ---------------------------------------------------------------------------


def euler_circuit(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    start: str,
    multiplicity: Sequence[int],
) -> List[RouteStep]:
    copies: List[Tuple[int, str, str, int]] = []
    for e in edges:
        for _ in range(multiplicity[e.index]):
            copies.append((e.index, e.u, e.v, e.length))

    adj: Dict[str, List[int]] = {n: [] for n in nodes}
    for ci, (ei, u, v, _) in enumerate(copies):
        adj[u].append(ci)
        adj[v].append(ci)

    used = [False] * len(copies)
    stack: List[Tuple[str, int]] = [(start, -1)]
    circuit: List[Tuple[str, int]] = []
    while stack:
        x, _ = stack[-1]
        chosen: Optional[int] = None
        for ci in adj[x]:  # incident lists in ascending copy order
            if not used[ci]:
                chosen = ci
                break
        if chosen is None:
            circuit.append(stack.pop())
        else:
            used[chosen] = True
            _, u, v, _ = copies[chosen]
            stack.append((v if x == u else u, chosen))

    circuit.reverse()
    walk_copies = [ci for _, ci in circuit[1:]]

    steps: List[RouteStep] = []
    dup_counter: Dict[int, int] = {}
    cur = start
    for ci in walk_copies:
        ei, u, v, length = copies[ci]
        frm, to = (u, v) if cur == u else (v, u)
        dup_counter[ei] = dup_counter.get(ei, 0) + 1
        steps.append(
            RouteStep(
                edge_index=ei,
                edge_id=edges[ei].eid,
                frm=frm,
                to=to,
                length=length,
                duplicate_no=dup_counter[ei],
            )
        )
        cur = to
    return steps


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def audit(
    nodes: Sequence[str], raw_edges: Sequence[dict], start: Optional[str]
) -> AuditResult:
    nodes, edges = validate_input(nodes, raw_edges, start)
    adj = adjacency(nodes, edges)

    comps = connected_components(nodes, adj)
    if len(comps) > 1:
        raise AuditError(
            "管网不连通，存在多个连通分量："
            + "；".join("{" + ",".join(c) + "}" for c in comps),
            ("edges", "nodes"),
            [{"field": "edges"}],
        )

    degree = {n: 0 for n in nodes}
    for e in edges:
        degree[e.u] += 1
        degree[e.v] += 1
    odd = tuple(sorted(n for n in nodes if degree[n] % 2 == 1))
    total_length = sum(e.length for e in edges)
    m = len(edges)

    if not odd:
        empty: FrozenSet[int] = frozenset()
        multiplicity = tuple(1 for _ in edges)
        route = euler_circuit(nodes, edges, start, multiplicity)
        return AuditResult(
            nodes=nodes,
            edges=edges,
            start=start,
            odd_vertices=odd,
            components=tuple(tuple(c) for c in comps),
            total_length=total_length,
            added_length=0,
            optimal_count=1,
            canonical_set=empty,
            bit_vector="0" * m,
            classification={i: "never" for i in range(m)},
            multiplicity=multiplicity,
            route=tuple(route),
        )

    # exact distinct-set count, membership and canonical vector via
    # meet-in-the-middle over edge subsets (matching DP representations can
    # describe the same edge set multiple times and cannot be used directly)
    optimum, total_count, in_any, in_all, canonical_mask = solve_tjoins(
        nodes, edges, odd
    )
    bit_vector = "".join(
        "1" if canonical_mask >> i & 1 else "0" for i in range(m)
    )

    classification: Dict[int, str] = {}
    for i in range(m):
        if in_all >> i & 1:
            classification[i] = "required"
        elif in_any >> i & 1:
            classification[i] = "optional"
        else:
            classification[i] = "never"

    multiplicity = tuple(1 + (1 if canonical_mask >> i & 1 else 0) for i in range(m))
    route = euler_circuit(nodes, edges, start, multiplicity)

    return AuditResult(
        nodes=nodes,
        edges=edges,
        start=start,
        odd_vertices=odd,
        components=tuple(tuple(c) for c in comps),
        total_length=total_length,
        added_length=int(optimum),
        optimal_count=total_count,
        canonical_set=frozenset(
            i for i in range(m) if canonical_mask >> i & 1
        ),
        bit_vector=bit_vector,
        classification=classification,
        multiplicity=multiplicity,
        route=tuple(route),
    )
