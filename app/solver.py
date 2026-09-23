"""Chinese postman / route inspection core.

Undirected multigraph (parallel edges allowed, self loops forbidden) with
positive integer edge lengths.  Finds the minimum added length needed for all
vertices to have even degree, the exact number of optimal duplicate sets, the
canonical duplicate set (0-preferred bit vector in edge order), per-edge
classification, and a closed Euler tour for the canonical augmentation.

Exact counting of the distinct edge sets
-----------------------------------------
A duplicate set is a T-join: in the subgraph formed by the duplicated edges
exactly the originally odd vertices T have odd degree.  T-join theorem:

  * minimum T-join weight = minimum weight of a perfect matching of T under
    the shortest-path metric;
  * every minimum T-join decomposes into edge-disjoint shortest paths whose
    endpoint pairs form such a minimum matching.

Enumerate the distinct optimal joins by subset DP over the k odd vertices,
gated by the standard minimum-matching cost DP:

    F[S] = { p xor q | j is a partner of the lowest vertex i of S with
                       dist(i, j) + cost[S \\ {i, j}] == cost[S],
                       p a shortest i-j path (edge mask),
                       q in F[S \\ {i, j}] }

with F[0] = {0}.  Every entry is the *whole set* of distinct optimal masks
for that subproblem -- retaining only one representative is wrong, because
two optimal joins of a subproblem can extend differently: one may share an
edge with the new path (the xor cancels it) while the other does not, and
both results stay optimal for the full problem.  Symmetric difference is
exactly the edge-set operation of chaining a path onto a join, and integer
masks deduplicate identical sets reached through different decompositions
(parallel edges / equal-length paths).  With at most 18 odd vertices and 32
edges this stays small (sub-second on the worst unit-metric case).

The total number is |F[T]|.  The canonical member is the one whose bit
vector is lexicographically smallest in edge-identifier order with 0
preferred.  Per-edge classification comes from the intersection (required)
and union (optional / never) of all members of F[T].
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
# Perfect matching DP over a subset of vertices
# ---------------------------------------------------------------------------


def matching_dp(dist_matrix, labels: Sequence[int]):
    """Minimum matching cost and number of matchings attaining it.

    dist_matrix[i][j] is the shortest-path distance; INF means unreachable.
    Returns (cost array by mask, count array by mask).
    """
    k = len(labels)
    size = 1 << k
    cost = [INF] * size
    count = [0] * size
    cost[0] = 0
    count[0] = 1
    for mask in range(1, size):
        if mask.bit_count() & 1:
            continue
        i = (mask & -mask).bit_length() - 1
        rest0 = mask ^ (1 << i)
        best = INF
        total = 0
        bits = rest0
        while bits:
            jb = bits & -bits
            j = jb.bit_length() - 1
            bits ^= jb
            d = dist_matrix[i][j]
            sub = rest0 ^ jb
            if d >= INF or cost[sub] >= INF:
                continue
            val = d + cost[sub]
            if val < best:
                best = val
                total = count[sub]
            elif val == best:
                total += count[sub]
        cost[mask] = best
        count[mask] = total
    return cost, count


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
# Enumeration of the distinct optimal T-join edge sets
# ---------------------------------------------------------------------------


def enumerate_optimal_tjoins(
    odd: Tuple[str, ...],
    dist_from: Dict[str, dict],
    nadj,
    costdp,
    dist_matrix,
):
    """All distinct minimum T-join masks for the odd vertices.

    dp[mask] is the *set* of distinct edge masks of all optimal T-joins that
    pair exactly the odd vertices in ``mask``.  Anchor the lowest-index
    vertex i and form, for every partner j that can occur in an optimum (gated
    by the matching cost DP), the symmetric difference of every shortest i-j
    path with every optimal join of the remaining mask:

        dp[mask] = { p xor q | j optimal partner,
                                p in shortestPaths(i, j),
                                q in dp[mask \\ {i, j}] }

    Keeping the whole set -- not only one canonical representative -- is
    essential: two optimal joins of a subproblem can differ while still
    producing distinct optimal joins of the full problem (e.g. one sub-join
    extends with a shared edge and the other does not).  Integer edge masks
    deduplicate identical sets produced via different decompositions.

    Returns (count, canonical_mask, union_mask, intersection_mask) where the
    canonical mask is the 0-preferred lexicographically smallest bit vector
    in edge-identifier order.
    """
    k = len(odd)
    size = 1 << k
    pair_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def paths_between(a: int, b: int) -> Tuple[int, ...]:
        key = (a, b)
        if key not in pair_cache:
            pair_cache[key] = shortest_path_masks(
                nadj, odd[a], odd[b], dist_from[odd[a]]
            )
        return pair_cache[key]

    edge_count = 1 + max(
        ei for incident in nadj.values() for _, ei, _ in incident
    )
    vector_weight = tuple(1 << (edge_count - 1 - i) for i in range(edge_count))

    def vector_rank(edge_mask: int) -> int:
        rank = 0
        bits = edge_mask
        while bits:
            bit = bits & -bits
            rank += vector_weight[bit.bit_length() - 1]
            bits ^= bit
        return rank

    dp: List[Optional[FrozenSet[int]]] = [None] * size
    dp[0] = frozenset({0})
    for mask in range(1, size):
        if mask.bit_count() & 1:
            continue
        ib = mask & -mask
        i = ib.bit_length() - 1
        rest0 = mask ^ ib
        target_cost = costdp[mask]
        if target_cost >= INF:
            continue
        represented: set[int] = set()
        bits = rest0
        while bits:
            jb = bits & -bits
            j = jb.bit_length() - 1
            bits ^= jb
            sub = rest0 ^ jb
            if dist_matrix[i][j] >= INF or costdp[sub] >= INF:
                continue
            if dist_matrix[i][j] + costdp[sub] != target_cost:
                continue
            sub_sets = dp[sub]
            if not sub_sets:
                continue
            for pmask in paths_between(i, j):
                for sub_mask in sub_sets:
                    represented.add(pmask ^ sub_mask)
        if represented:
            dp[mask] = frozenset(represented)

    full_sets = dp[size - 1]
    if not full_sets:
        return 0, 0, 0, 0

    canonical = min(full_sets, key=vector_rank)
    in_any = 0
    in_all: Optional[int] = None
    for edge_mask in full_sets:
        in_any |= edge_mask
        in_all = edge_mask if in_all is None else in_all & edge_mask
    return len(full_sets), canonical, in_any, in_all


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

    # shortest distances from every vertex
    nadj = build_nadj(nodes, edges)
    dist_from: Dict[str, dict] = {s: dijkstra(nodes, nadj, s) for s in nodes}

    k = len(odd)
    D = [[INF] * k for _ in range(k)]
    for i, s in enumerate(odd):
        for j, t in enumerate(odd):
            if t in dist_from[s]:
                D[i][j] = dist_from[s][t]

    costdp, _ = matching_dp(D, list(range(k)))
    full = (1 << k) - 1
    optimum = costdp[full]

    # distinct optimal duplicate sets, exact
    total_count, canonical_mask, in_any, in_all = enumerate_optimal_tjoins(
        odd, dist_from, nadj, costdp, D
    )
    bit_vector = "".join("1" if canonical_mask >> i & 1 else "0" for i in range(m))

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
