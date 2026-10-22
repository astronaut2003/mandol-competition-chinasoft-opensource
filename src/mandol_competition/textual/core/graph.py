"""SQLite-owned structural graph, bounded disposable rustworkx views."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import rustworkx as rx

from mandol_substrate.state.models import StateUnavailable, canonical_json


@dataclass(frozen=True)
class StructuralView:
    scope: object
    seq: int
    graph: object
    uid_to_index: dict
    estimated_bytes: int

    def neighbors(self, uid, *, direction="all", relations=None, limit=100):
        if (
            direction not in {"all", "successors", "predecessors"}
            or not 0 < limit <= 4096
        ):
            raise ValueError("invalid traversal parameters")
        index = self.uid_to_index.get(uid)
        if index is None:
            return []
        entries = []
        if direction in {"all", "successors"}:
            entries.extend(
                (self.graph[t], data[0], data[1], "outgoing")
                for _, t, data in self.graph.out_edges(index)
            )
        if direction in {"all", "predecessors"}:
            entries.extend(
                (self.graph[s], data[0], data[1], "incoming")
                for s, _, data in self.graph.in_edges(index)
            )
        return sorted(
            (
                e
                for e in entries
                if e[0] != uid and (not relations or e[2] in relations)
            ),
            key=lambda e: (e[3] != "outgoing", e[0], e[1]),
        )[:limit]

    def traverse(
        self, seeds, *, hops=1, relations=None, limit=100, per_node=10, allowed=None
    ):
        if not 0 <= hops <= 8 or not 0 < limit <= 4096 or len(seeds) > 1024:
            raise ValueError("traversal exceeds bounded budget")
        visited = set(seeds)
        frontier = sorted(visited)
        result = []
        for _ in range(hops):
            next_layer = []
            for source in frontier:
                # Parallel edges are durable; expansion deduplicates neighbors.
                entries = self.neighbors(source, relations=relations, limit=4096)
                neighbors = list(
                    dict.fromkeys(e[0] for e in entries if not e[0].startswith("ms:"))
                )
                if allowed is not None:
                    neighbors = allowed(neighbors)
                for uid in neighbors[:per_node]:
                    if uid not in visited:
                        visited.add(uid)
                        next_layer.append(uid)
                        result.append(uid)
                        if len(result) == limit:
                            return result
            frontier = sorted(next_layer)
            if not frontier:
                break
        return result

    def shortest_path(
        self, source, target, *, relations=None, max_hops=8, max_visited=4096
    ):
        if source not in self.uid_to_index or target not in self.uid_to_index:
            return []
        if not 0 < max_hops <= 32 or not 0 < max_visited <= 16384:
            raise ValueError("invalid path budget")
        queue = deque([(source, [source])])
        visited = {source}
        while queue:
            uid, path = queue.popleft()
            if uid == target:
                return path
            if len(path) - 1 == max_hops:
                continue
            for neighbor, *_ in self.neighbors(
                uid, direction="successors", relations=relations, limit=4096
            ):
                if neighbor not in visited:
                    if len(visited) >= max_visited:
                        raise StateUnavailable("path exploration budget exhausted")
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []


class GraphViewManager:
    format_version = 1

    def __init__(
        self,
        state,
        *,
        max_users=4,
        max_nodes=200000,
        max_edges=400000,
        max_bytes=256 * 1024 * 1024,
    ):
        if min(max_users, max_nodes, max_edges, max_bytes) <= 0:
            raise ValueError("invalid graph cache budget")
        self.state = state
        self.max_users, self.max_nodes, self.max_edges, self.max_bytes = (
            max_users,
            max_nodes,
            max_edges,
            max_bytes,
        )
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        # At most two transient builds in addition to the bounded LRU. A caller
        # holds a scope lease for a view's lifetime; API admission is bounded too.
        self._builds = threading.BoundedSemaphore(2)

    def _from_rows(self, scope, seq, nodes, edges):
        graph = rx.PyDiGraph(multigraph=True)
        mapping = {}
        estimated = 0
        for uid in nodes:
            if uid in mapping:
                raise StateUnavailable("duplicate graph node")
            estimated += 256 + len(uid.encode())
            if len(mapping) >= self.max_nodes or estimated > self.max_bytes:
                raise StateUnavailable("graph node/cache capacity exceeded")
            mapping[uid] = graph.add_node(uid)
        seen_edges = set()
        for source, target, edge_id, kind in edges:
            if edge_id in seen_edges or source not in mapping or target not in mapping:
                raise StateUnavailable("invalid structural graph edge")
            seen_edges.add(edge_id)
            estimated += 192 + len(edge_id.encode()) + len(kind.encode())
            if graph.num_edges() >= self.max_edges or estimated > self.max_bytes:
                raise StateUnavailable("graph edge/cache capacity exceeded")
            graph.add_edge(mapping[source], mapping[target], (edge_id, kind))
        return StructuralView(scope, seq, graph, mapping, estimated)

    def _install(self, view):
        with self._lock:
            self._cache.pop(view.scope.key, None)
            while self._cache and (
                len(self._cache) >= self.max_users
                or sum(v.estimated_bytes for v in self._cache.values())
                + view.estimated_bytes
                > self.max_bytes
            ):
                self._cache.popitem(last=False)
            self._cache[view.scope.key] = view
        return view

    def prepare(self, scope, seq):
        with self._lock:
            cached = self._cache.get(scope.key)
            if cached is not None and cached.seq == seq:
                self._cache.move_to_end(scope.key)
                return cached
        if not self._builds.acquire(timeout=30):
            raise StateUnavailable("graph build backpressure")
        try:
            if self.state.scope_state(scope)["canonical_seq"] != seq:
                raise StateUnavailable("graph sequence is not canonical")
            nodes = (
                r["node_uid"]
                for batch in self.state.graph_rows(scope, "nodes")
                for r in batch
            )
            edges = (
                (r["source_uid"], r["target_uid"], r["edge_id"], r["relation_type"])
                for batch in self.state.graph_rows(scope, "edges")
                for r in batch
            )
            view = self._from_rows(scope, seq, nodes, edges)
            if self.state.scope_state(scope)["canonical_seq"] != seq:
                raise StateUnavailable("graph changed without scope write lease")
            return self._install(view)
        finally:
            self._builds.release()

    get = prepare

    def evict(self, scope=None):
        with self._lock:
            if scope is None:
                self._cache.clear()
            else:
                self._cache.pop(scope.key, None)

    def stats(self):
        with self._lock:
            return {
                "loaded_users": len(self._cache),
                "nodes": sum(v.graph.num_nodes() for v in self._cache.values()),
                "edges": sum(v.graph.num_edges() for v in self._cache.values()),
                "estimated_bytes": sum(v.estimated_bytes for v in self._cache.values()),
            }

    def health(self):
        stats = self.stats()
        return (
            stats["loaded_users"] <= self.max_users
            and stats["estimated_bytes"] <= self.max_bytes
        )

    def _manifest(self, scope, seq):
        return {
            "db_uuid": self.state.db_uuid,
            "scope": scope.key,
            "published_seq": seq,
            "format": self.format_version,
            "profile_hash": self.state.profile_hash,
        }

    def save_graph(self, scope, path, *, fault=None):
        """Optional cache export under caller read lease; failure returns False."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            state = self.state.scope_state(scope)
            if state["canonical_seq"] != state["published_seq"]:
                raise StateUnavailable("cannot export unpublished graph")
            view = self.get(scope, state["published_seq"])
            data = {
                "nodes": list(view.graph.nodes()),
                "edges": [
                    [view.graph[s], view.graph[t], *e]
                    for s, t, e in view.graph.weighted_edge_list()
                ],
            }
            payload = canonical_json(data)
            envelope = {
                **self._manifest(scope, view.seq),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "payload": data,
            }
            encoded = canonical_json(envelope).encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as handle:
                half = len(encoded) // 2
                handle.write(encoded[:half])
                handle.flush()
                if fault:
                    fault("graph_cache_temp_write")
                handle.write(encoded[half:])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return True
        except (OSError, ValueError, StateUnavailable):
            return False

    def load_graph(self, scope, path):
        """Any absent, stale, or invalid cache is rebuilt from canonical SQLite."""
        seq = self.state.scope_state(scope)["published_seq"]
        if self.state.scope_state(scope)["canonical_seq"] != seq:
            raise StateUnavailable("scope not published")
        try:
            path = Path(path)
            if path.stat().st_size > self.max_bytes * 2:
                raise ValueError("oversized graph cache")
            data = json.loads(path.read_bytes())
            if not isinstance(data, dict):
                raise TypeError("graph cache envelope must be an object")
            if any(data.get(k) != v for k, v in self._manifest(scope, seq).items()):
                raise ValueError("graph cache manifest mismatch")
            payload = data["payload"]
            if not isinstance(payload, dict):
                raise TypeError("graph cache payload must be an object")
            if (
                data["sha256"]
                != hashlib.sha256(canonical_json(payload).encode()).hexdigest()
            ):
                raise ValueError("graph cache checksum mismatch")
            view = self._from_rows(scope, seq, payload["nodes"], payload["edges"])
            return self._install(view)
        except (OSError, ValueError, KeyError, TypeError, StateUnavailable):
            self.evict(scope)
            return self.prepare(scope, seq)


def semantic_neighbors(
    projection, scope, seq, source_uid, *, limit=5, threshold=0.7, spaces=None
):
    if not 0 < limit <= 1024:
        raise ValueError("invalid semantic neighbor budget")
    vector = projection.get_candidate_embeddings(scope, [source_uid])[source_uid]
    hits = projection.search_dense(
        scope, vector, limit=limit + 1, published_seq=seq, spaces=spaces
    )
    return [h for h in hits if h.logical_uid != source_uid and h.score >= threshold][
        :limit
    ]
