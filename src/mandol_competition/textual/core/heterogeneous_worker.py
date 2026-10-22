"""Experimental typed publications using existing BuildJob/T1/T2 and graph rows.

Internal namespaces isolate raw lexical statistics. Navigation Reference records
contain identities only, are inactive/unindexed, and never constitute evidence.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

from mandol_substrate.state import BuildJob, CanonicalMemory, Edge, Mutation, Scope
from mandol_substrate.state.models import StateUnavailable, digest
from .derivation import DerivationWorker
from .gates import ScopeGates
from .heterogeneous_admission import validate_individually
from .heterogeneous_memory import (
    envelope,
    ground_episode_actors,
    messages,
    render,
    terminal_length_rejection,
    validate,
)
from .user_profile import checked_response

TYPES = ("entity", "episode", "relation")
ALL_TYPES = (*TYPES, "hierarchical", "profile", "procedure")


def typed_scope(raw_scope, kind, recipe):
    if kind not in (*ALL_TYPES, "contextual"):
        raise ValueError("Unknown derived partition")
    return Scope(
        ("view:" if kind == "contextual" else "hlm:")
        + digest([raw_scope.namespace, recipe.fingerprint, kind]),
        raw_scope.user_id,
    )


def reference_uid(scope, uid, version):
    return "ref_" + digest([scope.key, uid, version])


def support_versions(row, sources):
    ids = [e["source_uid"] for e in row.get("evidence", row.get("mentions", []))]
    ids.extend(row.get("context_source_uids", []))
    return {u: sources[u].version for u in sorted(set(ids))}


class _Receipts(DerivationWorker):
    def _receipt(self, messages, stage):
        # Same BuildJob protocol as the shared worker. A returned length failure
        # is a durable rejected outcome, not a successful derived memory and not
        # a reason to buy the identical response repeatedly.
        versions = {u: m.version for u, m in self._sources.items()}
        job = BuildJob.planned(
            self._scope, versions, digest([self.recipe.fingerprint, stage, messages])
        )
        self._resume_pending(self._scope, job)
        old = self._job(self._scope, job.job_id)
        if old and old.status == "completed":
            rec = old.progress["response"]
            if old.progress.get("input_hash") != digest(messages) or old.progress.get(
                "response_hash"
            ) != digest(rec):
                raise StateUnavailable("Durable provider receipt corrupt")
        else:
            if old is None:
                self._apply(self._scope, "intent", job)
            self.fault("before_gpt_" + stage)
            rec = self.provider(messages)
            if not terminal_length_rejection(rec):
                checked_response(rec)
            self.fault("after_gpt_" + stage + "_before_durable")
            completed = replace(
                job,
                status="completed",
                progress={
                    "input_hash": digest(messages),
                    "response_hash": digest(rec),
                    "response": rec,
                },
            )
            self._apply(self._scope, "completed", completed)
            self.fault("after_durable_" + stage)
        if not terminal_length_rejection(rec):
            checked_response(rec)
        return rec

    def __init__(self, coordinator, recipe, provider, fault, lock, scope, sources):
        super().__init__(coordinator, recipe, provider, fault=fault)
        self.provider, self.lock = provider, lock
        self._scope, self._sources = scope, sources

    def _apply(self, *args, **kwargs):
        with self.lock:
            return super()._apply(*args, **kwargs)

    def _resume_pending(self, scope, job):
        with self.lock:
            return super()._resume_pending(scope, job)


class HeterogeneousRuntime:
    def __init__(self, coordinator, recipe, provider, *, concurrency=4, fault=None):
        if concurrency not in (4, 8):
            raise ValueError("Bounded qualified provider concurrency required")
        self.coordinator, self.state = coordinator, coordinator.state
        self.recipe, self.provider = recipe, provider
        if self.state.identity is not None:
            from mandol_substrate.contracts import DerivedNamespaceBinding
            raw = Scope(self.state.identity.namespace, "namespace-authorization")
            for kind in self.kinds():
                self.state.authorize_derived_namespace(DerivedNamespaceBinding(
                    typed_scope(raw, kind, recipe).namespace, raw.namespace,
                    self.state.identity.hash, recipe.fingerprint, kind))
        self.fault = fault or (lambda _: None)
        self.gates = ScopeGates(timeout=600, max_inflight_scopes=16)
        self.calls = threading.BoundedSemaphore(concurrency)
        self.receipt_lock = threading.RLock()

    def _provider(self, messages):
        with self.calls:
            return self.provider(messages)

    def build(self, scope, source_uids):
        # One owner runtime per run. The overlay also uses this gate around Add
        # and Search. This never holds a SQLite transaction across model calls.
        with self.gates.lease(scope, write=True):
            return self._build(scope, source_uids)

    def _recover(self, scope):
        state = self.state.scope_state(scope)
        if state["canonical_seq"] == state["published_seq"]:
            return
        rows = self.state._rows(
            "SELECT request_id,payload_digest FROM requests WHERE scope=? AND mutation_seq=?",
            (scope.key, state["canonical_seq"]),
        )
        if len(rows) != 1:
            raise StateUnavailable("Unidentified pending publication")
        row = rows[0]
        self.coordinator.apply(
            scope, row["request_id"], row["payload_digest"], Mutation(), {}
        )

    def root_fingerprint(self, scope, sources):
        return self.recipe.fingerprint

    def provenance_bindings(self, scope, kind, row, sources):
        evidence = row.get("evidence", row.get("mentions", []))
        supporting = {e["source_uid"] for e in evidence}
        return [
            (
                scope,
                sources[u],
                u in supporting,
                [e["quote"] for e in evidence if e["source_uid"] == u],
            )
            for u in support_versions(row, sources)
        ]

    def publish_mutation(self, *args):
        return self.coordinator.apply(*args)

    def kinds(self):
        return TYPES

    def render_item(self, kind, row):
        return render(kind, row)

    def canonical_fields(self, kind, row):
        return {"memory_level": "Derived"}

    def item_metadata(self, scope, kind, row, uid, root):
        return {}

    def item_edges(self, scope, kind, row, uid, root):
        return []

    def _build(self, scope, source_uids):
        self._recover(scope)
        sources = self.state.get_memories(scope, source_uids)
        if set(sources) != set(source_uids) or len(source_uids) != len(
            set(source_uids)
        ):
            raise ValueError("Missing or duplicate source")
        envelope(scope, sources, self.recipe)
        root = BuildJob.planned(
            scope,
            {u: m.version for u, m in sources.items()},
            self.root_fingerprint(scope, sources),
        )
        worker = _Receipts(
            self.coordinator,
            self.recipe,
            self._provider,
            self.fault,
            self.receipt_lock,
            scope,
            sources,
        )
        old = worker._job(scope, root.job_id)
        if old and old.status == "completed":
            return old.progress["result"]
        if old is None:
            worker._apply(scope, "intent", root)
        if old and "prepared" in old.progress:
            prepared = old.progress["prepared"]
        else:
            prepared = self.prepare(scope, sources, worker, root)
            worker._apply(
                scope, "prepared", replace(root, progress={"prepared": prepared})
            )
        outputs = prepared["outputs"]
        if "source_versions" in prepared:
            sources = self.state.get_memories(scope, list(prepared["source_versions"]))
            if {u: m.version for u, m in sources.items()} != prepared[
                "source_versions"
            ]:
                raise StateUnavailable("Prepared source version changed")
        current = self.state.get_memories(scope, list(sources))
        if current != sources:
            raise StateUnavailable("Sources changed during preparation")
        type_outputs = {}
        entity_map = {}
        for kind in self.kinds():
            partition = typed_scope(scope, kind, self.recipe)
            self.state.ensure_scope(partition)
            self._recover(partition)
            memories, edges, references = [], [], {}

            def ref(
                target_scope,
                target_uid,
                target_version,
                text_hash,
                references=references,
            ):
                uid = reference_uid(target_scope, target_uid, target_version)
                references[uid] = CanonicalMemory(
                    uid,
                    "Canonical reference " + target_uid,
                    metadata={
                        "user_id": scope.user_id,
                        "hlm_reference": {
                            "namespace": target_scope.namespace,
                            "user_id": scope.user_id,
                            "scope_key": target_scope.key,
                            "uid": target_uid,
                            "version": target_version,
                            "text_hash": text_hash,
                        },
                    },
                    memory_level="Reference",
                    memory_type="canonical_reference",
                    active=False,
                    spaces=("hlm:reference",),
                    request_id=uid,
                )
                return uid

            def link(source, target, forward, reverse, properties, edges=edges):
                for a, b, relation in (
                    (source, target, forward),
                    (target, source, reverse),
                ):
                    edge = Edge(
                        "edge_" + digest([a, b, relation]),
                        a,
                        b,
                        relation,
                        properties=properties,
                        provenance={"root_job": root.job_id},
                    )
                    previous = next(
                        (e for e in edges if e.edge_id == edge.edge_id), None
                    )
                    if previous is not None and previous != edge:
                        raise StateUnavailable("Conflicting repeated edge identity")
                    if previous is None:
                        edges.append(edge)

            for row in outputs[kind]:
                uid = "hlm_" + digest([root.job_id, kind, row])
                versions = support_versions(row, sources)
                payload = {
                    "user_id": scope.user_id,
                    "hlm_kind": kind,
                    "hlm_item": row,
                    "hlm_job_id": root.job_id,
                    "raw_namespace": scope.namespace,
                    "source_versions": versions,
                    "recipe_hash": self.recipe.fingerprint,
                    **self.item_metadata(scope, kind, row, uid, root),
                }
                memories.append(
                    CanonicalMemory(
                        uid,
                        self.render_item(kind, row),
                        metadata=payload,
                        raw_data={"text_content": self.render_item(kind, row)},
                        request_id=root.job_id,
                        **self.canonical_fields(kind, row),
                        memory_type=kind,
                        spaces=("hlm:" + kind,),
                    )
                )
                edges.extend(self.item_edges(scope, kind, row, uid, root))
                if kind == "entity":
                    entity_map[row["id"]] = (uid, memories[-1])
                for source_scope, m, support, quotes in self.provenance_bindings(
                    scope, kind, row, sources
                ):
                    source_uid, version = m.logical_uid, m.version
                    target = ref(
                        source_scope, source_uid, version, digest(m.text_content)
                    )
                    props = {
                        "source_uid": source_uid,
                        "source_version": version,
                        "source_scope": source_scope.key,
                        "supporting": support,
                        "quotes": quotes,
                    }
                    link(
                        uid,
                        target,
                        "DERIVED_FROM" if support else "CONTEXT_FROM",
                        "SUPPORTS_" + kind.upper() if support else "CONTEXT_FOR",
                        props,
                    )
                if kind == "relation":
                    for field, relation in (
                        ("subject_id", "SUBJECT"),
                        ("object_id", "OBJECT"),
                    ):
                        target_uid, target_m = entity_map[row[field]]
                        target = ref(
                            typed_scope(scope, "entity", self.recipe),
                            target_uid,
                            1,
                            digest(target_m.text_content),
                        )
                        link(uid, target, relation, "RELATION_" + relation, {})
                if kind == "episode":
                    # Exact unambiguous mention names only; uncertain aliases
                    # cannot produce a graph link from substring similarity.
                    for name in {row["actor"], *row["participants"]}:
                        matches = [
                            (u, m)
                            for u, m in entity_map.values()
                            if name
                            in [
                                m.metadata["hlm_item"]["name"],
                                *m.metadata["hlm_item"]["aliases"],
                            ]
                        ]
                        if len(matches) == 1:
                            target_uid, target_m = matches[0]
                            target = ref(
                                typed_scope(scope, "entity", self.recipe),
                                target_uid,
                                1,
                                digest(target_m.text_content),
                            )
                            link(uid, target, "HAS_PARTICIPANT", "PARTICIPATES_IN", {})
            type_outputs[kind] = [m.logical_uid for m in memories]
            if memories:
                request_id = root.job_id + ":typed:" + kind
                previous = self.state.request(partition, request_id)
                if previous:
                    stored = self.state.get_memories(partition, type_outputs[kind])
                    if set(stored) != set(type_outputs[kind]) or any(
                        stored[m.logical_uid].metadata != m.metadata
                        or stored[m.logical_uid].text_content != m.text_content
                        for m in memories
                    ):
                        raise StateUnavailable(
                            "Published typed receipt differs from proposal"
                        )
                    continue
                old_refs = self.state.get_memories(partition, list(references))
                for uid, record in old_refs.items():
                    if record.metadata != references[uid].metadata or record.active:
                        raise StateUnavailable("Canonical reference identity changed")
                references = {u: m for u, m in references.items() if u not in old_refs}
                # All producers finish before one publication writer. Existing
                # graph machinery writes both directions in the same T1.
                mutation = Mutation(
                    memories=(*references.values(), *memories),
                    edges=tuple(edges),
                    spaces=("hlm:reference", "hlm:" + kind),
                    kind="heterogeneous_derived",
                )
                self.fault("before_" + kind + "_publication")
                self.publish_mutation(
                    partition,
                    request_id,
                    digest(asdict(mutation)),
                    mutation,
                    {},
                )
                self.fault("after_" + kind + "_publication")
        result = {
            "job_id": root.job_id,
            "output_uids": type_outputs,
            "errors": prepared["errors"],
            "receipt_hashes": prepared["receipt_hashes"],
            "status": "COMPLETE"
            if not prepared["errors"]
            else "COMPLETE_WITH_REJECTIONS",
            "actor_rejections": prepared.get("actor_rejections", []),
        }
        if getattr(self.recipe, "admission", None):
            result["item_rejections"] = prepared.get("item_rejections", {})
        complete = replace(
            root,
            status="completed",
            progress={"result": result, **prepared.get("root_progress", {})},
        )
        self.fault("before_complete_root")
        worker._apply(scope, "completed", complete)
        self.fault("after_complete_root")
        return result

    def prepare(self, scope, sources, worker, root):
        self.fault("before_entity_episode")
        receipts, outputs, errors = {}, {}, {}
        item_rejections = {}

        def stage(kind, entities=()):
            msg = messages(kind, scope, sources, self.recipe, entities=entities)
            rec = worker._receipt(msg, kind)
            receipts[kind] = rec
            if terminal_length_rejection(rec):
                outputs[kind] = []
                errors[kind] = "MODEL_LENGTH_REJECTED; original failed receipt retained"
                return
            try:
                if getattr(self.recipe, "admission", None) == "world_grounded_v9":
                    from .heterogeneous_grounding import validate_grounded

                    outputs[kind], item_rejections[kind] = validate_grounded(
                        kind, rec, scope, sources, self.recipe, entities=entities
                    )
                    if item_rejections[kind]:
                        errors[kind] = (
                            f"Rejected {len(item_rejections[kind])} grounded items"
                        )
                elif getattr(self.recipe, "admission", None) in {
                    "individual_strict_v1",
                    "source_review_v10",
                }:
                    outputs[kind], item_rejections[kind] = validate_individually(
                        kind, rec, scope, sources, self.recipe, entities=entities
                    )
                    if item_rejections[kind]:
                        errors[kind] = (
                            f"Rejected {len(item_rejections[kind])} invalid items"
                        )
                else:
                    outputs[kind] = validate(
                        kind, rec, scope, sources, self.recipe, entities=entities
                    )
            except ValueError as exc:
                # Fail closed for this typed proposal. Raw remains published.
                outputs[kind] = []
                errors[kind] = str(exc)[:240]

        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(stage, kind) for kind in ("entity", "episode")]
            for task in tasks:
                task.result()

        def review(kinds, label, entities=()):
            if getattr(self.recipe, "admission", None) != "source_review_v10":
                return
            from .heterogeneous_review import apply_review, review_messages

            proposed = {k: outputs[k] for k in kinds}
            if not any(proposed.values()):
                return
            msg = review_messages(
                scope, sources, self.recipe, proposed, entities=entities
            )
            rec = worker._receipt(msg, label)
            receipts[label] = rec
            try:
                approved, rejects = apply_review(rec, proposed, self.recipe)
                outputs.update(approved)
                item_rejections[label] = rejects
                if rejects:
                    errors[label] = f"Source review rejected {len(rejects)} items"
            except ValueError as exc:
                for k in kinds:
                    outputs[k] = []
                errors[label] = str(exc)[:240]

        review(("entity", "episode"), "grounding_entity_episode")
        outputs["episode"], actor_rejections = ground_episode_actors(
            outputs["episode"], outputs["entity"]
        )
        if outputs["entity"] and getattr(self.recipe, "build_relation", True):
            stage("relation", outputs["entity"])
        else:
            outputs["relation"] = []
        review(("relation",), "grounding_relation", outputs["entity"])
        self.fault("after_all_receipts")
        return {
            "outputs": outputs,
            "errors": errors,
            "receipt_hashes": {k: digest(v) for k, v in receipts.items()},
            "actor_rejections": actor_rejections,
            "item_rejections": item_rejections,
        }

    def eligible(self, raw_scope, memories):
        """Read-only source and complete-root guard, including source deletion."""
        result = {}
        root_validity = {}
        for uid, m in memories.items():
            meta = m.metadata
            if (
                meta.get("user_id") != raw_scope.user_id
                or meta.get("raw_namespace") != raw_scope.namespace
                or meta.get("recipe_hash") != self.recipe.fingerprint
                or not m.active
                or m.lifecycle == "deleted"
            ):
                continue
            roots = self.state._rows(
                "SELECT status,progress,input_versions FROM build_jobs WHERE scope=? AND job_id=?",
                (raw_scope.key, meta["hlm_job_id"]),
            )
            if len(roots) != 1 or roots[0]["status"] != "completed":
                continue
            job_id = meta["hlm_job_id"]
            if job_id not in root_validity:
                versions = json.loads(roots[0]["input_versions"])
                inputs = self.state.get_memories(raw_scope, list(versions))
                root_validity[job_id] = set(inputs) == set(versions) and all(
                    m.version == versions[u]
                    and m.active
                    and m.lifecycle != "deleted"
                    and m.metadata.get("user_id") == raw_scope.user_id
                    for u, m in inputs.items()
                )
            if not root_validity[job_id]:
                continue
            published = json.loads(roots[0]["progress"])["result"]["output_uids"]
            if uid not in published.get(meta["hlm_kind"], []):
                continue
            source = self.state.get_memories(raw_scope, list(meta["source_versions"]))
            if set(source) == set(meta["source_versions"]) and all(
                s.active
                and s.lifecycle != "deleted"
                and s.version == meta["source_versions"][u]
                and s.metadata.get("user_id") == raw_scope.user_id
                for u, s in source.items()
            ):
                result[uid] = m
        return result

    def verify_provenance(self, raw_scope, kind, memories):
        partition = typed_scope(raw_scope, kind, self.recipe)
        missing, wrong = [], []
        for uid, m in memories.items():
            for source, version in m.metadata["source_versions"].items():
                target = reference_uid(raw_scope, source, version)
                item = m.metadata["hlm_item"]
                supported = source in {
                    e["source_uid"]
                    for e in item.get("evidence", item.get("mentions", []))
                }
                expected_relations = {
                    (uid, target, "DERIVED_FROM" if supported else "CONTEXT_FROM"),
                    (
                        target,
                        uid,
                        "SUPPORTS_" + kind.upper() if supported else "CONTEXT_FOR",
                    ),
                }
                rows = self.state._rows(
                    "SELECT source_uid,target_uid,relation_type,properties FROM graph_edges WHERE scope=? AND active=1 AND ((source_uid=? AND target_uid=?) OR (source_uid=? AND target_uid=?))",
                    (partition.key, uid, target, target, uid),
                )
                if len(rows) != 2 or {
                    (r["source_uid"], r["target_uid"]) for r in rows
                } != {
                    (uid, target),
                    (target, uid),
                }:
                    missing.append([uid, source])
                if {
                    (r["source_uid"], r["target_uid"], r["relation_type"]) for r in rows
                } != expected_relations:
                    missing.append([uid, source, "wrong edge type"])
                if any(
                    json.loads(r["properties"]).get("source_scope") != raw_scope.key
                    or json.loads(r["properties"]).get("source_uid") != source
                    or json.loads(r["properties"]).get("source_version") != version
                    for r in rows
                ):
                    wrong.append([uid, source])
                reference = self.state.get_memories(partition, [target]).get(target)
                pointer = (
                    reference.metadata.get("hlm_reference", {}) if reference else {}
                )
                raw = self.state.get_memories(raw_scope, [source]).get(source)
                if (
                    reference is None
                    or reference.active
                    or pointer.get("namespace") != raw_scope.namespace
                    or pointer.get("user_id") != raw_scope.user_id
                    or pointer.get("scope_key") != raw_scope.key
                    or pointer.get("uid") != source
                    or pointer.get("version") != version
                    or raw is None
                    or raw.version != version
                    or pointer.get("text_hash") != digest(raw.text_content)
                ):
                    wrong.append([uid, source])
        return {
            "missing_reverse_links": missing,
            "wrong_scope_links": wrong,
            "status": "PASS" if not missing and not wrong else "FAIL",
        }
