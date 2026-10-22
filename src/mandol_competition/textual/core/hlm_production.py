"""Four-tower canonical derivation, explicitly enabled and independent of raw Search.

Extends the existing typed publisher; no alternate transaction, vector store or
job ledger. SemanticFrame is optional, not a mandatory extraction stage.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from pydantic import Field

from mandol_substrate.state import Edge, StateUnavailable
from mandol_substrate.state.models import digest
from .heterogeneous_admission import validate_individually
from .heterogeneous_memory import (
    COMMON,
    PROMPTS,
    SCHEMAS,
    HeterogeneousProvider,
    HeterogeneousRecipe,
    Strict,
    Support,
    messages,
    render,
    terminal_length_rejection,
)
from .heterogeneous_worker import HeterogeneousRuntime, typed_scope
from .profile_discourse import (
    DiscourseProvider,
    DiscourseRecipe,
    chunks,
    compatible_existing,
    empty_frame,
    merge_assertions,
    messages_a,
    messages_c,
    validate_a,
    validate_c,
)
from .user_profile import checked_response, strict_json


@dataclass(frozen=True)
class HighLevelMemoryPolicy:
    hierarchical_enabled: bool = False
    episodic_enabled: bool = False
    entity_relation_enabled: bool = False
    profile_enabled: bool = False
    navigation_enabled: bool = False
    semantic_evidence_enabled: bool = False
    # Optional helper remains available separately; never implicitly purchased.
    semantic_frame_enabled: bool = False

    @property
    def types(self):
        return tuple(
            k
            for k, enabled in (
                ("entity", self.entity_relation_enabled),
                ("episode", self.episodic_enabled),
                ("relation", self.entity_relation_enabled),
                ("hierarchical", self.hierarchical_enabled),
                ("profile", self.profile_enabled),
            )
            if enabled
        )

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise ValueError("Explicit boolean HLM flags required")
        if self.semantic_frame_enabled:
            raise ValueError(
                "SemanticFrame is a separately invoked helper, not a qualified common frontend"
            )


class Summary(Strict):
    text: str = Field(min_length=1, max_length=2200)
    evidence: list[Support] = Field(min_length=1, max_length=12)
    context_source_uids: list[str] = Field(max_length=12)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class Summaries(Strict):
    summaries: list[Summary] = Field(max_length=4)


SUMMARY_PROMPT = (
    COMMON
    + """Produce bounded session/topic compression, not insight.
Summarize what the speakers explicitly discussed or experienced, preserving who
said it, negation, conditions, uncertainty and time qualifiers. Organize related
claims compactly. Do not predict, advise, infer deep connections, assert a profile,
or resolve entity identity/state conflicts. A quoted/drafted/hypothetical topic
must remain explicitly described as that topic, never an actual life event.
Each independent summary has its own exact supporting quotes and context-only
UIDs. Include all essential qualifiers in text. No summary is also valid.
"""
)
PRODUCTION_PROMPTS = {
    **PROMPTS,
    "entity": PROMPTS["entity"]
    + "\nA PERSON denotes one individual, never multiple authors/speakers as one identity. Keep uncertain identities separate; no inferred alias or name-only cross-session merge.",
    "episode": PROMPTS["episode"]
    + "\nThis tower materializes only EVENT_OCCURRENCE with explicit STARTED or COMPLETED status. Plans/cancellations/questions/advice/static habits/preferences are outside this tower. Actor text may be grounded without a named entity node. Do not manufacture E0 ownership from another speaker's first person.",
    "relation": PROMPTS["relation"]
    + "\nPreserve every conditional/scope qualifier in condition AND text. CURRENT does not erase a condition. Use only the supplied entity IDs; reject ambiguous endpoints. No substring matching.",
    "hierarchical": SUMMARY_PROMPT,
}


@dataclass(frozen=True)
class ProductionHLMRecipe(HeterogeneousRecipe):
    version: str = "source_b1_four_tower_production_v2"
    max_sources: int = 64
    max_input_chars: int = 96000
    policy: HighLevelMemoryPolicy = HighLevelMemoryPolicy()

    @property
    def prompt_map(self):
        return PRODUCTION_PROMPTS

    @property
    def admissible_episode_states(self):
        return (("EVENT_OCCURRENCE", "STARTED"), ("EVENT_OCCURRENCE", "COMPLETED"))

    @property
    def fingerprint(self):
        return digest(
            {
                "recipe": asdict(self),
                "prompts": self.prompt_map,
                "schemas": {
                    k: v.model_json_schema()
                    for k, v in {
                        **getattr(self, "schema_map", SCHEMAS),
                        "hierarchical": Summaries,
                    }.items()
                },
                "profile": self.profile_recipe.fingerprint,
                "protocol": "four-tower-prepared-root-v1",
            }
        )

    @property
    def profile_recipe(self):
        # Reuse v12 semantic decisions; only whole-session capacity changes.
        return DiscourseRecipe(
            self.storage_profile_hash,
            model=self.model, actual_model=self.actual_model, provider=self.provider,
            version="source_b1_profile_discourse_industrial_v12",
            max_sources=72,
            max_input_chars=104000,
            whole_add_turns=64,
        )


class ProductionHLMProvider(HeterogeneousProvider):
    def response_format(self, messages, json_mode):
        stage = json.loads(messages[-1]["content"])["stage"]
        if stage in ("A", "C"):
            result = DiscourseProvider.response_format(self, messages, json_mode)
            if stage == "A":
                # Equivalent schema factoring: repeated per-turn enum values
                # otherwise exceed the provider's 1,000-enum wire limit.
                schema = result["json_schema"]["schema"]
                assessments = schema["properties"]["assessments"]["properties"]
                if assessments:
                    shared = next(iter(assessments.values()))
                    if not all(value == shared for value in assessments.values()):
                        raise ValueError("Assessment schemas unexpectedly differ")
                    schema["$defs"] = {"turn_assessment": shared}
                    for uid in assessments:
                        assessments[uid] = {"$ref": "#/$defs/turn_assessment"}
            return result
        if stage == "hierarchical":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "source_backed_summary",
                    "strict": True,
                    "schema": Summaries.model_json_schema(),
                },
            }
        return super().response_format(messages, json_mode)


class CanonicalHighLevelMemory(HeterogeneousRuntime):
    """Implementation behind HighLevelMemoryBuilder.from_competition().

    One owning instance per run. Same-user jobs serialize; different users and
    typed model producers run concurrently. A frozen proposal precedes all typed
    publications so recovery never reselects Profile targets after partial T1/T2.
    """

    def __init__(self, *args, publication_writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        from .hlm_builder import CanonicalPublicationWriter

        self._owns_writer = publication_writer is None
        self._writer = publication_writer or CanonicalPublicationWriter(self.coordinator)

    def close(self):
        if self._owns_writer:
            self._writer.close()

    def publish_mutation(self, *args):
        # One owned writer for every producer/user in this run's facade. Each
        # queue item is a complete canonical mutation owned by existing T1/T2.
        return self._writer.submit_publication(*args).result(timeout=600)

    def kinds(self):
        return self.recipe.policy.types

    def build(self, scope, source_uids):
        if not self.kinds():
            return {"status": "DISABLED", "output_uids": {}, "gpt_calls": 0}
        with self.gates.lease(scope, write=True):
            # Resume an earlier same-user root before interpreting a later Add.
            roots = self.state._rows(
                "SELECT job_id,input_versions FROM build_jobs WHERE scope=? AND recipe_fingerprint=? AND status!='completed' ORDER BY commit_seq,job_id",
                (scope.key, self.recipe.fingerprint),
            )
            for pending in roots:
                ids = list(json.loads(pending["input_versions"]))
                self._ordered_build(scope, ids)
            return self._ordered_build(scope, source_uids)

    def _ordered_build(self, scope, source_uids):
        sources = self.state.get_memories(scope, source_uids)
        if len(set(source_uids)) != len(source_uids) or set(sources) != set(
            source_uids
        ):
            raise StateUnavailable("Exact, unique canonical source UIDs required")
        if any("," in u for u in source_uids):
            # Canonical IDs are opaque, but a list disguised as one ID is never a mention.
            raise ValueError("Comma-joined source identities are not allowed")
        if len({m.session_id for m in sources.values()}) != 1:
            raise ValueError(
                "Build one ordered canonical session/Add; do not blend session owners"
            )
        ordered = sorted(
            source_uids,
            key=lambda u: (sources[u].commit_seq, sources[u].turn_index or 0, u),
        )
        # State-store hydration order is not an input ordering contract. prepare
        # explicitly reorders; no 512-character L0 transformation is performed.
        return super()._build(scope, ordered)

    def _profiles(self, scope):
        partition = typed_scope(scope, "profile", self.recipe)
        rows = self.state._rows(
            "SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND memory_type='profile'",
            (partition.key,),
        )
        return self.eligible(
            scope, self.state.get_memories(partition, [r["logical_uid"] for r in rows])
        )

    def _frame(self, scope, session):
        rows = self.state._rows(
            "SELECT progress FROM build_jobs WHERE scope=? AND recipe_fingerprint=? AND status='completed' AND json_extract(progress,'$.discourse_session_id') IS ? ORDER BY commit_seq DESC LIMIT 1",
            (scope.key, self.recipe.fingerprint, session),
        )
        return (
            json.loads(rows[0]["progress"]).get("frame", empty_frame())
            if rows
            else empty_frame()
        )

    def prepare(self, scope, sources, worker, root):
        sources = dict(
            sorted(
                sources.items(),
                key=lambda x: (x[1].commit_seq, x[1].turn_index or 0, x[0]),
            )
        )
        outputs = {k: [] for k in self.kinds()}
        errors, receipts, root_progress = (
            {},
            {},
            {"production_recipe": asdict(self.recipe)},
        )
        profile_sources = {}

        def admitted_content(record):
            # A durably recorded truncation cannot become valid by retrying its
            # receipt. Reject this tower proposal, preserving the raw publication
            # and other independent outputs. Transport failures still propagate.
            if terminal_length_rejection(record):
                raise ValueError("TERMINAL_LENGTH_REJECTION")
            return checked_response(record)

        def typed(kind, entities=()):
            msg = messages(kind, scope, sources, self.recipe, entities=entities)
            payload = json.loads(msg[-1]["content"])
            positions = {u: i for i, u in enumerate(sources)}
            payload["sources"].sort(key=lambda x: positions[x["source_uid"]])
            msg[-1]["content"] = json.dumps(payload, ensure_ascii=False)
            rec = worker._receipt(msg, kind)
            receipts[kind] = digest(rec)
            try:
                content = admitted_content(rec)
                if kind == "hierarchical":
                    value = strict_json(content)
                    if (
                        not isinstance(value, dict)
                        or set(value) != {"summaries"}
                        or not isinstance(value["summaries"], list)
                        or len(value["summaries"]) > 4
                    ):
                        raise ValueError("Invalid bounded summary root")
                    rows = []
                    rejected = []
                    for index, proposed in enumerate(value["summaries"]):
                        try:
                            row = Summary.model_validate(proposed).model_dump()
                            self.validate_evidence(scope, row, sources)
                            row["level"] = "L1_Summary"
                            row["semantic_authority"] = "SOURCE_COMPRESSION_ONLY"
                            rows.append(row)
                        except ValueError as exc:
                            rejected.append(
                                {"position": index, "reason": str(exc)[:240]}
                            )
                    if rejected:
                        errors[kind] = {"individual_rejections": rejected}
                    outputs[kind] = rows
                else:
                    rows, rejected = validate_individually(
                        kind, rec, scope, sources, self.recipe, entities=entities
                    )
                    if rejected:
                        errors[kind] = {"individual_rejections": rejected}
                    if kind == "episode":
                        rows = [
                            r
                            for r in rows
                            if (r["assertion_kind"], r["status"]) in self.recipe.admissible_episode_states
                        ]
                        limit = getattr(self.recipe, "episode_limit", 24)
                        if len(rows) > limit:
                            errors.setdefault(kind, {})["bounded_promotion_excluded"] = len(rows) - limit
                            rows = rows[:limit]
                    for row in rows:
                        evidence = row.get("mentions", row.get("evidence", []))
                        for e in evidence:
                            m = sources[e["source_uid"]]
                            # Mention-level identity, occurrence span and version.
                            if kind == "entity":
                                e.update(
                                    source_version=m.version,
                                    session_id=m.session_id,
                                    turn_index=m.turn_index,
                                    mention_time=m.business_timestamp_ms,
                                    span_start=m.text_content.index(e["quote"]),
                                )
                        if kind == "episode":
                            row["mention_time"] = sources[
                                evidence[0]["source_uid"]
                            ].business_timestamp_ms
                    outputs[kind] = rows
            except ValueError as exc:
                errors[kind] = str(exc)[:240]

        def profile():
            recipe = self.recipe.profile_recipe
            session = next(iter(sources.values())).session_id
            frame = self._frame(scope, session)
            ids = list(dict.fromkeys([*frame["source_uids"], *sources]))
            all_sources = self.state.get_memories(scope, ids)
            if set(all_sources) != set(ids):
                raise StateUnavailable("Persistent discourse source missing")
            # Old frame sources are context; input is original ordered canonical text.
            all_sources = dict(
                sorted(
                    all_sources.items(),
                    key=lambda x: (x[1].commit_seq, x[1].turn_index or 0, x[0]),
                )
            )
            groups = chunks(all_sources, list(sources), recipe, frame)

            def stage_a(index_group):
                index, (src, core) = index_group
                msg = messages_a(scope, src, core, recipe, frame)
                rec = worker._receipt(msg, "profile_A" + str(index))
                receipts["profile_A" + str(index)] = digest(rec)
                return validate_a(
                    admitted_content(rec), scope, src, core, recipe, frame
                )

            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    parts = list(pool.map(stage_a, enumerate(groups)))
                merged = merge_assertions(parts)
                current = self._profiles(scope)
                used = set()
                for index, candidate in enumerate(merged["candidates"]):
                    old = compatible_existing(
                        scope, candidate, current, limit=recipe.max_existing
                    )
                    msg = messages_c(candidate, old)
                    rec = worker._receipt(msg, "profile_C" + str(index))
                    receipts["profile_C" + str(index)] = digest(rec)
                    op = validate_c(admitted_content(rec), scope, candidate, old)
                    if op is None:
                        continue
                    target = old.get(op["target_uid"])
                    if target and target.logical_uid in used:
                        raise ValueError(
                            "Multiple mutations of one Profile target in a single Add"
                        )
                    if target:
                        used.add(target.logical_uid)
                    item = (
                        target.metadata["profile_delta_item"]
                        if op["op"] in ("CLOSE", "REINFORCE")
                        else op["item"]
                    )
                    row = {
                        "profile_item": item,
                        "operation": op["op"],
                        "target_uid": op["target_uid"],
                        "target_version": target.version if target else None,
                        "lineage_id": target.metadata["profile_id"] if target else None,
                        "profile_version": target.metadata["profile_version"] + 1
                        if target
                        else 1,
                        "evidence": op["evidence"],
                        "context_source_uids": op["context_uids"],
                    }
                    self.validate_evidence(scope, row, all_sources)
                    outputs["profile"].append(row)
                # Publisher needs every support/context source, even across Adds.
                profile_sources.update(all_sources)
                root_progress.update(
                    frame=merged["frame"],
                    discourse_session_id=session,
                    chunk_disagreements=merged["disagreement_uids"],
                )
            except ValueError as exc:
                outputs["profile"] = []
                errors["profile"] = str(exc)[:240]
                root_progress.update(frame=frame, discourse_session_id=session)

        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [
                pool.submit(typed, k)
                for k in self.kinds()
                if k not in ("relation", "profile")
            ]
            if "profile" in outputs:
                tasks.append(pool.submit(profile))
            for task in tasks:
                task.result()
        if "relation" in outputs and outputs["entity"]:
            # Internal enrichment fields are not needed in the model schema.
            typed("relation", outputs["entity"])
        sources.update(profile_sources)
        return {
            "outputs": outputs,
            "errors": errors,
            "receipt_hashes": receipts,
            "root_progress": root_progress,
            "source_versions": {u: m.version for u, m in sources.items()},
        }

    @staticmethod
    def validate_evidence(scope, row, sources):
        ev = row["evidence"]
        support = {e["source_uid"] for e in ev}
        context = row.get("context_source_uids", [])
        if not ev or len(context) != len(set(context)) or support & set(context):
            raise ValueError("Distinct nonempty support and context required")
        for uid in support | set(context):
            m = sources.get(uid)
            if (
                m is None
                or not m.active
                or m.lifecycle == "deleted"
                or m.metadata.get("user_id") != scope.user_id
            ):
                raise ValueError("Invalid exact-user live source")
        if any(
            not e["quote"] or e["quote"] not in sources[e["source_uid"]].text_content
            for e in ev
        ):
            raise ValueError("Fabricated supporting quote")

    def render_item(self, kind, row):
        if kind == "hierarchical":
            return "[Session compression] " + row["text"]
        if kind == "profile":
            p = row["profile_item"]
            return f"[User Profile · {p['lifetime']}] {p['statement']}"
        if kind == "episode" and hasattr(self.recipe, "render_episode"):
            return self.recipe.render_episode(row)
        return render(kind, row)

    def item_metadata(self, scope, kind, row, uid, root):
        if kind != "profile":
            return {}
        return {
            "profile_delta_item": row["profile_item"],
            "profile_id": row["lineage_id"] or uid,
            "profile_version": row["profile_version"],
            "operation": row["operation"],
        }

    def canonical_fields(self, kind, row):
        from mandol_substrate.spaces import TowerSpace

        tower = {
            "hierarchical": TowerSpace.HIERARCHICAL_ROOT,
            "episode": TowerSpace.EPISODIC_ROOT,
            "entity": TowerSpace.GRAPH_ROOT,
            "relation": TowerSpace.GRAPH_ROOT,
            "profile": TowerSpace.USER_PROFILE_ROOT,
        }[kind]
        return {
            "memory_tower": tower.value,
            "memory_level": row.get("level", "Derived"),
        }

    def item_edges(self, scope, kind, row, uid, root):
        if kind != "profile" or row["target_uid"] is None:
            return []
        partition = typed_scope(scope, kind, self.recipe)
        target = self.state.get_memories(partition, [row["target_uid"]]).get(
            row["target_uid"]
        )
        if (
            target is None
            or target.version != row["target_version"]
            or target.metadata.get("user_id") != scope.user_id
        ):
            raise StateUnavailable("Stale or foreign Profile mutation target")
        old = target.metadata["profile_delta_item"]
        new = row["profile_item"]
        if any(
            old[k] != new[k] for k in ("subject", "attribute", "scope", "condition")
        ):
            raise ValueError("Profile mutation exceeds structural target boundary")
        rel = {"SUPERSEDE": "PROFILE_SUPERSEDES", "CLOSE": "PROFILE_RETRACTS"}.get(
            row["operation"], "PROFILE_REPLACES"
        )
        return [
            Edge(
                "edge_" + digest([uid, target.logical_uid, rel]),
                uid,
                target.logical_uid,
                rel,
                provenance={"root_job": root.job_id},
            )
        ]

    def eligible_history(self, raw_scope, memories):
        """Read-only Profile history, excluding retractions and invalid sources.

        The ordinary current-view callback remains unchanged. Supersession edges
        determine historical visibility; model text does not decide lineage.
        """
        from copy import deepcopy
        from dataclasses import replace

        accepted = super().eligible(raw_scope, memories)
        partition = typed_scope(raw_scope, "profile", self.recipe)
        result = {}
        for uid, m in accepted.items():
            if (
                m.metadata.get("hlm_kind") != "profile"
                or m.metadata.get("operation") == "CLOSE"
            ):
                continue
            edges = self.state._rows(
                "SELECT source_uid,relation_type FROM graph_edges WHERE scope=? AND target_uid=? AND active=1 "
                "AND relation_type IN ('PROFILE_REPLACES','PROFILE_SUPERSEDES','PROFILE_RETRACTS')",
                (partition.key, uid),
            )
            successors = super().eligible(
                raw_scope,
                self.state.get_memories(partition, [e["source_uid"] for e in edges]),
            )
            live_edges = [e for e in edges if e["source_uid"] in successors]
            if any(e["relation_type"] == "PROFILE_RETRACTS" for e in live_edges):
                continue
            if live_edges:
                metadata = deepcopy(m.metadata)
                metadata["hlm_item"]["profile_item"]["status"] = "historical"
                metadata["historical_view_from_publication_edges"] = True
                m = replace(m, metadata=metadata)
            result[uid] = m
        return result

    def eligible(self, raw_scope, memories):
        accepted = super().eligible(raw_scope, memories)
        partition = typed_scope(raw_scope, "profile", self.recipe)
        result = {}
        for uid, m in accepted.items():
            parents = m.metadata["hlm_item"].get("aggregate_parents", [])
            if parents:
                hs = typed_scope(raw_scope, "hierarchical", self.recipe)
                loaded = self.state.get_memories(hs, [p["uid"] for p in parents])
                live = super().eligible(raw_scope, loaded)
                if any(
                    p["uid"] not in live
                    or live[p["uid"]].version != p["version"]
                    or digest(live[p["uid"]].text_content) != p["text_hash"]
                    for p in parents
                ):
                    continue
            if m.metadata["hlm_kind"] == "profile":
                if m.metadata["operation"] == "CLOSE":
                    continue
                edges = self.state._rows(
                    "SELECT source_uid FROM graph_edges WHERE scope=? AND target_uid=? AND active=1 AND relation_type IN ('PROFILE_REPLACES','PROFILE_SUPERSEDES','PROFILE_RETRACTS')",
                    (partition.key, uid),
                )
                # An incomplete successor root cannot hide a completed predecessor.
                successors = self.state.get_memories(
                    partition, [e["source_uid"] for e in edges]
                )
                if super().eligible(raw_scope, successors):
                    continue
            result[uid] = m
        return result

    def provenance_bindings(self, scope, kind, row, sources):
        if not row.get("aggregate_parents"):
            return super().provenance_bindings(scope, kind, row, sources)
        partition = typed_scope(scope, "hierarchical", self.recipe)
        parents = self.state.get_memories(
            partition, [p["uid"] for p in row["aggregate_parents"]]
        )
        bound = []
        for p in row["aggregate_parents"]:
            parent = parents.get(p["uid"])
            if (
                parent is None
                or parent.version != p["version"]
                or digest(parent.text_content) != p["text_hash"]
            ):
                raise StateUnavailable("Overview parent identity changed")
            bound.append((partition, parent, True, [parent.text_content]))
        return bound

    def build_overview(self, scope, parent_uids):
        """Bounded, non-generative cross-session organization of accepted L1s.

        No new claims/predictions: retain each complete L1 with its session label.
        Edges point to L1 only; Raw hydration follows the existing L1 provenance.
        """
        if "hierarchical" not in self.kinds():
            raise ValueError("Hierarchical tower disabled")
        if not 2 <= len(parent_uids) <= 4 or len(set(parent_uids)) != len(parent_uids):
            raise ValueError("Overview takes two to four unique L1 parents")
        with self.gates.lease(scope, write=True):
            partition = typed_scope(scope, "hierarchical", self.recipe)
            parents = self.eligible(
                scope, self.state.get_memories(partition, parent_uids)
            )
            if set(parents) != set(parent_uids) or any(
                p.metadata["hlm_item"].get("level") != "L1_Summary"
                for p in parents.values()
            ):
                raise ValueError("Only complete current L1 parents allowed")
            versions = {
                u: v
                for p in parents.values()
                for u, v in p.metadata["source_versions"].items()
            }
            evidence = []
            contexts = set()
            parts = []
            for uid in parent_uids:
                p = parents[uid]
                item = p.metadata["hlm_item"]
                evidence.extend(e for e in item["evidence"] if e not in evidence)
                contexts.update(item.get("context_source_uids", []))
                parts.append(p.text_content)
            contexts.difference_update(e["source_uid"] for e in evidence)
            text = (
                "[Cross-session overview; constituent summaries retain their original attribution]\n"
                + "\n".join(parts)
            )
            if len(text) > 9000:
                raise ValueError("Overview exceeds bounded compression contract")
            row = {
                "text": text,
                "evidence": evidence,
                "context_source_uids": sorted(contexts),
                "confidence": min(
                    p.metadata["hlm_item"]["confidence"] for p in parents.values()
                ),
                "level": "L2_Aggregate",
                "semantic_authority": "SOURCE_COMPRESSION_ONLY",
                "aggregate_parents": [
                    {
                        "uid": u,
                        "version": parents[u].version,
                        "text_hash": digest(parents[u].text_content),
                    }
                    for u in parent_uids
                ],
            }
            recipe = self.recipe
            publish = self.publish_mutation

            class AggregatePublisher(CanonicalHighLevelMemory):
                def publish_mutation(self, *args):
                    return publish(*args)

                def kinds(self):
                    return ("hierarchical",)

                def root_fingerprint(self, scope, sources):
                    return digest([recipe.fingerprint, "L2", row["aggregate_parents"]])

                def prepare(self, scope, sources, worker, root):
                    return {
                        "outputs": {"hierarchical": [row]},
                        "errors": {},
                        "receipt_hashes": {},
                    }

            writer = AggregatePublisher(self.coordinator, self.recipe, None)
            try:
                return writer._build(scope, list(versions))
            finally:
                writer.close()

    def verify_provenance(self, scope, kind, memories):
        ordinary = {
            u: m
            for u, m in memories.items()
            if not m.metadata["hlm_item"].get("aggregate_parents")
        }
        result = super().verify_provenance(scope, kind, ordinary)
        partition = typed_scope(scope, kind, self.recipe)
        from .heterogeneous_worker import reference_uid

        for uid, m in memories.items():
            for parent in m.metadata["hlm_item"].get("aggregate_parents", []):
                ref = reference_uid(partition, parent["uid"], parent["version"])
                edges = self.state._rows(
                    "SELECT source_uid,target_uid,relation_type FROM graph_edges WHERE scope=? AND active=1 AND ((source_uid=? AND target_uid=?) OR (source_uid=? AND target_uid=?))",
                    (partition.key, uid, ref, ref, uid),
                )
                if {
                    (e["source_uid"], e["target_uid"], e["relation_type"])
                    for e in edges
                } != {(uid, ref, "DERIVED_FROM"), (ref, uid, "SUPPORTS_HIERARCHICAL")}:
                    result["missing_reverse_links"].append([uid, parent["uid"]])
        result["status"] = (
            "PASS"
            if not result["missing_reverse_links"] and not result["wrong_scope_links"]
            else "FAIL"
        )
        return result

    def raw_to_derived(self, scope, source_uid):
        source = self.state.get_memories(scope, [source_uid]).get(source_uid)
        if source is None:
            return {}
        from .heterogeneous_worker import reference_uid

        ref = reference_uid(scope, source_uid, source.version)
        result = {}
        for kind in self.kinds():
            partition = typed_scope(scope, kind, self.recipe)
            rows = self.state._rows(
                "SELECT target_uid FROM graph_edges WHERE scope=? AND source_uid=? AND active=1 AND relation_type LIKE 'SUPPORTS_%'",
                (partition.key, ref),
            )
            result[kind] = list(
                self.eligible(
                    scope,
                    self.state.get_memories(partition, [r["target_uid"] for r in rows]),
                )
            )
        if result.get("hierarchical"):
            partition = typed_scope(scope, "hierarchical", self.recipe)
            l1 = self.state.get_memories(partition, result["hierarchical"])
            aggregates = []
            for uid, m in l1.items():
                pointer = reference_uid(partition, uid, m.version)
                rows = self.state._rows(
                    "SELECT target_uid FROM graph_edges WHERE scope=? AND source_uid=? AND active=1 AND relation_type='SUPPORTS_HIERARCHICAL'",
                    (partition.key, pointer),
                )
                aggregates.extend(r["target_uid"] for r in rows)
            visible = self.eligible(
                scope, self.state.get_memories(partition, aggregates)
            )
            result["hierarchical"] = list(
                dict.fromkeys([*result["hierarchical"], *visible])
            )
        return result

    def derived_to_raw(self, scope, kind, uid):
        partition = typed_scope(scope, kind, self.recipe)
        memories = self.eligible(scope, self.state.get_memories(partition, [uid]))
        if uid not in memories:
            return {"support": {}, "context": {}}
        row = memories[uid].metadata["hlm_item"]
        supporting = {
            e["source_uid"] for e in row.get("evidence", row.get("mentions", []))
        }
        contexts = set(row.get("context_source_uids", []))
        return {
            "support": self.state.get_memories(scope, list(supporting)),
            "context": self.state.get_memories(scope, list(contexts)),
        }

    def check_consistency(self, scope):
        report = {}
        visible_stale = 0
        cross_user = []
        orphan = []
        duplicate = []
        for kind in self.kinds():
            partition = typed_scope(scope, kind, self.recipe)
            rows = self.state._rows(
                "SELECT logical_uid FROM memories WHERE scope=? AND memory_type=?",
                (partition.key, kind),
            )
            all_rows = self.state.get_memories(
                partition, [r["logical_uid"] for r in rows]
            )
            eligible = self.eligible(scope, all_rows)
            checked = self.verify_provenance(scope, kind, eligible)
            checked.update(
                total=len(all_rows),
                visible=len(eligible),
                hidden=len(all_rows) - len(eligible),
            )
            visible_stale += len(checked["wrong_scope_links"])
            graph = self.state._rows(
                "SELECT source_uid,target_uid,relation_type FROM graph_edges WHERE scope=? AND active=1",
                (partition.key,),
            )
            nodes = self.state.get_memories(
                partition,
                list({u for e in graph for u in (e["source_uid"], e["target_uid"])}),
            )
            seen = set()
            for edge in graph:
                key = (edge["source_uid"], edge["target_uid"], edge["relation_type"])
                if key in seen:
                    duplicate.append([kind, *key])
                seen.add(key)
                for u in key[:2]:
                    if u not in nodes:
                        orphan.append([kind, u])
                        continue
                    node = nodes[u]
                    pointer = node.metadata.get("hlm_reference", {})
                    if node.metadata.get("user_id") != scope.user_id or (
                        pointer and pointer.get("user_id") != scope.user_id
                    ):
                        cross_user.append([kind, u])
            report[kind] = checked
        return {
            "types": report,
            "stale_source_visible": visible_stale,
            "cross_user_edges": cross_user,
            "orphan_nodes": orphan,
            "duplicate_edges": duplicate,
            "status": "PASS"
            if not cross_user
            and not orphan
            and not duplicate
            and all(r["status"] == "PASS" for r in report.values())
            else "FAIL",
        }
