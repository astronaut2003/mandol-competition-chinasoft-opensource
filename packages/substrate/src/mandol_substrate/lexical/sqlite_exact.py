"""Mutable Mandol BM25: SQL postings, exact live per-user statistics.

Python holds only a prepared document, bounded mutation, query, or Top-K. Empty
eligible documents count in N, as in the legacy _ingest_document. Inactive and
deleted memories are outside the formal live corpus; space filters only restrict
candidates. Search requires a publication read lease from the coordinator.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

from ..state.models import ProfileMismatch, StateUnavailable, digest


@dataclass(frozen=True)
class PreparedDocument:
    terms: tuple[tuple[str, int], ...]
    doc_len: int
    token_fingerprint: str
    profile_hash: str
    eligible: bool


@dataclass(frozen=True)
class LexicalHit:
    logical_uid: str
    score: float


class SqliteExactBm25Backend:
    scorer_version = "mandol-exact-bm25-1"

    def __init__(self, state, profile, processor):
        if processor.fingerprint != profile.preprocessing_fingerprint:
            raise ProfileMismatch("lexical processor does not match frozen profile")
        self.state, self.processor = state, processor
        self.k1, self.b = profile.bm25_k1, profile.bm25_b
        self.statistics_policy = profile.options.get("lexical_statistics", "scope-v1")
        self.minimum_score = profile.options.get("lexical_minimum_score", 1e-6)
        if self.statistics_policy not in {"scope-v1", "candidate-subset-v1"} or self.minimum_score < 0:
            raise ValueError("unsupported lexical statistics policy")
        settings = {
                "schema": 1,
                "scorer": self.scorer_version,
                "processor": processor.fingerprint,
                "k1": self.k1,
                "b": self.b,
                "eligibility": "active-and-not-deleted",
                "tie": "logical-uid-ascending",
                "threshold": self.minimum_score,
            }
        if self.statistics_policy != "scope-v1":
            settings["statistics"] = self.statistics_policy
        self.profile_hash = digest(settings)

    @property
    def fault(self):
        return self.state.fault

    @contextmanager
    def reader(self):
        # Independent WAL readers: a high-DF query does not hold the global
        # writer connection lock or serialize other users' searches.
        self.state._check_open()
        db = sqlite3.connect(self.state.path, isolation_level=None, timeout=5)
        db.row_factory = sqlite3.Row
        db.create_function("ln", 1, math.log, deterministic=True)
        try:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("BEGIN")
            yield db
        finally:
            db.close()

    def ensure_scope_in_transaction(self, db, scope):
        db.execute(
            "INSERT OR IGNORE INTO lexical_corpus_stats VALUES (?,0,0,0,?)",
            (scope.key, self.profile_hash),
        )

    def prepare_document(self, memory):
        tokens = self.processor.document(memory.text_content).split()
        return PreparedDocument(
            tuple(sorted(Counter(tokens).items())),
            len(tokens),
            digest(tokens),
            self.profile_hash,
            memory.active and memory.lifecycle != "deleted",
        )

    def delete_in_transaction(self, db, scope, uid):
        if not db.in_transaction:
            raise RuntimeError("lexical mutation requires caller transaction")
        old = db.execute(
            "SELECT doc_len FROM lexical_documents WHERE scope=? AND logical_uid=?",
            (scope.key, uid),
        ).fetchone()
        if old is None:
            return
        # Bounded by this document, never the corpus vocabulary/postings.
        terms = db.execute(
            "SELECT term FROM lexical_postings WHERE scope=? AND logical_uid=?",
            (scope.key, uid),
        ).fetchall()
        db.execute(
            "DELETE FROM lexical_postings WHERE scope=? AND logical_uid=?",
            (scope.key, uid),
        )
        self.fault("lexical_old_postings_deleted")
        for term in terms:
            changed = db.execute(
                "UPDATE lexical_term_stats SET df=df-1 WHERE scope=? AND term=? AND df>0",
                (scope.key, term[0]),
            )
            if changed.rowcount != 1:
                raise StateUnavailable("lexical DF missing/corrupt")
            db.execute(
                "DELETE FROM lexical_term_stats WHERE scope=? AND term=? AND df=0",
                (scope.key, term[0]),
            )
        self.fault("lexical_old_df_decremented")
        db.execute(
            "DELETE FROM lexical_documents WHERE scope=? AND logical_uid=?",
            (scope.key, uid),
        )
        db.execute(
            "UPDATE lexical_corpus_stats SET n_docs=n_docs-1,total_doc_len=total_doc_len-? WHERE scope=?",
            (old[0], scope.key),
        )
        self.fault("lexical_corpus_decremented")

    def update_in_transaction(self, db, scope, memory, prepared, seq):
        if not db.in_transaction:
            raise RuntimeError("lexical mutation requires caller transaction")
        if prepared.profile_hash != self.profile_hash:
            raise ProfileMismatch("prepared lexical profile mismatch")
        old = db.execute(
            "SELECT * FROM lexical_documents WHERE scope=? AND logical_uid=?",
            (scope.key, memory.logical_uid),
        ).fetchone()
        if (
            old
            and prepared.eligible
            and old["token_fingerprint"] == prepared.token_fingerprint
        ):
            if old["lexical_profile_hash"] != self.profile_hash:
                raise ProfileMismatch("existing lexical document profile mismatch")
            db.execute(
                "UPDATE lexical_documents SET version=?,updated_seq=? WHERE scope=? AND logical_uid=?",
                (memory.version, seq, scope.key, memory.logical_uid),
            )
            return
        self.delete_in_transaction(db, scope, memory.logical_uid)
        if not prepared.eligible:
            return
        db.execute(
            "INSERT INTO lexical_documents VALUES (?,?,?,?,?,?,?)",
            (
                scope.key,
                memory.logical_uid,
                memory.version,
                prepared.doc_len,
                prepared.token_fingerprint,
                self.profile_hash,
                seq,
            ),
        )
        self.fault("lexical_new_document")
        db.executemany(
            "INSERT INTO lexical_postings VALUES (?,?,?,?)",
            ((scope.key, term, memory.logical_uid, tf) for term, tf in prepared.terms),
        )
        self.fault("lexical_new_postings")
        db.executemany(
            "INSERT INTO lexical_term_stats VALUES (?,?,1) ON CONFLICT(scope,term) DO UPDATE SET df=df+1",
            ((scope.key, term) for term, _ in prepared.terms),
        )
        self.fault("lexical_new_df_incremented")
        db.execute(
            "UPDATE lexical_corpus_stats SET n_docs=n_docs+1,total_doc_len=total_doc_len+? WHERE scope=?",
            (prepared.doc_len, scope.key),
        )
        self.fault("lexical_corpus_incremented")

    insert_in_transaction = update_in_transaction

    def advance_in_transaction(self, db, scope, seq):
        changed = db.execute(
            "UPDATE lexical_corpus_stats SET lexical_seq=? WHERE scope=? AND lexical_seq=? AND lexical_profile_hash=?",
            (seq, scope.key, seq - 1, self.profile_hash),
        )
        if changed.rowcount != 1:
            raise StateUnavailable(
                "lexical sequence/profile mismatch; explicit repair required"
            )
        self.fault("lexical_seq_advanced")

    def _assert_ready(self, db, scope, published_seq=None):
        row = db.execute(
            "SELECT s.canonical_seq,s.published_seq,l.lexical_seq,l.lexical_profile_hash "
            "FROM scopes s LEFT JOIN lexical_corpus_stats l USING(scope) WHERE s.scope=?",
            (scope.key,),
        ).fetchone()
        if (
            not row
            or row["canonical_seq"] != row["published_seq"]
            or row["lexical_seq"] != row["published_seq"]
            or row["lexical_profile_hash"] != self.profile_hash
            or (published_seq is not None and row["published_seq"] != published_seq)
        ):
            raise StateUnavailable("lexical publication invariant not satisfied")

    def _query(self, scope, query, limit, spaces=None, candidate_uids=None):
        if not 0 < limit <= 16384:
            raise ValueError("lexical limit outside bounded candidate budget")
        terms = Counter(self.processor.query(query).split())
        if len(terms) > 512:
            raise ValueError("too many distinct query terms")
        if not terms or candidate_uids == [] or spaces == []:
            return None, ()
        if self.statistics_policy == "candidate-subset-v1" and candidate_uids is not None:
            return self._subset_query(scope, terms, limit, spaces, candidate_uids)
        values = ",".join("(?,?)" for _ in terms)
        params = [v for item in terms.items() for v in item]
        sql = f"""WITH q(term,qtf) AS (VALUES {values}),
        scores AS (
          SELECT p.logical_uid, SUM(q.qtf * ln(1.0+(c.n_docs-s.df+0.5)/(s.df+0.5))
            * p.tf*(?+1.0)/(p.tf+?*(1.0-?+?*d.doc_len /
            CASE WHEN c.total_doc_len=0 THEN 1.0 ELSE 1.0*c.total_doc_len/c.n_docs END))) AS score
          FROM q CROSS JOIN lexical_postings p
          JOIN lexical_documents d ON d.scope=p.scope AND d.logical_uid=p.logical_uid
          JOIN lexical_term_stats s ON s.scope=p.scope AND s.term=p.term
          JOIN lexical_corpus_stats c ON c.scope=p.scope
          WHERE p.scope=? AND p.term=q.term"""
        params.extend([self.k1, self.k1, self.b, self.b, scope.key])
        if spaces is not None:
            if len(spaces) > 128:
                raise ValueError("too many candidate spaces")
            sql += (
                " AND EXISTS (SELECT 1 FROM memberships m WHERE m.scope=p.scope AND m.logical_uid=p.logical_uid AND m.space_id IN ("
                + ",".join("?" for _ in spaces)
                + "))"
            )
            params.extend(spaces)
        if candidate_uids is not None:
            if len(candidate_uids) > 16384:
                raise ValueError("candidate filter exceeds bounded budget")
            sql += (
                " AND p.logical_uid IN (" + ",".join("?" for _ in candidate_uids) + ")"
            )
            params.extend(candidate_uids)
        sql += " GROUP BY p.logical_uid) SELECT logical_uid,score FROM scores WHERE score>? ORDER BY score DESC,logical_uid ASC LIMIT ?"
        params.extend([self.minimum_score, limit])
        return sql, params

    def _subset_query(self, scope, terms, limit, spaces, candidate_uids):
        if len(candidate_uids) > 16384:
            raise ValueError("candidate filter exceeds bounded budget")
        ids = sorted(set(candidate_uids))
        params = [scope.key, *ids]
        eligible = "SELECT * FROM lexical_documents WHERE scope=? AND logical_uid IN (" + ",".join("?" for _ in ids) + ")"
        if spaces is not None:
            if len(spaces) > 128:
                raise ValueError("too many candidate spaces")
            eligible += " AND logical_uid IN (SELECT logical_uid FROM memberships WHERE scope=? AND space_id IN (" + ",".join("?" for _ in spaces) + "))"
            params.extend([scope.key, *spaces])
        values = ",".join("(?,?)" for _ in terms)
        params.extend(v for item in terms.items() for v in item)
        sql = f"""WITH eligible AS ({eligible}), q(term,qtf) AS (VALUES {values}),
          c AS (SELECT count(*) AS n, coalesce(avg(doc_len),1.0) AS average FROM eligible),
          df AS (SELECT p.term,count(*) AS n FROM lexical_postings p JOIN eligible d
                 ON p.scope=d.scope AND p.logical_uid=d.logical_uid GROUP BY p.term),
          scores AS (SELECT p.logical_uid, SUM(q.qtf*ln(1.0+(c.n-df.n+0.5)/(df.n+0.5))
           *p.tf*(?+1.0)/(p.tf+?*(1.0-?+?*d.doc_len/CASE WHEN c.average=0 THEN 1.0 ELSE c.average END))) AS score
           FROM q JOIN lexical_postings p ON p.term=q.term
           JOIN eligible d ON d.scope=p.scope AND d.logical_uid=p.logical_uid
           JOIN df ON df.term=p.term CROSS JOIN c GROUP BY p.logical_uid)
          SELECT logical_uid,score FROM scores WHERE score>? ORDER BY score DESC,logical_uid ASC LIMIT ?"""
        params.extend([self.k1, self.k1, self.b, self.b, self.minimum_score, limit])
        return sql, params

    def search(
        self,
        scope,
        query,
        *,
        limit=10,
        published_seq=None,
        spaces=None,
        candidate_uids=None,
    ):
        sql, params = self._query(scope, query, limit, spaces, candidate_uids)
        with self.reader() as db:
            self._assert_ready(db, scope, published_seq)
            return (
                [] if sql is None else [LexicalHit(*r) for r in db.execute(sql, params)]
            )

    def explain(self, scope, query, **kwargs):
        sql, params = self._query(scope, query, kwargs.pop("limit", 10), **kwargs)
        with self.reader() as db:
            return (
                []
                if sql is None
                else [dict(r) for r in db.execute("EXPLAIN QUERY PLAN " + sql, params)]
            )

    def inspect_stats(self, scope, *, terms=()):
        if len(terms) > 512:
            raise ValueError("stats term batch exceeds budget")
        with self.reader() as db:
            row = db.execute(
                "SELECT * FROM lexical_corpus_stats WHERE scope=?", (scope.key,)
            ).fetchone()
            if row is None:
                raise StateUnavailable("missing lexical corpus manifest")
            result = dict(row)
            result["avgdl"] = row["total_doc_len"] / max(1, row["n_docs"])
            result["df"] = {
                term: (
                    db.execute(
                        "SELECT df FROM lexical_term_stats WHERE scope=? AND term=?",
                        (scope.key, term),
                    ).fetchone()
                    or [0]
                )[0]
                for term in terms
            }
            return result

    def verify_scope(self, scope):
        with self.reader() as db:
            row = db.execute(
                "SELECT * FROM lexical_corpus_stats WHERE scope=?", (scope.key,)
            ).fetchone()
            current = db.execute(
                "SELECT canonical_seq FROM scopes WHERE scope=?", (scope.key,)
            ).fetchone()
            counts = db.execute(
                "SELECT count(*),coalesce(sum(doc_len),0) FROM lexical_documents WHERE scope=?",
                (scope.key,),
            ).fetchone()
            if (
                not row
                or not current
                or row["lexical_seq"] != current[0]
                or row["lexical_profile_hash"] != self.profile_hash
                or (row["n_docs"], row["total_doc_len"]) != tuple(counts)
            ):
                raise StateUnavailable("lexical corpus statistics/sequence corrupt")
            checks = [
                (
                    "SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND lifecycle<>'deleted' EXCEPT SELECT logical_uid FROM lexical_documents WHERE scope=?",
                    (scope.key, scope.key),
                ),
                (
                    "SELECT logical_uid FROM lexical_documents WHERE scope=? EXCEPT SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND lifecycle<>'deleted'",
                    (scope.key, scope.key),
                ),
                (
                    "SELECT term,df FROM lexical_term_stats WHERE scope=? EXCEPT SELECT term,count(*) FROM lexical_postings WHERE scope=? GROUP BY term",
                    (scope.key, scope.key),
                ),
                (
                    "SELECT term,count(*) FROM lexical_postings WHERE scope=? GROUP BY term EXCEPT SELECT term,df FROM lexical_term_stats WHERE scope=?",
                    (scope.key, scope.key),
                ),
                (
                    "SELECT d.logical_uid FROM lexical_documents d JOIN memories m USING(scope,logical_uid) WHERE d.scope=? AND (d.version<>m.version OR d.updated_seq<>m.commit_seq OR d.lexical_profile_hash<>? OR d.doc_len<>(SELECT coalesce(sum(tf),0) FROM lexical_postings p WHERE p.scope=d.scope AND p.logical_uid=d.logical_uid))",
                    (scope.key, self.profile_hash),
                ),
            ]
            for sql, params in checks:
                if db.execute(sql, params).fetchone():
                    raise StateUnavailable(
                        "lexical document/posting/DF invariant corrupt"
                    )
        return True

    def rebuild_scope(self, scope):
        """Explicit offline repair. Caller MUST own scope write lease.

        Tokens stream into disk-backed TEMP staging before the single atomic
        replacement. No model inference in T1, no O(corpus) Python dictionary.
        A crash discards staging or rolls back the replacement.
        """
        current = self.state.scope_state(scope)["canonical_seq"]
        with self.state._lock:
            db = self.state._connection
            db.execute(
                "CREATE TEMP TABLE repair_docs(uid TEXT PRIMARY KEY,version INT,dl INT,fp TEXT,seq INT)"
            )
            db.execute("CREATE TEMP TABLE repair_terms(uid TEXT,term TEXT,tf INT)")
            try:
                for batch in self.state.iter_memories(scope):
                    for m in batch:
                        p = self.prepare_document(m)
                        if p.eligible:
                            db.execute(
                                "INSERT INTO repair_docs VALUES (?,?,?,?,?)",
                                (
                                    m.logical_uid,
                                    m.version,
                                    p.doc_len,
                                    p.token_fingerprint,
                                    m.commit_seq,
                                ),
                            )
                            db.executemany(
                                "INSERT INTO repair_terms VALUES (?,?,?)",
                                ((m.logical_uid, t, tf) for t, tf in p.terms),
                            )
                with self.state.transaction():
                    if self.state.scope_state(scope)["canonical_seq"] != current:
                        raise StateUnavailable("scope changed during explicit repair")
                    for table in (
                        "lexical_postings",
                        "lexical_documents",
                        "lexical_term_stats",
                        "lexical_corpus_stats",
                    ):
                        db.execute(f"DELETE FROM {table} WHERE scope=?", (scope.key,))
                    db.execute(
                        "INSERT INTO lexical_documents SELECT ?,uid,version,dl,fp,?,seq FROM repair_docs",
                        (scope.key, self.profile_hash),
                    )
                    db.execute(
                        "INSERT INTO lexical_postings SELECT ?,term,uid,tf FROM repair_terms",
                        (scope.key,),
                    )
                    db.execute(
                        "INSERT INTO lexical_term_stats SELECT ?,term,count(*) FROM repair_terms GROUP BY term",
                        (scope.key,),
                    )
                    db.execute(
                        "INSERT INTO lexical_corpus_stats SELECT ?,count(*),coalesce(sum(dl),0),?,? FROM repair_docs",
                        (scope.key, current, self.profile_hash),
                    )
            finally:
                db.execute("DROP TABLE repair_terms")
                db.execute("DROP TABLE repair_docs")
        return self.verify_scope(scope)

    def health(self):
        with self.reader() as db:
            return db.execute("SELECT 1").fetchone()[0] == 1

    def stats(self):
        return {
            "backend": "sqlite_exact",
            "scorer": self.scorer_version,
            "profile_hash": self.profile_hash,
            "python_corpus_state": False,
        }
