"""Schema v2: canonical semantics and transactional lexical projection; no vectors."""

SCHEMA_VERSION = 5

SCHEMA = """
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE scopes (
 scope TEXT PRIMARY KEY, namespace TEXT NOT NULL, user_id TEXT NOT NULL,
 canonical_seq INTEGER NOT NULL DEFAULT 0, published_seq INTEGER NOT NULL DEFAULT 0,
 profile_hash TEXT NOT NULL, profile_json TEXT NOT NULL, schema_version INTEGER NOT NULL,
 created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL,
 UNIQUE(namespace,user_id), CHECK(published_seq<=canonical_seq)
);
CREATE TABLE memory_spaces (
 scope TEXT NOT NULL REFERENCES scopes(scope), space_id TEXT NOT NULL,
 kind TEXT NOT NULL DEFAULT 'logical', PRIMARY KEY(scope,space_id)
);
CREATE TABLE space_edges (
 scope TEXT NOT NULL, parent_space TEXT NOT NULL, child_space TEXT NOT NULL,
 PRIMARY KEY(scope,parent_space,child_space), CHECK(parent_space<>child_space),
 FOREIGN KEY(scope,parent_space) REFERENCES memory_spaces(scope,space_id),
 FOREIGN KEY(scope,child_space) REFERENCES memory_spaces(scope,space_id)
);
CREATE TABLE memories (
 scope TEXT NOT NULL REFERENCES scopes(scope), logical_uid TEXT NOT NULL,
 text_content TEXT NOT NULL, raw_data TEXT NOT NULL, metadata TEXT NOT NULL,
 session_id TEXT NOT NULL, request_id TEXT NOT NULL, turn_index INTEGER NOT NULL, role TEXT NOT NULL,
 memory_tower TEXT NOT NULL, memory_level TEXT NOT NULL, memory_type TEXT NOT NULL,
 version INTEGER NOT NULL CHECK(version>0), active INTEGER NOT NULL CHECK(active IN (0,1)),
 lifecycle TEXT NOT NULL, supersedes_uid TEXT,
 business_timestamp_ms INTEGER, valid_from_ms INTEGER, valid_to_ms INTEGER,
 created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL, commit_seq INTEGER NOT NULL,
 PRIMARY KEY(scope,logical_uid),
 FOREIGN KEY(scope,supersedes_uid) REFERENCES memories(scope,logical_uid) DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX memories_seq ON memories(scope,commit_seq);
CREATE TABLE memberships (
 scope TEXT NOT NULL, logical_uid TEXT NOT NULL, space_id TEXT NOT NULL,
 PRIMARY KEY(scope,logical_uid,space_id),
 FOREIGN KEY(scope,logical_uid) REFERENCES memories(scope,logical_uid),
 FOREIGN KEY(scope,space_id) REFERENCES memory_spaces(scope,space_id)
);
CREATE INDEX memberships_space ON memberships(scope,space_id,logical_uid);
CREATE TABLE graph_nodes (
 scope TEXT NOT NULL, node_uid TEXT NOT NULL, node_kind TEXT NOT NULL,
 memory_uid TEXT, space_id TEXT, metadata TEXT NOT NULL DEFAULT '{}',
 PRIMARY KEY(scope,node_uid),
 FOREIGN KEY(scope,memory_uid) REFERENCES memories(scope,logical_uid),
 FOREIGN KEY(scope,space_id) REFERENCES memory_spaces(scope,space_id),
 CHECK((node_kind='memory' AND memory_uid IS NOT NULL AND space_id IS NULL)
    OR (node_kind='space' AND space_id IS NOT NULL AND memory_uid IS NULL))
);
CREATE TABLE graph_edges (
 scope TEXT NOT NULL, edge_id TEXT NOT NULL, source_uid TEXT NOT NULL, target_uid TEXT NOT NULL,
 source_kind TEXT NOT NULL, target_kind TEXT NOT NULL, relation_type TEXT NOT NULL,
 properties TEXT NOT NULL, provenance TEXT NOT NULL, active INTEGER NOT NULL,
 version INTEGER NOT NULL, commit_seq INTEGER NOT NULL,
 created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL,
 PRIMARY KEY(scope,edge_id),
 FOREIGN KEY(scope,source_uid) REFERENCES graph_nodes(scope,node_uid),
 FOREIGN KEY(scope,target_uid) REFERENCES graph_nodes(scope,node_uid)
);
CREATE INDEX graph_outgoing ON graph_edges(scope,source_uid,active,relation_type);
CREATE INDEX graph_incoming ON graph_edges(scope,target_uid,active,relation_type);
CREATE TABLE mutations (
 scope TEXT NOT NULL REFERENCES scopes(scope), mutation_seq INTEGER NOT NULL,
 mutation_type TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','done')),
 projection_status TEXT NOT NULL, targets_hash TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
 created_at_ms INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL,
 PRIMARY KEY(scope,mutation_seq)
);
CREATE TABLE outbox_targets (
 scope TEXT NOT NULL, mutation_seq INTEGER NOT NULL, logical_uid TEXT NOT NULL,
 operation TEXT NOT NULL CHECK(operation IN ('upsert','delete')),
 PRIMARY KEY(scope,mutation_seq,logical_uid),
 FOREIGN KEY(scope,mutation_seq) REFERENCES mutations(scope,mutation_seq)
);
CREATE TABLE requests (
 scope TEXT NOT NULL, request_id TEXT NOT NULL, payload_digest TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','success')),
 mutation_seq INTEGER NOT NULL, result_json TEXT NOT NULL, last_error TEXT,
 PRIMARY KEY(scope,request_id),
 FOREIGN KEY(scope,mutation_seq) REFERENCES mutations(scope,mutation_seq)
);
CREATE TABLE tombstones (
 scope TEXT NOT NULL, logical_uid TEXT NOT NULL, version INTEGER NOT NULL,
 commit_seq INTEGER NOT NULL, deleted_at_ms INTEGER NOT NULL,
 PRIMARY KEY(scope,logical_uid),
 FOREIGN KEY(scope,logical_uid) REFERENCES memories(scope,logical_uid)
);
CREATE TABLE build_jobs (
 scope TEXT NOT NULL REFERENCES scopes(scope), job_id TEXT NOT NULL,
 input_versions TEXT NOT NULL, recipe_fingerprint TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
 progress TEXT NOT NULL, output_uids TEXT NOT NULL, commit_seq INTEGER NOT NULL,
 PRIMARY KEY(scope,job_id)
);
"""

LEXICAL_SCHEMA = """
CREATE TABLE lexical_corpus_stats (
 scope TEXT PRIMARY KEY REFERENCES scopes(scope),
 n_docs INTEGER NOT NULL CHECK(n_docs>=0),
 total_doc_len INTEGER NOT NULL CHECK(total_doc_len>=0),
 lexical_seq INTEGER NOT NULL, lexical_profile_hash TEXT NOT NULL
);
CREATE TABLE lexical_documents (
 scope TEXT NOT NULL, logical_uid TEXT NOT NULL, version INTEGER NOT NULL,
 doc_len INTEGER NOT NULL CHECK(doc_len>=0), token_fingerprint TEXT NOT NULL,
 lexical_profile_hash TEXT NOT NULL, updated_seq INTEGER NOT NULL,
 PRIMARY KEY(scope,logical_uid),
 FOREIGN KEY(scope,logical_uid) REFERENCES memories(scope,logical_uid)
);
CREATE TABLE lexical_postings (
 scope TEXT NOT NULL, term TEXT NOT NULL, logical_uid TEXT NOT NULL,
 tf INTEGER NOT NULL CHECK(tf>0), PRIMARY KEY(scope,term,logical_uid),
 FOREIGN KEY(scope,logical_uid) REFERENCES lexical_documents(scope,logical_uid)
);
CREATE INDEX lexical_postings_document ON lexical_postings(scope,logical_uid);
CREATE TABLE lexical_term_stats (
 scope TEXT NOT NULL REFERENCES scopes(scope), term TEXT NOT NULL,
 df INTEGER NOT NULL CHECK(df>=0), PRIMARY KEY(scope,term)
);
"""

PURGE_SCHEMA = """
CREATE TABLE scope_purges (
 scope TEXT PRIMARY KEY REFERENCES scopes(scope),
 purge_seq INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','done')),
 result_json TEXT NOT NULL
);
"""

RUNTIME_SCHEMA = """
CREATE TABLE request_preparations (
 scope TEXT NOT NULL REFERENCES scopes(scope), request_id TEXT NOT NULL,
 payload_digest TEXT NOT NULL, checkpoint_json TEXT NOT NULL, updated_at_ms INTEGER NOT NULL,
 PRIMARY KEY(scope,request_id)
);
CREATE TABLE projection_frontiers (
 scope TEXT NOT NULL REFERENCES scopes(scope), channel TEXT NOT NULL,
 published_seq INTEGER NOT NULL, status TEXT NOT NULL,
 error_class TEXT, updated_at_ms INTEGER NOT NULL,
 PRIMARY KEY(scope,channel)
);
CREATE TABLE runtime_events (
 scope TEXT NOT NULL REFERENCES scopes(scope), operation TEXT NOT NULL,
 channel TEXT NOT NULL, error_class TEXT NOT NULL, count INTEGER NOT NULL,
 updated_at_ms INTEGER NOT NULL,
 PRIMARY KEY(scope,operation,channel,error_class)
);
"""

SCOPE_ATTRIBUTES_SCHEMA = """
CREATE TABLE scope_attributes (
 scope TEXT PRIMARY KEY REFERENCES scopes(scope), value_json TEXT NOT NULL
);
"""
