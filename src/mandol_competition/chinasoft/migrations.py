"""Additive, transactional and restart-safe application schema migration."""
from .store import now

REPOSITORY_DDL = '''
CREATE TABLE IF NOT EXISTS cs_repositories(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, name TEXT NOT NULL, root TEXT NOT NULL, status TEXT NOT NULL, head TEXT, worktree TEXT NOT NULL DEFAULT '{}', generation INTEGER NOT NULL DEFAULT 0, scanned_at INTEGER, error TEXT, statistics TEXT NOT NULL DEFAULT '{}', created_at INTEGER NOT NULL, UNIQUE(owner,workspace,root));
CREATE TABLE IF NOT EXISTS cs_repository_files(repository TEXT NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL, bytes INTEGER NOT NULL, language TEXT NOT NULL, parser TEXT NOT NULL, status TEXT NOT NULL, error TEXT, version INTEGER NOT NULL, uids TEXT NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY(repository,path));
CREATE TABLE IF NOT EXISTS cs_repository_symbols(repository TEXT NOT NULL, path TEXT NOT NULL, uid TEXT NOT NULL, name TEXT NOT NULL, qualified_name TEXT NOT NULL, kind TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, file_hash TEXT NOT NULL, PRIMARY KEY(repository,uid));
CREATE INDEX IF NOT EXISTS cs_repository_symbol_name ON cs_repository_symbols(repository,name);
CREATE TABLE IF NOT EXISTS cs_repository_links(repository TEXT NOT NULL, path TEXT NOT NULL, uid TEXT NOT NULL, target_path TEXT, target_symbol TEXT, relation TEXT NOT NULL, resolved INTEGER NOT NULL, generation INTEGER NOT NULL, PRIMARY KEY(repository,uid));
CREATE TABLE IF NOT EXISTS cs_repository_scans(job_id TEXT PRIMARY KEY, repository TEXT NOT NULL, snapshot TEXT NOT NULL, statistics TEXT NOT NULL, state TEXT NOT NULL, created_at INTEGER NOT NULL);
'''

def migrate(state):
    with state.transaction() as db:
        db.execute('CREATE TABLE IF NOT EXISTS cs_schema_migrations(version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)')
        for table, column, default in [('cs_scopes','repository',''),('cs_sessions','repository',''),('cs_chat','route','{}')]:
            columns={r['name'] for r in db.execute('PRAGMA table_info('+table+')')}
            if column not in columns: db.execute('ALTER TABLE '+table+' ADD COLUMN '+column+" TEXT NOT NULL DEFAULT '"+default+"'")
        for sql in REPOSITORY_DDL.split(';'):
            if sql.strip(): db.execute(sql)
        db.execute('INSERT OR IGNORE INTO cs_schema_migrations VALUES (2,?)',(now(),))
        if db.execute('SELECT 1 FROM cs_schema_migrations WHERE version=3').fetchone():
            migrate_v31(db)
            migrate_v32(db)
            return
        for column, definition in [('updated_at','INTEGER NOT NULL DEFAULT 0'),('manual_title','INTEGER NOT NULL DEFAULT 0'),
                                   ('archived','INTEGER NOT NULL DEFAULT 0'),('deleted','INTEGER NOT NULL DEFAULT 0'),('unified','INTEGER NOT NULL DEFAULT 0')]:
            if column not in {r['name'] for r in db.execute('PRAGMA table_info(cs_sessions)')}:
                db.execute('ALTER TABLE cs_sessions ADD COLUMN '+column+' '+definition)
        db.execute('UPDATE cs_sessions SET updated_at=created_at WHERE updated_at=0')
        for column,definition in [('attachments',"TEXT NOT NULL DEFAULT '[]'")]:
            if column not in {r['name'] for r in db.execute('PRAGMA table_info(cs_chat)')}:
                db.execute('ALTER TABLE cs_chat ADD COLUMN '+column+' '+definition)
        ddl='''
CREATE TABLE IF NOT EXISTS cs_accounts(id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, display_name TEXT NOT NULL, role TEXT NOT NULL, status TEXT NOT NULL, local_repository INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_account_owners(account TEXT NOT NULL, owner TEXT NOT NULL UNIQUE, claimed_at INTEGER NOT NULL, PRIMARY KEY(account,owner));
CREATE TABLE IF NOT EXISTS cs_auth_sessions(token_hash TEXT PRIMARY KEY, account TEXT NOT NULL, csrf TEXT NOT NULL, expires_at INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS cs_auth_account ON cs_auth_sessions(account,expires_at);
CREATE TABLE IF NOT EXISTS cs_auth_limits(key TEXT PRIMARY KEY, hits INTEGER NOT NULL, resets_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_attachments(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, session TEXT NOT NULL, name TEXT NOT NULL, mime TEXT NOT NULL, kind TEXT NOT NULL, hash TEXT NOT NULL, text TEXT NOT NULL, parser TEXT NOT NULL, bytes INTEGER NOT NULL, created_at INTEGER NOT NULL, UNIQUE(owner,workspace,session,hash,name));
CREATE TABLE IF NOT EXISTS cs_admin_audit(id TEXT PRIMARY KEY, account TEXT NOT NULL, action TEXT NOT NULL, subject TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS cs_session_recent ON cs_sessions(owner,workspace,deleted,archived,updated_at DESC,id);
CREATE INDEX IF NOT EXISTS cs_ingestion_session ON cs_ingestions(owner,workspace,session,created_at);
CREATE INDEX IF NOT EXISTS cs_attachment_session ON cs_attachments(owner,workspace,session,created_at);
CREATE INDEX IF NOT EXISTS cs_memory_session ON memories(scope,session_id,active,created_at_ms);
'''
        for sql in ddl.split(';'):
            if sql.strip(): db.execute(sql)
        db.execute('INSERT INTO cs_schema_migrations VALUES (3,?)',(now(),))
        migrate_v31(db)
        migrate_v32(db)


def migrate_v31(db):
    """Only additive attachment fields; existing canonical rows stay intact."""
    if db.execute('SELECT 1 FROM cs_schema_migrations WHERE version=31').fetchone(): return
    columns = {r['name'] for r in db.execute('PRAGMA table_info(cs_attachments)')}
    for name, definition in [('version', 'INTEGER NOT NULL DEFAULT 1'),
                              ('active', 'INTEGER NOT NULL DEFAULT 1'),
                              ('superseded_by', "TEXT NOT NULL DEFAULT ''")]:
        if name not in columns: db.execute('ALTER TABLE cs_attachments ADD COLUMN '+name+' '+definition)
    db.execute('CREATE INDEX IF NOT EXISTS cs_attachment_version ON cs_attachments(owner,workspace,session,name,version DESC)')
    db.execute('INSERT INTO cs_schema_migrations VALUES (31,?)', (now(),))


def migrate_v32(db):
    """Additive only: existing owners, sources and repository roots remain untouched."""
    if db.execute('SELECT 1 FROM cs_schema_migrations WHERE version=32').fetchone(): return
    additions={'cs_accounts': [('last_login_at','INTEGER')],
               'cs_repositories': [('source_kind',"TEXT NOT NULL DEFAULT 'local'"),('source_url',"TEXT NOT NULL DEFAULT ''"),('source_ref',"TEXT NOT NULL DEFAULT ''"),('deleted','INTEGER NOT NULL DEFAULT 0')],
               'cs_admin_audit': [('result',"TEXT NOT NULL DEFAULT 'success'"),('details',"TEXT NOT NULL DEFAULT '{}'")]}
    for table, fields in additions.items():
        columns={r['name'] for r in db.execute('PRAGMA table_info('+table+')')}
        for name,definition in fields:
            if name not in columns:db.execute('ALTER TABLE '+table+' ADD COLUMN '+name+' '+definition)
    for sql in '''
CREATE TABLE IF NOT EXISTS cs_account_quotas(account TEXT PRIMARY KEY, limits_json TEXT NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_quota_defaults(id INTEGER PRIMARY KEY CHECK(id=1), limits_json TEXT NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_usage_buckets(account TEXT NOT NULL, action TEXT NOT NULL, window INTEGER NOT NULL, used INTEGER NOT NULL, PRIMARY KEY(account,action,window));
CREATE TABLE IF NOT EXISTS cs_remote_snapshots(job_id TEXT PRIMARY KEY, repository TEXT NOT NULL, commit_sha TEXT NOT NULL, root TEXT NOT NULL, manifest TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS cs_account_role_status ON cs_accounts(role,status,created_at);
CREATE INDEX IF NOT EXISTS cs_repository_owner ON cs_repositories(owner,workspace,deleted);
CREATE INDEX IF NOT EXISTS cs_audit_recent ON cs_admin_audit(created_at DESC,id);
CREATE INDEX IF NOT EXISTS cs_chat_owner_status ON cs_chat(owner,status,created_at);
'''.split(';'):
        if sql.strip():db.execute(sql)
    db.execute('INSERT INTO cs_schema_migrations VALUES (32,?)',(now(),))
