"""Small ingestion/outbox adapter inside the canonical SQLite owner.

Every successful acceptance is one FULL/WAL commit. Model calls and file IO are
outside transactions; leases are owned by the single state-store process lock.
"""
from pathlib import Path
import json, time, uuid, secrets, hashlib, hmac, os
from mandol_substrate.state import Scope, Conflict
from mandol_substrate.state.models import digest, canonical_json


class AppError(Exception):
    def __init__(self, message, status=400, *, code='invalid_request'):
        self.message, self.status, self.code = message, status, code
        self.retryable = code == 'queue_full'


def now():
    return int(time.time() * 1000)


def uid(prefix):
    return prefix + uuid.uuid4().hex


DDL = '''
CREATE TABLE IF NOT EXISTS cs_workspaces(id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, demo INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_sessions(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, mode TEXT NOT NULL, title TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_scopes(scope TEXT PRIMARY KEY, namespace TEXT NOT NULL, user_id TEXT NOT NULL, owner TEXT NOT NULL, workspace TEXT NOT NULL, mode TEXT NOT NULL, kind TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cs_ingestions(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, session TEXT NOT NULL, mode TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL, receipt TEXT NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL, saved_at INTEGER, UNIQUE(owner,workspace,request_id));
CREATE TABLE IF NOT EXISTS cs_jobs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, ingestion TEXT, kind TEXT NOT NULL, priority INTEGER NOT NULL, status TEXT NOT NULL, dependency TEXT, payload TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '{}', attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, available_at INTEGER NOT NULL, error_code TEXT, error TEXT, unknown_remote INTEGER NOT NULL DEFAULT 0, lease TEXT);
CREATE INDEX IF NOT EXISTS cs_jobs_queue ON cs_jobs(kind,status,available_at,priority,created_at);
CREATE INDEX IF NOT EXISTS cs_jobs_owner ON cs_jobs(owner,workspace,created_at);
CREATE TABLE IF NOT EXISTS cs_media(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, session TEXT NOT NULL, sha256 TEXT NOT NULL, path TEXT NOT NULL, thumb TEXT NOT NULL, mime TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, bytes INTEGER NOT NULL, created_at INTEGER NOT NULL, UNIQUE(owner,workspace,session,sha256));
CREATE TABLE IF NOT EXISTS cs_chat(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, session TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL, question TEXT NOT NULL, media TEXT NOT NULL, answer TEXT NOT NULL DEFAULT '', citations TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL, model TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, error TEXT, UNIQUE(owner,workspace,request_id));
CREATE INDEX IF NOT EXISTS cs_chat_session ON cs_chat(owner,session,created_at);
CREATE TABLE IF NOT EXISTS cs_calls(id TEXT PRIMARY KEY, job_id TEXT, state TEXT NOT NULL, response TEXT, error_code TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_metrics(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, milliseconds REAL NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS cs_memory_clears(id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL, request_id TEXT NOT NULL, status TEXT NOT NULL, manifest TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}', created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, UNIQUE(owner,workspace,request_id));
CREATE INDEX IF NOT EXISTS cs_memory_clear_workspace ON cs_memory_clears(owner,workspace,status);
'''


class AppStore:
    def __init__(self, state, settings):
        self.state, self.settings = state, settings
        with state.transaction() as db:
            for stmt in DDL.split(';'):
                if stmt.strip(): db.execute(stmt)
        from .migrations import migrate
        migrate(state)
        from .access import Access
        self.access=Access(self)
        key = settings.data_dir / 'auth.key'
        if not key.exists():
            key.parent.mkdir(parents=True, exist_ok=True)
            try:
                with os.fdopen(os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as f:
                    f.write(secrets.token_bytes(32)); f.flush(); os.fsync(f.fileno())
                descriptor=os.open(key.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
            except FileExistsError: pass
        self.auth_key = key.read_bytes()

    def rows(self, sql, params=()):
        return self.state._rows(sql, params)

    def token(self, owner):
        return owner + '.' + hmac.new(self.auth_key, owner.encode(), hashlib.sha256).hexdigest()

    def owner(self, token):
        if not token or len(token) != 97: return None
        candidate = token.split('.')[0]
        if len(candidate) != 32 or any(c not in '0123456789abcdef' for c in candidate): return None
        return candidate if hmac.compare_digest(token, self.token(candidate)) else None

    def workspace(self, owner, workspace):
        rows = self.rows('SELECT * FROM cs_workspaces WHERE id=? AND owner=?', (workspace, owner))
        if not rows: raise AppError('工作区不存在或无权访问', 404, code='not_found')
        if rows[0]['demo'] == -1: raise AppError('演示工作区正在重置',409)
        return rows[0]

    def assert_memory_writable(self, db, owner, workspace):
        # Transactional admission fence; a failed clear must resume its frozen
        # deletion plan before new work can enter this workspace.
        if db.execute("SELECT 1 FROM cs_memory_clears WHERE owner=? AND workspace=? AND status<>'completed' LIMIT 1", (owner, workspace)).fetchone():
            raise AppError('工作区记忆正在清除或等待恢复，请在记忆探索中重试清除后再写入',409,code='memory_clear_pending')

    def session(self, owner, workspace, session, mode=None):
        self.workspace(owner, workspace)
        rows = self.rows('SELECT * FROM cs_sessions WHERE id=? AND owner=? AND workspace=? AND deleted=0', (session, owner, workspace))
        if not rows: raise AppError('会话不存在或无权访问', 404, code='not_found')
        if mode and not rows[0]['unified'] and rows[0]['mode'] != mode: raise AppError('记忆类型与会话不一致，请创建对应会话')
        return rows[0]

    def create_workspace(self, owner, name, demo=False):
        key = uid('ws_')
        with self.state.transaction() as db:
            if not self.access.account_for(db,owner) and db.execute('SELECT COUNT(*) FROM cs_workspaces WHERE owner=?', (owner,)).fetchone()[0] >= 30:
                raise AppError('每个用户最多 30 个工作区', 429)
            self.access.check(db,owner,'workspaces')
            db.execute('INSERT INTO cs_workspaces VALUES (?,?,?,?,?)', (key, owner, name, int(demo), now()))
        return self.workspace(owner, key)

    def create_session(self, owner, workspace, mode, title, repository_id=None, unified=False):
        self.workspace(owner, workspace); key = uid('ses_')
        if repository_id:
            if mode!='code' and not unified: raise AppError('Repository 仅属于 Code 场景')
            self.repository(owner,workspace,repository_id)
        with self.state.transaction() as db:
            self.access.check(db,owner,'conversations')
            db.execute('INSERT INTO cs_sessions(id,owner,workspace,mode,title,created_at,repository,updated_at,unified) VALUES (?,?,?,?,?,?,?,?,?)', (key, owner, workspace, mode, title, now(),repository_id or '',now(),int(unified)))
        return self.session(owner, workspace, key)

    def scope(self, owner, workspace, mode, repository_id=None):
        self.workspace(owner,workspace)
        if repository_id:
            if mode!='code': raise AppError('Repository 仅属于 Code 场景')
            self.repository(owner,workspace,repository_id)
        scope = Scope('chinasoft.raw.v1', f'{owner}.{workspace}.{mode}'+('.'+repository_id if repository_id else ''))
        self.register_scope(scope, owner, workspace, mode, 'raw',repository_id)
        self.state.ensure_scope(scope)
        return scope

    def register_scope(self, scope, owner, workspace, mode, kind, repository_id=None):
        self.workspace(owner,workspace)
        expected = f'{owner}.{workspace}.{mode}'+('.'+repository_id if repository_id else '')
        if scope.user_id != expected: raise AppError('Scope 作用域不匹配',403,code='scope_rejected')
        with self.state.transaction() as db:
            db.execute('INSERT OR IGNORE INTO cs_scopes(scope,namespace,user_id,owner,workspace,mode,kind,repository) VALUES (?,?,?,?,?,?,?,?)',
                       (scope.key, scope.namespace, scope.user_id, owner, workspace, mode, kind,repository_id or ''))

    def scopes(self, owner, workspace, mode=None, repository_id=None):
        self.workspace(owner, workspace)
        rows = self.rows('SELECT * FROM cs_scopes WHERE owner=? AND workspace=?' + (' AND mode=?' if mode else ''),
                         (owner, workspace, mode) if mode else (owner, workspace))
        return [(Scope(r['namespace'], r['user_id']), r) for r in rows if repository_id is None or r['repository']==repository_id]

    def repository(self,owner,workspace,key):
        self.workspace(owner,workspace)
        rows=self.rows('SELECT * FROM cs_repositories WHERE id=? AND owner=? AND workspace=? AND deleted=0',(key,owner,workspace))
        if not rows: raise AppError('仓库不存在或无权访问',404,code='not_found')
        return rows[0]

    def media(self, owner, media_id, workspace=None, session=None):
        rows = self.rows('SELECT * FROM cs_media WHERE id=? AND owner=?', (media_id, owner))
        if not rows or (workspace and rows[0]['workspace'] != workspace) or (session and rows[0]['session'] != session):
            raise AppError('图片不存在或不属于当前会话', 404, code='not_found')
        return rows[0]

    def attachment(self,owner,key,workspace,session=None):
        self.workspace(owner,workspace)
        rows=self.rows('SELECT * FROM cs_attachments WHERE id=? AND owner=? AND workspace=?',(key,owner,workspace))
        if not rows or (session and rows[0]['session']!=session):raise AppError('附件不存在或无权访问',404)
        self.session(owner,workspace,rows[0]['session'])
        return rows[0]

    def _job(self, db, key, owner, workspace, ingestion, kind, payload, dependency=None):
        self.assert_memory_writable(db,owner,workspace)
        t = now()
        payload=dict(payload)
        if ingestion:
            accepted=db.execute('SELECT mode,payload FROM cs_ingestions WHERE id=? AND owner=? AND workspace=?',(ingestion,owner,workspace)).fetchone()
            if not accepted:raise AppError('任务来源不属于当前工作区',403,code='scope_rejected')
            payload.update(memory_family={'textual':'conversation','code':'code','multimodal':'multimodal'}[accepted['mode']],repository_id=json.loads(accepted['payload']).get('repository_id') or '')
            original=json.loads(accepted['payload'])
            if kind=='index' and original.get('unified_raw'):
                payload['associated_repository_id']=payload['repository_id']
                payload['repository_id']='' # Unified raw truth is conversation-scoped.
        if payload.get('route'):
            payload['memory_family']={'conversation':'conversation','engineering':'code','source_code':'code','visual':'multimodal'}.get(payload['route'],'conversation')
        if kind=='repository':payload['memory_family']='code'
        db.execute('INSERT OR IGNORE INTO cs_jobs(id,owner,workspace,ingestion,kind,priority,status,dependency,payload,max_attempts,created_at,updated_at,available_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (key, owner, workspace, ingestion, kind, {'ingest':0,'repository':5,'index':10,'derive':20}[kind],
                    'blocked' if dependency else 'pending', dependency, canonical_json(payload), self.settings.worker_max_attempts, t, t, t))

    def accept(self, owner, request):
        data = request.model_dump() if hasattr(request, 'model_dump') else json.loads(canonical_json(request))
        workspace, session, mode = data['workspace_id'], data['session_id'], data['mode']
        selected=self.session(owner, workspace, session, mode)
        replay=self.rows('SELECT payload,receipt FROM cs_ingestions WHERE owner=? AND workspace=? AND request_id=?',(owner,workspace,data['request_id']))
        if replay:
            original=json.loads(replay[0]['payload'])
            def signature(value):
                messages=[{k:m.get(k,[] if k=='media_ids' else 'user' if k=='role' else '') for k in ('role','content','media_ids')} for m in value.get('messages',[]) if m.get('role')!='attachment']
                revision=value.get('revision')
                if revision:revision={k:revision.get(k,False if k=='delete' else None) for k in ('uid','expected_version','content','delete')}
                return [value['session_id'],messages,value.get('attachment_ids',[]),revision]
            supplied_repository=data.get('repository_id')
            if signature(original)!=signature(data) or (supplied_repository is not None and supplied_repository!=(original.get('repository_id') or '')):
                raise Conflict('相同 request_id 对应不同内容')
            return json.loads(replay[0]['receipt'])
        if selected['unified'] and not data.get('revision'): mode=data['mode']='textual'
        if selected['unified']:data['unified_raw']=not bool(data.get('revision'))
        repository=data.get('repository_id')
        if repository is not None and repository != selected['repository']: raise AppError('仓库与会话归属不一致',409)
        data['repository_id']=(data['revision'].get('repository_id',selected['repository']) if data.get('revision') else selected['repository']) or None
        for message in data['messages']:
            for mid in message.get('media_ids', []): self.media(owner, mid, workspace, session)
        raw_repository=data['revision'].get('repository_id',selected['repository']) if data.get('revision') else '' if data.get('unified_raw') else selected['repository']
        data.setdefault('attachment_ids',[])
        if len(data['attachment_ids'])>self.settings.attachment_max_files:raise AppError('附件数量超过配置限制',413)
        for attachment_id in data['attachment_ids']:
            a=self.attachment(owner,attachment_id,workspace,session)
            if not a['active']:raise AppError('附件已删除或被新版本替换，请重新选择',409)
            data['messages'].append({'role':'attachment','content':a['text'],'media_ids':[], 'attachment_id':a['id'],'attachment_kind':a['kind'],'attachment_name':a['name'],'attachment_version':a['version'],'attachment_hash':a['hash'],'attachment_parser':a['parser']})
        if sum(len(m['content']) for m in data['messages'])>64000:raise AppError('消息与附件合计超过 64,000 字符',413)
        from .routing import routes, ROUTING_RECIPE
        if selected['unified'] and not data.get('revision'):
            for message in data['messages']:message['routes']=routes(message)
            data['routing_recipe']=ROUTING_RECIPE
        scope = self.scope(owner, workspace, mode,raw_repository); fingerprint = digest(data)
        key = 'ing_' + digest([owner, workspace, data['request_id']])
        source_ids = [data['revision']['uid']] if data.get('revision') else [scope.memory_uid(session, data['request_id'], i) for i in range(len(data['messages']))]
        receipt = {'ingestion_id': key, 'accepted': True, 'durable': True, 'status': 'pending',
                   'memory_ids': source_ids, 'jobs': [key + ':' + k for k in ('ingest','index','derive')]}
        route_names=sorted({r for m in data['messages'] for r in m.get('routes',[])}) if selected['unified'] else []
        if data.get('revision') and selected['unified']:route_names=['conversation','engineering','visual','source_code']
        if selected['unified']:
            receipt['routes']=route_names
            receipt['jobs']=[key+':ingest',key+':index']+[key+':derive:'+r for r in route_names if r!='document']
        with self.state.transaction() as db:
            if not db.execute('SELECT 1 FROM cs_workspaces WHERE id=? AND owner=? AND demo<>-1',(workspace,owner)).fetchone():
                raise AppError('工作区已删除或正在重置',409)
            old = db.execute('SELECT digest,receipt,payload FROM cs_ingestions WHERE owner=? AND workspace=? AND request_id=?',
                             (owner, workspace, data['request_id'])).fetchone()
            if old:
                previous=json.loads(old['payload']);previous.setdefault('repository_id',None);previous.setdefault('attachment_ids',[])
                if selected['unified']:previous.setdefault('unified_raw',data.get('unified_raw'))
                if old['digest'] != fingerprint and digest(previous)!=fingerprint: raise Conflict('相同 request_id 对应不同内容')
                return json.loads(old['receipt'])
            pending = db.execute("SELECT COUNT(*) FROM cs_jobs WHERE status IN ('pending','blocked','running','retry_wait')").fetchone()[0]
            own_pending=db.execute("SELECT COUNT(*) FROM cs_jobs WHERE owner=? AND status IN ('pending','blocked','running','retry_wait')",(owner,)).fetchone()[0]
            self.access.check(db,owner,'active_jobs',len(receipt['jobs']))
            self.access.check(db,owner,'storage_bytes',len(canonical_json(data).encode()))
            if pending + len(receipt['jobs']) > self.settings.queue_capacity or own_pending+len(receipt['jobs'])>self.settings.account_job_quota:
                raise AppError('后台队列已满，请稍后重试', 503, code='queue_full')
            db.execute('INSERT INTO cs_ingestions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (key, owner, workspace, session, mode, data['request_id'], fingerprint, canonical_json(data), canonical_json(receipt), 'pending', now(), None))
            self._job(db, key + ':ingest', owner, workspace, key, 'ingest', {'ingestion': key})
            self._job(db,key+':index',owner,workspace,key,'index',{'ingestion':key},key+':ingest')
            for route in ([r for r in route_names if r!='document'] if selected['unified'] else [None]):
                self._job(db,key+':derive'+(':'+route if route else ''),owner,workspace,key,'derive',{'ingestion':key,**({'route':route} if route else {})},key+':ingest')
            first=next((m['content'] for m in data['messages'] if m['role']=='user' and m['content'].strip()),'图片或附件')
            title=' '.join(first.split())[:36]
            db.execute("UPDATE cs_sessions SET title=CASE WHEN manual_title=0 AND title IN ('新会话','新的对话') THEN ? ELSE title END,updated_at=? WHERE id=? AND owner=? AND deleted=0",(title,now(),session,owner))
        return receipt

    def enqueue(self, owner, workspace, kind, payload, identity):
        self.workspace(owner,workspace)
        key = 'job_' + digest([owner, workspace, kind, identity])
        with self.state.transaction() as db:
            if db.execute('SELECT 1 FROM cs_jobs WHERE id=?',(key,)).fetchone(): return key
            self.access.check(db,owner,'active_jobs')
            pending = db.execute("SELECT COUNT(*) FROM cs_jobs WHERE status IN ('pending','blocked','running','retry_wait')").fetchone()[0]
            if pending >= self.settings.queue_capacity: raise AppError('后台队列已满，请稍后重试',503,code='queue_full')
            self._job(db, key, owner, workspace, payload.get('ingestion'), kind, payload)
        return key

    def ingestion(self, owner, key):
        rows = self.rows('SELECT * FROM cs_ingestions WHERE id=? AND owner=?', (key, owner))
        if not rows: raise AppError('提交记录不存在', 404)
        row = rows[0]; row['payload'] = json.loads(row['payload']); row['receipt'] = json.loads(row['receipt'])
        return row

    def claim(self, kind, lease):
        with self.state.transaction() as db:
            if kind == 'derive' and db.execute("SELECT COUNT(*) FROM cs_jobs WHERE kind='ingest' AND status IN ('pending','running')").fetchone()[0] > 12:
                return None
            row = db.execute("SELECT j.* FROM cs_jobs j WHERE j.kind=? AND j.status IN ('pending','retry_wait') AND j.available_at<=? AND NOT EXISTS (SELECT 1 FROM cs_memory_clears c WHERE c.owner=j.owner AND c.workspace=j.workspace AND c.status<>'completed') AND NOT EXISTS (SELECT 1 FROM cs_account_owners o JOIN cs_accounts a ON a.id=o.account WHERE o.owner=j.owner AND a.status<>'active') ORDER BY j.priority,j.created_at LIMIT 1", (kind, now())).fetchone()
            if not row: return None
            db.execute("UPDATE cs_jobs SET status='running',lease=?,attempts=attempts+1,updated_at=? WHERE id=?", (lease, now(), row['id']))
        result = dict(row); result['attempts'] += 1; result['payload'] = json.loads(result['payload']); result['lease'] = lease
        return result

    def finish(self, job, progress):
        with self.state.transaction() as db:
            changed = db.execute("UPDATE cs_jobs SET status='completed',progress=?,updated_at=?,lease=NULL,error=NULL,error_code=NULL WHERE id=? AND lease=?",
                       (canonical_json(progress), now(), job['id'], job['lease'])).rowcount
            if changed != 1: return
            db.execute("UPDATE cs_jobs SET status='pending',dependency=NULL,available_at=? WHERE dependency=? AND status='blocked'", (now(), job['id']))

    def fail(self, job, error):
        unknown = bool(getattr(error, 'unknown', False))
        retryable = bool(getattr(error, 'retryable', False)) and not unknown
        status = 'retry_wait' if retryable and job['attempts'] < job['max_attempts'] else 'failed'
        code = getattr(error, 'code', type(error).__name__)
        message = getattr(error, 'message', '后台处理失败；请检查系统状态后重试')
        with self.state.transaction() as db:
            changed = db.execute('UPDATE cs_jobs SET status=?,error_code=?,error=?,unknown_remote=?,available_at=?,updated_at=?,lease=NULL WHERE id=? AND lease=?',
                       (status, code, message[:500], int(unknown), now() + min(60000, 2000 * 2 ** job['attempts']), now(), job['id'], job['lease'])).rowcount
            if changed != 1: return
            if status == 'failed':
                db.execute("UPDATE cs_jobs SET status='failed',error_code='dependency_failed',error='依赖任务失败' WHERE dependency=? AND status='blocked'", (job['id'],))

    def cancel_superseded(self,job):
        with self.state.transaction() as db:
            db.execute("UPDATE cs_jobs SET status='cancelled',lease=NULL,error_code='source_changed',error=NULL,progress=?,updated_at=? WHERE id=? AND lease=?",(canonical_json({'generated':0,'stage':'来源版本已更新；旧构建取消，由修订任务重新构建'}),now(),job['id'],job['lease']))

    def recover(self):
        with self.state.transaction() as db:
            db.execute("UPDATE cs_calls SET state='unknown',error_code='process_interrupted',updated_at=? WHERE state='running'", (now(),))
            db.execute("UPDATE cs_calls SET state='failed',error_code='not_sent',updated_at=? WHERE state='reserved'", (now(),))
            db.execute("UPDATE cs_jobs SET status=CASE WHEN attempts>=max_attempts THEN 'failed' ELSE 'retry_wait' END,available_at=?,lease=NULL,error_code='worker_restarted',error='进程中断后恢复',updated_at=? WHERE status='running'", (now(), now()))
            db.execute("UPDATE cs_chat SET status='interrupted',error='回答在服务重启时中断，可重新提问',updated_at=? WHERE status='processing'", (now(),))
            db.execute('UPDATE cs_workspaces SET demo=1 WHERE demo=-1')

    def jobs(self, owner, workspace, offset=0, limit=30):
        self.workspace(owner, workspace)
        rows = self.rows('SELECT * FROM cs_jobs WHERE owner=? AND workspace=? ORDER BY created_at DESC,id LIMIT ? OFFSET ?', (owner, workspace, limit, offset))
        for r in rows:
            payload=json.loads(r.pop('payload'));r['memory_family']=payload.get('memory_family');r['repository_id']=payload.get('repository_id','')
            r['task_type']= {'conversation':'conversation_derivation','code':'engineering_experience_derivation','multimodal':'visual_derivation'}.get(r['memory_family'],'derivation') if r['kind']=='derive' else {'ingest':'raw_ingestion','index':'retrieval_index','repository':'repository_scan_parse'}.get(r['kind'],r['kind'])
            r['progress'] = json.loads(r['progress']); r.pop('lease')
        total = self.rows('SELECT COUNT(*) n FROM cs_jobs WHERE owner=? AND workspace=?', (owner, workspace))[0]['n']
        return {'items':rows,'total':total,'offset':offset,'limit':limit}

    def retry(self, owner, job_id, allow_unknown=False, current_versions=None):
        rows = self.rows('SELECT * FROM cs_jobs WHERE id=? AND owner=?', (job_id, owner))
        if not rows: raise AppError('任务不存在', 404)
        job = rows[0]
        if job['status'] != 'failed': raise AppError('仅可重试失败任务', 409)
        if job['error_code'] == 'semantic_rejected':
            raise AppError('该结果已被来源审计拒绝。请修订原始记录或调整模型配方后重新构建',409)
        if job['unknown_remote'] and not allow_unknown:
            raise AppError('远程调用结果未知，重试可能重复计费；需明确确认', 409, code='unknown_remote')
        if job['attempts'] >= 6: raise AppError('人工重试已达上限，请修复配置后重新提交', 409)
        with self.state.transaction() as db:
            self.assert_memory_writable(db,owner,job['workspace'])
            # Recheck after acquiring the transaction: a concurrent workspace
            # clear may have cancelled this job after the initial read above.
            current_job=db.execute('SELECT status FROM cs_jobs WHERE id=? AND owner=?',(job_id,owner)).fetchone()
            if not current_job or current_job['status']!='failed':
                raise AppError('任务状态已变化，请刷新后重试',409)
            self.access.check(db,owner,'active_jobs',1+db.execute("SELECT COUNT(*) FROM cs_jobs WHERE dependency=? AND error_code='dependency_failed'",(job_id,)).fetchone()[0])
            ledger = db.execute('SELECT scope,checkpoint_json FROM request_preparations WHERE request_id=?',(job_id,)).fetchone()
            if ledger:
                checkpoint=json.loads(ledger['checkpoint_json']);code=checkpoint.get('code_experience')
                if code and code.get('state') in ('failed','inflight'):
                    versions=code.get('input_versions')
                    if not versions: raise AppError('旧任务缺少可安全复用的来源快照，请修订原始记录后重建',409)
                    if current_versions is not None and current_versions != versions:
                        raise AppError('来源窗口已变化，请使用当前版本的新构建任务',409)
                    current=db.execute('SELECT logical_uid,version,active FROM memories WHERE scope=? AND logical_uid IN ('+','.join('?' for _ in versions)+')',(ledger['scope'],*versions)).fetchall()
                    if {r['logical_uid']:r['version'] for r in current if r['active']} != versions:
                        raise AppError('来源版本已经变化，请使用新版本的构建任务',409)
                    checkpoint.pop('code_experience')
                    db.execute('UPDATE request_preparations SET checkpoint_json=? WHERE scope=? AND request_id=?',(canonical_json(checkpoint),ledger['scope'],job_id))
            if allow_unknown:
                db.execute("DELETE FROM cs_calls WHERE job_id=? AND state='unknown'", (job_id,))
            db.execute("UPDATE cs_jobs SET status='pending',max_attempts=MIN(6,attempts+?),error=NULL,error_code=NULL,unknown_remote=0,available_at=? WHERE id=?", (self.settings.worker_max_attempts, now(), job_id))
            db.execute("UPDATE cs_jobs SET status='blocked',error=NULL,error_code=NULL WHERE dependency=? AND error_code='dependency_failed'", (job_id,))
        return {'job_id':job_id,'status':'pending'}

    def metric(self, name, milliseconds):
        with self.state.transaction() as db:
            db.execute('INSERT INTO cs_metrics(name,milliseconds,created_at) VALUES (?,?,?)', (name, milliseconds, now()))
            db.execute('DELETE FROM cs_metrics WHERE id<(SELECT COALESCE(MAX(id),0)-2000 FROM cs_metrics)')
