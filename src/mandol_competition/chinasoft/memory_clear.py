"""Explicit workspace memory clearing through the existing publication protocol.

Conversation history, uploaded files and audit records are retained. No model
calls, physical database purge or collection-wide Qdrant deletion is performed.
"""
from dataclasses import asdict
import json
import logging
from threading import Lock

from mandol_substrate.state import Mutation, Scope
from mandol_substrate.state.models import canonical_json, digest
from .store import AppError, now

CONFIRMATION = '清除所有记忆'
logger = logging.getLogger(__name__)


class MemoryClearing:
    def __init__(self, platform):
        self.p = platform
        self.lock = Lock()

    def clear(self, owner, workspace, request_id):
        self.p.store.workspace(owner, workspace)
        if not self.lock.acquire(blocking=False):
            raise AppError('记忆清除请求正在处理，请稍后重试',409,code='memory_clear_busy')
        try:
            operation = self._prepare(owner, workspace, request_id)
            if operation['status'] == 'completed':
                return json.loads(operation['result'])
            return self._execute(operation)
        finally:
            self.lock.release()

    def _prepare(self, owner, workspace, request_id):
        with self.p.state.transaction() as db:
            previous = db.execute('SELECT * FROM cs_memory_clears WHERE owner=? AND workspace=? AND request_id=?',
                                  (owner,workspace,request_id)).fetchone()
            if previous:
                return dict(previous)
            # Resume the original snapshot even if a browser lost its request ID.
            pending = db.execute("SELECT * FROM cs_memory_clears WHERE owner=? AND workspace=? AND status<>'completed' LIMIT 1",
                                 (owner,workspace)).fetchone()
            if pending:
                return dict(pending)
            busy = db.execute("SELECT 1 FROM cs_jobs WHERE owner=? AND workspace=? AND status='running' LIMIT 1",(owner,workspace)).fetchone()
            chat = db.execute("SELECT 1 FROM cs_chat WHERE owner=? AND workspace=? AND status='processing' LIMIT 1",(owner,workspace)).fetchone()
            if busy or chat:
                raise AppError('当前工作区还有回答或后台任务正在运行，请等待完成后重试清除',409,code='workspace_busy')
            manifest = []
            counts = {'records':0,'basic':0,'higher':0,'canonical_records':0}
            for row in db.execute('SELECT * FROM cs_scopes WHERE owner=? AND workspace=? ORDER BY scope',(owner,workspace)).fetchall():
                frontier = db.execute('SELECT canonical_seq,published_seq FROM scopes WHERE scope=?',(row['scope'],)).fetchone()
                if not frontier:
                    continue
                if frontier['canonical_seq'] != frontier['published_seq']:
                    raise AppError('记忆发布尚未完成，请先恢复后台任务或重启服务后重试',409,code='publication_pending')
                records = db.execute("SELECT logical_uid,memory_level,role,memory_type FROM memories WHERE scope=? AND lifecycle<>'deleted' ORDER BY logical_uid",(row['scope'],)).fetchall()
                if not records:
                    continue
                manifest.append({'namespace':row['namespace'],'user_id':row['user_id'],'uids':[r['logical_uid'] for r in records]})
                counts['canonical_records'] += len(records)
                for r in records:
                    if r['memory_type'] in ('reference','canonical_reference'):
                        continue
                    counts['records'] += 1
                    counts['basic' if r['memory_level']=='L0_Observation' and r['role']!='derived' else 'higher'] += 1
            cancelled = db.execute("UPDATE cs_jobs SET status='cancelled',lease=NULL,error_code='memory_cleared',error='用户清除工作区记忆；旧任务不再重试',updated_at=? WHERE owner=? AND workspace=? AND status IN ('pending','blocked','retry_wait','failed')",(now(),owner,workspace)).rowcount
            key = 'clear_'+digest([owner,workspace,request_id])
            result = dict(counts,workspace_id=workspace,cancelled_jobs=cancelled,operation_id=key,
                          retained=['conversations','attachments','repository_settings','deletion_audit'])
            db.execute('INSERT INTO cs_memory_clears VALUES (?,?,?,?,?,?,?,?,?)',
                       (key,owner,workspace,request_id,'clearing',canonical_json(manifest),canonical_json(result),now(),now()))
            return dict(db.execute('SELECT * FROM cs_memory_clears WHERE id=?',(key,)).fetchone())

    def _execute(self, operation):
        manifest = json.loads(operation['manifest'])
        try:
            for item in manifest:
                scope = Scope(item['namespace'],item['user_id'])
                for start in range(0,len(item['uids']),200):
                    mutation = Mutation(delete_uids=tuple(item['uids'][start:start+200]))
                    # Freeze batches so crash-after-commit replay is safe. A
                    # replay cannot delete memories added after completion.
                    key = operation['id']+':'+digest(scope.key)+':'+str(start//200)
                    self.p.coordinator.apply(scope,key,digest(asdict(mutation)),mutation,{})
            result = json.loads(operation['result'])
            result.update(status='completed',projection_status='pending' if manifest else 'unchanged')
            with self.p.state.transaction() as db:
                repositories = [r['id'] for r in db.execute('SELECT id FROM cs_repositories WHERE owner=? AND workspace=? AND deleted=0',(operation['owner'],operation['workspace']))]
                for repo in repositories:
                    # Exact-lookup projections are rebuildable. Preserve file
                    # hashes/versions for a later explicit, versioned rescan.
                    db.execute("UPDATE cs_repository_files SET status='cleared',updated_at=? WHERE repository=? AND status='indexed'",(now(),repo))
                    db.execute('DELETE FROM cs_repository_symbols WHERE repository=?',(repo,))
                    db.execute('DELETE FROM cs_repository_links WHERE repository=?',(repo,))
                    db.execute("UPDATE cs_repositories SET status='cleared',statistics=?,error=NULL WHERE id=?",(canonical_json({'files':0,'symbols':0,'memory_cleared':True}),repo))
                db.execute("UPDATE cs_memory_clears SET status='completed',result=?,updated_at=? WHERE id=?",(canonical_json(result),now(),operation['id']))
                if manifest:
                    # Empty scopes are skipped by automatic backfill. Queue an
                    # explicit reconciliation for their vector tombstones.
                    self.p.store._job(db,operation['id']+':index',operation['owner'],operation['workspace'],None,'index',
                                      {'scopes':[{'namespace':i['namespace'],'user_id':i['user_id']} for i in manifest],
                                       'memory_clear_id':operation['id']})
            return result
        except Exception:
            with self.p.state.transaction() as db:
                db.execute("UPDATE cs_memory_clears SET status='interrupted',updated_at=? WHERE id=?",(now(),operation['id']))
            logger.exception('Workspace memory clear interrupted: %s',operation['id'])
            raise AppError('记忆清除尚未全部完成，写入已暂停；请重试清除以继续，不会重复清除新的记忆',503,code='memory_clear_interrupted')

    def recover(self):
        # After canonical publication recovery, before background workers start.
        with self.lock:
            for operation in self.p.store.rows("SELECT * FROM cs_memory_clears WHERE status<>'completed' ORDER BY created_at"):
                try:
                    self._execute(operation)
                except AppError:
                    logger.error('Memory clear remains fenced pending retry: %s',operation['id'])
