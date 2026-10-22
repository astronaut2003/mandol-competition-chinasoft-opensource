"""Account management and bounded, persistent resource admission.

Management grants metadata access only; it never grants another owner's Scope.
All admission checks accept an existing transaction so concurrent writes cannot
race a quota. Logical bytes include retained history, not SQLite file overhead.
"""
import json
from .store import AppError, now, uid

ACTIVE="('pending','running','blocked','retry_wait')"
RANGES={'workspaces':(1,100),'conversations':(1,10000),'repositories':(1,50),
        'storage_bytes':(1024,1024*1024*1024),'active_jobs':(3,192),
        'chat_per_minute':(1,60),'retrieval_per_minute':(1,120),'uploads_per_minute':(1,120),
        'media_files':(1,10000),'attachment_files':(1,5000),'model_per_day':(1,10000),'repository_imports':(1,1),
        'web_per_day':(1,1000),'web_per_minute':(1,30)}


class Access:
    def __init__(self,store):self.store,self.settings=store,store.settings

    def account_for(self,db,owner):
        row=db.execute('SELECT a.* FROM cs_accounts a JOIN cs_account_owners o ON o.account=a.id WHERE o.owner=?',(owner,)).fetchone()
        return dict(row) if row else None

    def limits(self,db,account):
        s=self.settings
        result=dict(workspaces=s.quota_workspaces,conversations=s.quota_conversations,repositories=s.quota_repositories,
                    storage_bytes=s.quota_storage_bytes,active_jobs=min(48,s.account_job_quota),
                    chat_per_minute=s.quota_chat_per_minute,retrieval_per_minute=30,uploads_per_minute=20,media_files=1000,attachment_files=500,model_per_day=s.quota_model_per_day,repository_imports=1)
        result.update(web_per_day=100,web_per_minute=10)
        if account and account['role']=='admin':result.update(workspaces=100,conversations=10000,repositories=50,storage_bytes=1024**3,active_jobs=s.account_job_quota,model_per_day=5000)
        row=db.execute('SELECT limits_json FROM cs_quota_defaults WHERE id=1').fetchone()
        if row:result.update(json.loads(row['limits_json']))
        if account:
            row=db.execute('SELECT limits_json FROM cs_account_quotas WHERE account=?',(account['id'],)).fetchone()
            if row:result.update(json.loads(row['limits_json']))
        return result

    def usage(self,db,account,include_storage=True):
        owners=[r['owner'] for r in db.execute('SELECT owner FROM cs_account_owners WHERE account=?',(account,))]
        if not owners:owners=[account]
        marks=','.join('?' for _ in owners)
        def count(table,condition='1=1'):
            return db.execute(f'SELECT COUNT(*) FROM {table} WHERE owner IN ({marks}) AND {condition}',owners).fetchone()[0]
        def size(table,column):
            return db.execute(f'SELECT COALESCE(SUM({column}),0) FROM {table} WHERE owner IN ({marks})',owners).fetchone()[0]
        result={'workspaces':count('cs_workspaces'),'conversations':count('cs_sessions','deleted=0'),
                'repositories':count('cs_repositories','deleted=0'),
                'active_jobs':count('cs_jobs','status IN '+ACTIVE),
                'repository_imports':count('cs_jobs',"kind='repository' AND status IN "+ACTIVE),
                'media_files':count('cs_media'),'attachment_files':count('cs_attachments'),
                'chat_per_minute':self.used(db,account,'chat_per_minute',60000),
                'retrieval_per_minute':self.used(db,account,'retrieval_per_minute',60000),
                'uploads_per_minute':self.used(db,account,'uploads_per_minute',60000),
                'web_per_day':self.used(db,account,'web_per_day',86400000),
                'web_per_minute':self.used(db,account,'web_per_minute',60000),
                'model_per_day':self.used(db,account,'model_per_day',86400000)}
        if include_storage:
            repo_bytes=db.execute(f'SELECT COALESCE(SUM(f.bytes),0) FROM cs_repository_files f JOIN cs_repositories r ON r.id=f.repository WHERE r.owner IN ({marks}) AND f.status=\'indexed\'',owners).fetchone()[0]
            checkpoints=db.execute(f'SELECT COALESCE(SUM(length(CAST(s.snapshot||s.statistics AS BLOB))),0) FROM cs_repository_scans s JOIN cs_repositories r ON r.id=s.repository WHERE r.owner IN ({marks})',owners).fetchone()[0]
            remote=db.execute(f'SELECT COALESCE(SUM(length(CAST(s.manifest AS BLOB))),0) FROM cs_remote_snapshots s JOIN cs_repositories r ON r.id=s.repository WHERE r.owner IN ({marks})',owners).fetchone()[0]
            cache=db.execute(f'SELECT COALESCE(SUM(length(CAST(COALESCE(c.response,\'\') AS BLOB))),0) FROM cs_calls c JOIN cs_jobs j ON j.id=c.job_id WHERE j.owner IN ({marks})',owners).fetchone()[0]
            frozen=db.execute(f"SELECT COALESCE(SUM(bytes),0) FROM (SELECT s.root,MAX(COALESCE(json_extract(s.manifest,'$.total_bytes'),0)) bytes FROM cs_remote_snapshots s JOIN cs_repositories r ON r.id=s.repository WHERE r.owner IN ({marks}) GROUP BY s.root)",owners).fetchone()[0]
            result['storage_bytes']=size('cs_media','bytes')+size('cs_attachments','bytes')+size('cs_ingestions','length(CAST(payload AS BLOB))')+size('cs_chat','length(CAST(question||answer||route AS BLOB))')+size('cs_jobs','length(CAST(payload||progress AS BLOB))')+repo_bytes+checkpoints+remote+cache+frozen
        return result

    def used(self,db,account,action,period):
        r=db.execute('SELECT used FROM cs_usage_buckets WHERE account=? AND action=? AND window=?',(account,action,now()//period)).fetchone()
        return r['used'] if r else 0

    def check(self,db,owner,resource,amount=1):
        account=self.account_for(db,owner)
        if not account:return # Kept for internal legacy scopes and explicit local migration.
        if account['status']!='active':raise AppError('账号未激活或已禁用',403)
        if self.usage(db,account['id'],include_storage=resource=='storage_bytes')[resource]+amount>self.limits(db,account)[resource]:
            raise AppError('已达到账号资源配额：'+resource,429,code='quota_exceeded')

    def consume(self,db,account,action,period):
        row=db.execute('SELECT * FROM cs_accounts WHERE id=?',(account,)).fetchone()
        if not row or row['status']!='active':raise AppError('账号不可用',403)
        if self.used(db,account,action,period)>=self.limits(db,dict(row))[action]:
            raise AppError('已达到账号调用限额：'+action,429,code='quota_exceeded')
        db.execute('DELETE FROM cs_usage_buckets WHERE window<? AND action=?',(now()//period-2,action))
        db.execute('INSERT INTO cs_usage_buckets VALUES (?,?,?,1) ON CONFLICT(account,action,window) DO UPDATE SET used=used+1',(account,action,now()//period))
        if action=='model_per_day':
            if self.used(db,'__platform__',action,period)>=self.settings.model_requests_per_day:
                raise AppError('平台今日模型调用预算已用尽，请稍后再试',429,code='platform_model_quota')
            db.execute('INSERT INTO cs_usage_buckets VALUES (?,?,?,1) ON CONFLICT(account,action,window) DO UPDATE SET used=used+1',('__platform__',action,now()//period))

    def summary(self,account):
        with self.store.state.transaction() as db:
            row=db.execute('SELECT * FROM cs_accounts WHERE id=?',(account,)).fetchone()
            if not row:raise AppError('账号不存在',404)
            override=db.execute('SELECT limits_json FROM cs_account_quotas WHERE account=?',(account,)).fetchone()
            return {'limits':self.limits(db,dict(row)),'usage':self.usage(db,account),
                    'overrides':json.loads(override['limits_json']) if override else {},'storage_measurement':'logical retained input/media/source/checkpoint/job/generation-cache bytes; excludes vector/SQLite overhead'}

    @staticmethod
    def audit(db,actor,action,target,details=None,result='success'):
        db.execute('INSERT INTO cs_admin_audit(id,account,action,subject,created_at,result,details) VALUES (?,?,?,?,?,?,?)',
                   (uid('audit_'),actor,action,target,now(),result,json.dumps(details or {},ensure_ascii=False)))

    @staticmethod
    def require_admin(db,actor):
        if not db.execute("SELECT 1 FROM cs_accounts WHERE id=? AND role='admin' AND status='active'",(actor,)).fetchone():raise AppError('管理员权限已失效',403,code='admin_required')

    def users(self,query='',role=None,status=None,offset=0,limit=20):
        where=['1=1'];args=[]
        if query:where.append('(instr(username,?)>0 OR instr(display_name,?)>0)');args.extend([query,query])
        for key,value in [('role',role),('status',status)]:
            if value:where.append(key+'=?');args.append(value)
        with self.store.state.transaction() as db:
            total=db.execute('SELECT COUNT(*) FROM cs_accounts WHERE '+' AND '.join(where),args).fetchone()[0]
            rows=db.execute('SELECT id,username,display_name,role,status,created_at,last_login_at FROM cs_accounts WHERE '+' AND '.join(where)+' ORDER BY created_at DESC,id LIMIT ? OFFSET ?',(*args,limit,offset)).fetchall()
            return {'items':[dict(r,usage=self.usage(db,r['id'],include_storage=False)) for r in rows],'total':total,'offset':offset,'limit':limit}

    def change(self,actor,target,*,status=None,role=None,revoke=False):
        try:
            with self.store.state.transaction() as db:
                self.require_admin(db,actor)
                row=db.execute('SELECT * FROM cs_accounts WHERE id=?',(target,)).fetchone()
                if not row:raise AppError('账号不存在',404)
                if row['role']=='admin' and row['status']=='active' and (role=='user' or status in ('disabled','pending')):
                    if db.execute("SELECT COUNT(*) FROM cs_accounts WHERE role='admin' AND status='active'").fetchone()[0]<=1:
                        raise AppError('不能停用或降级最后一位有效管理员',409,code='last_admin')
                if status:
                    db.execute('UPDATE cs_accounts SET status=? WHERE id=?',(status,target));self.audit(db,actor,'account_status',target,{'from':row['status'],'to':status})
                if role:
                    db.execute('UPDATE cs_accounts SET role=?,local_repository=CASE WHEN ?=\'user\' THEN 0 ELSE local_repository END WHERE id=?',(role,role,target));self.audit(db,actor,'account_role',target,{'from':row['role'],'to':role})
                if revoke or status or role:
                    db.execute('UPDATE cs_auth_sessions SET revoked=1 WHERE account=?',(target,));self.audit(db,actor,'sessions_revoke',target)
        except AppError as exc:
            with self.store.state.transaction() as db:self.audit(db,actor,'account_change',target,{'reason':exc.code},'denied')
            raise
        return {'id':target,'updated':True,'sessions_revoked':True}

    def set_quotas(self,actor,target,limits):
        if limits is not None:
            for key,value in limits.items():
                if key not in RANGES or type(value) is not int or not RANGES[key][0]<=value<=RANGES[key][1]:
                    raise AppError('配额字段或范围无效：'+key,422)
        with self.store.state.transaction() as db:
            self.require_admin(db,actor)
            if target=='platform':
                db.execute('INSERT INTO cs_quota_defaults VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET limits_json=excluded.limits_json,updated_at=excluded.updated_at',(json.dumps(limits or {}),now()))
            else:
                if not db.execute('SELECT 1 FROM cs_accounts WHERE id=?',(target,)).fetchone():raise AppError('账号不存在',404)
                if limits is None:db.execute('DELETE FROM cs_account_quotas WHERE account=?',(target,))
                else:db.execute('INSERT INTO cs_account_quotas VALUES (?,?,?) ON CONFLICT(account) DO UPDATE SET limits_json=excluded.limits_json,updated_at=excluded.updated_at',(target,json.dumps(limits),now()))
            self.audit(db,actor,'quota_change',target,{'limits':limits})
        return {'updated':True}
