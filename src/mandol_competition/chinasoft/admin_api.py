"""Admin control plane: metadata, quotas, state changes; never private content."""
import json
from typing import Literal
from fastapi import Request, Query
from .schemas import Registration,AccountChange,QuotaUpdate
from .store import AppError
from .auth import HASHER,HASH_LIMIT
from argon2.exceptions import VerificationError,InvalidHashError


def install(app,p,principal):
    def admin(request):
        account=principal(request)
        if account['role']!='admin':raise AppError('此操作需要管理员权限',403,code='admin_required')
        return account

    @app.get('/api/account/resources')
    def own_resources(request:Request):return p(request).store.access.summary(principal(request)['id'])

    @app.get('/api/admin/users')
    def users(request:Request,q:str=Query(default='',max_length=64),role:Literal['admin','user']|None=None,status:Literal['active','disabled','pending']|None=None,
              offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=20,ge=1,le=50)):
        admin(request);return p(request).store.access.users(q,role,status,offset,limit)

    @app.get('/api/admin/users/{key}/resources')
    def resources(key:str,request:Request):admin(request);return p(request).store.access.summary(key)

    @app.post('/api/admin/users',status_code=201)
    def create(data:Registration,request:Request):
        actor=admin(request);p(request).accounts.limit('admin-create:'+actor['id'],20)
        return p(request).accounts.create(data.username,data.password,data.display_name,actor=actor['id'],approved=True)

    @app.patch('/api/admin/users/{key}')
    def change(key:str,data:AccountChange,request:Request):
        actor=admin(request)
        if not data.confirm:raise AppError('请明确确认账号管理操作',409)
        if data.role:
            p(request).accounts.limit('admin-role:'+actor['id'],10)
            try:
                with HASH_LIMIT:valid=HASHER.verify(actor['password_hash'],data.current_password)
            except (VerificationError,InvalidHashError):valid=False
            if not valid:
                with p(request).state.transaction() as db:p(request).store.access.audit(db,actor['id'],'account_role',key,{'reason':'reauthentication_required'},'denied')
                raise AppError('角色调整需要验证当前管理员密码',403)
        if not (data.status or data.role or data.revoke_sessions):raise AppError('没有选择管理操作')
        return p(request).store.access.change(actor['id'],key,status=data.status,role=data.role,revoke=data.revoke_sessions)

    @app.put('/api/admin/users/{key}/quotas')
    def quotas(key:str,data:QuotaUpdate,request:Request):
        actor=admin(request);return p(request).store.access.set_quotas(actor['id'],key,data.limits)

    @app.put('/api/admin/default-quotas')
    def defaults(data:QuotaUpdate,request:Request):
        actor=admin(request);return p(request).store.access.set_quotas(actor['id'],'platform',data.limits)

    @app.get('/api/admin/overview')
    def overview(request:Request):
        admin(request);store=p(request).store
        return {'accounts':store.rows('SELECT role,status,COUNT(*) count FROM cs_accounts GROUP BY role,status'),
                'jobs':store.rows('SELECT kind,status,COUNT(*) count FROM cs_jobs GROUP BY kind,status'),
                'max_accounts':store.settings.max_accounts,'registration_enabled':store.settings.registration_enabled}

    @app.get('/api/admin/audit')
    def audit(request:Request,offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=30,ge=1,le=50)):
        admin(request);store=p(request).store
        rows=store.rows('SELECT id,account,action,subject,created_at,result,details FROM cs_admin_audit ORDER BY created_at DESC,id LIMIT ? OFFSET ?',(limit,offset))
        for r in rows:r['details']=json.loads(r['details'])
        return {'items':rows,'total':store.rows('SELECT COUNT(*) n FROM cs_admin_audit')[0]['n'],'offset':offset,'limit':limit}
