from contextlib import asynccontextmanager
from pathlib import Path
import asyncio, json, time, uuid, re, hashlib, os
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from mandol_substrate.state.models import Conflict, StateUnavailable, digest, canonical_json
from .config import Settings, ROOT
from .schemas import AddMemory, Chat, Workspace, Session, Revision, Retry, RepositoryRegistration, RepositoryScan, Credentials, Registration, SessionUpdate, RemoteRepositoryRegistration, ClearMemories
from .auth import COOKIE
import hmac
from .typed_memory import FAMILIES, MODES, TYPE_MAP
from .service import Platform
from .models import ModelError
from .store import AppError, now, uid


class BoundedRequestBody:
    """Enforce limits while ASGI receives data, including chunked uploads."""
    def __init__(self,app,max_bytes):self.app,self.max_bytes=app,max_bytes

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('method') not in ('POST','PUT','PATCH','DELETE'):
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]));limit=self.max_bytes
        if b'application/json' in headers.get(b'content-type',b''):limit=min(limit,384*1024)
        if scope.get('path') in ('/api/auth/login','/api/auth/register'):limit=min(limit,16*1024)
        received=0
        async def bounded_receive():
            nonlocal received
            message=await receive()
            if message['type']=='http.request':
                received+=len(message.get('body',b''))
                if received>limit:raise HTTPException(413,'请求内容超过接收上限')
            return message
        return await self.app(scope,bounded_receive,send)


def create_app(settings=None, *, platform=None, start_workers=True):
    settings = settings or (platform.settings if platform else Settings())

    @asynccontextmanager
    async def lifespan(app):
        app.state.platform = platform or Platform(settings)
        if start_workers: app.state.platform.start()
        yield
        await asyncio.to_thread(app.state.platform.close)

    app = FastAPI(title='Mandol Agent Memory Platform', version='3.2.0-chinasoft', lifespan=lifespan)
    app.add_middleware(BoundedRequestBody,max_bytes=settings.max_media_bytes+65536)

    def p(request): return request.app.state.platform

    def principal(request):
        value=p(request).accounts.authenticate(request.cookies.get(COOKIE))
        if not value:raise AppError('请登录后继续',401,code='unauthorized')
        return value

    def owner(request,workspace=None):
        account=principal(request)
        return p(request).accounts.owner_for(account['id'],workspace or request.query_params.get('workspace_id'))

    def local_repository_access(request):
        account=principal(request)
        if account['role']!='admin':raise AppError('服务器本地仓库路径仅对管理员开放',403,code='repository_forbidden')

    def selected_mode(mode=None,family=None):
        if family and family not in MODES:raise AppError('无效记忆场景')
        result=MODES.get(family) if family else mode or 'textual'
        if result not in FAMILIES or (family and mode and result!=mode):raise AppError('记忆场景参数不一致')
        return result

    def citations_current(platform,user,workspace,citations):
        for citation in citations:
            try: scope,current,_=platform.locate(user,workspace,citation['id'])
            except AppError:return False
            if current.version!=citation['version'] or not platform.eligible(scope,current):return False
        return True

    def files_current(platform,user,workspace,route):
        for source in route.get('attachment_sources',[]):
            try:current=platform.store.attachment(user,source['attachment_id'],workspace)
            except AppError:return False
            if not current['active'] or current['hash']!=source['file_hash'] or current['version']!=source.get('version',1):return False
        return True

    @app.middleware('http')
    async def bounds(request, call_next):
        if request.method in ('POST','PUT','DELETE','PATCH'):
            origin = request.headers.get('origin')
            if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
                return JSONResponse({'error':{'code':'origin_rejected','message':'请求来源校验失败'}},status_code=403)
            try: content_length = int(request.headers.get('content-length','0'))
            except ValueError: return JSONResponse({'error':{'code':'invalid_length','message':'请求长度无效'}},status_code=400)
            if content_length > settings.max_media_bytes+65536:
                return JSONResponse({'error':{'code':'payload_too_large','message':'请求超过 8 MiB 上限'}},status_code=413)
            if request.url.path.startswith('/api/'):
                public_auth=request.url.path in ('/api/auth/login','/api/auth/register','/api/auth/session')
                if public_auth:
                    if request.headers.get('x-mandol-client')!='studio' or not request.headers.get('content-type','').startswith('application/json'):
                        return JSONResponse({'error':{'code':'csrf_rejected','message':'认证请求校验失败'}},status_code=403)
                else:
                    account=p(request).accounts.authenticate(request.cookies.get(COOKIE))
                    if not account:return JSONResponse({'error':{'code':'unauthorized','message':'请登录后继续'}},status_code=401)
                    if not hmac.compare_digest(request.headers.get('x-csrf-token',''),account['csrf']):
                        return JSONResponse({'error':{'code':'csrf_rejected','message':'会话校验失败，请重新打开页面'}},status_code=403)
                    if request.url.path in ('/api/media','/api/attachments'):
                        # Admission before multipart parsing/spooling consumes the body.
                        try:
                            with p(request).state.transaction() as db:p(request).store.access.consume(db,account['id'],'uploads_per_minute',60000)
                        except AppError as error:
                            return JSONResponse({'error':{'code':error.code,'message':error.message}},status_code=error.status)
        result = await call_next(request)
        result.headers['X-Content-Type-Options']='nosniff'
        result.headers['Referrer-Policy']='same-origin'
        result.headers['X-Frame-Options']='DENY'
        result.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        if request.url.path.startswith('/api/'): result.headers['Cache-Control']='no-store'
        return result

    @app.exception_handler(AppError)
    async def app_error(request, error):
        return JSONResponse({'error':{'code':error.code,'message':error.message}},status_code=error.status,
                            headers={'Retry-After':'3'} if error.code=='queue_full' else None)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request,error):
        # Pydantic's default response echoes input, including supplied passwords.
        return JSONResponse({'error':{'code':'validation_error','message':'请求字段格式无效，请检查必填项、确认密码及长度限制'},
                             'fields':[list(e['loc']) for e in error.errors()][:12]},status_code=422)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({'error':{'code':'conflict','message':'请求标识已使用或记忆版本冲突'}},status_code=409)

    @app.exception_handler(ModelError)
    async def model_error(request,error):
        return JSONResponse({'error':{'code':error.code,'message':error.message,'unknown_remote':error.unknown}},status_code=503)

    @app.exception_handler(StateUnavailable)
    async def storage_error(request,error):
        return JSONResponse({'error':{'code':'state_unavailable','message':'记忆状态暂不可用，请稍后重试'}},status_code=503)

    @app.exception_handler(Exception)
    async def internal_error(request,error):
        # No raw exception text, provider body, key, path or traceback in user responses.
        return JSONResponse({'error':{'code':'internal_error','message':'服务处理失败，请检查后台任务和系统状态'}},status_code=500)

    @app.get('/healthz')
    def health(request:Request):
        return {'status':'up','sqlite':p(request).state.health(),'service':'chinasoft','port':settings.port}

    @app.get('/api/auth/session')
    @app.post('/api/auth/session')
    def auth(request:Request):
        account=principal(request)
        return {'user':p(request).accounts.public(account),'csrf_token':account['csrf'],'workspaces':p(request).accounts.workspaces(account['id'])}

    @app.get('/api/auth/config')
    def auth_config():return {'registration_enabled':settings.registration_enabled,'approval_required':settings.registration_approval}

    @app.post('/api/auth/register',status_code=201)
    def register(data:Registration,request:Request):
        if not settings.registration_enabled:raise AppError('公开注册未开启，请联系管理员创建账号',403,code='registration_disabled')
        p(request).accounts.limit('register:'+request.client.host,5)
        p(request).accounts.limit('register:platform',50)
        try:p(request).accounts.create(data.username,data.password,data.display_name)
        except AppError as error:
            if error.code!='registration_failed':raise
        return {'status':'pending' if settings.registration_approval else 'registered','role':'user',
                'message':'登记请求已处理，请尝试登录或联系管理员；账号归属以登录结果为准'}

    @app.post('/api/auth/login')
    def login(data:Credentials,request:Request,response:Response):
        token,csrf,account=p(request).accounts.login(data.username,data.password,request.client.host)
        p(request).accounts.revoke(request.cookies.get(COOKIE))
        response.set_cookie(COOKIE,token,httponly=True,secure=settings.secure_cookie,samesite='strict',max_age=settings.session_ttl,path='/')
        return {'user':account,'csrf_token':csrf,'workspaces':p(request).accounts.workspaces(account['id'])}

    @app.post('/api/auth/logout')
    def logout(request:Request,response:Response):
        principal(request);p(request).accounts.revoke(request.cookies.get(COOKIE));response.delete_cookie(COOKIE,path='/');return {'logged_out':True}

    from .admin_api import install as install_admin
    install_admin(app,p,principal)

    @app.get('/api/workspaces')
    def workspaces(request:Request):
        return {'items':p(request).accounts.workspaces(principal(request)['id'])}

    @app.post('/api/workspaces',status_code=201)
    def workspace(data:Workspace,request:Request): return p(request).store.create_workspace(owner(request),data.name)

    @app.get('/api/sessions')
    def sessions(request:Request,workspace_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=40,ge=1,le=100),archived:bool=False,q:str=Query(default='',max_length=80)):
        user=owner(request);p(request).store.workspace(user,workspace_id)
        where='owner=? AND workspace=? AND archived=? AND deleted=0 AND instr(title,?)>0';args=(user,workspace_id,int(archived),q)
        return {'items':p(request).store.rows('SELECT id,workspace,mode,title,created_at,updated_at,manual_title,archived,unified,repository AS repository_id FROM cs_sessions WHERE '+where+' ORDER BY updated_at DESC,id DESC LIMIT ? OFFSET ?',args+(limit,offset)),
                'total':p(request).store.rows('SELECT COUNT(*) n FROM cs_sessions WHERE '+where,args)[0]['n'],'offset':offset,'limit':limit}

    @app.post('/api/sessions',status_code=201)
    def session(data:Session,request:Request):
        result=p(request).store.create_session(owner(request,data.workspace_id),data.workspace_id,data.mode,data.title,data.repository_id,data.unified)
        return {**result,'repository_id':result['repository']}

    @app.get('/api/sessions/{key}')
    def get_session(key:str,request:Request,workspace_id:str):
        result=p(request).store.session(owner(request),workspace_id,key);return {**result,'repository_id':result['repository']}

    @app.patch('/api/sessions/{key}')
    def update_session(key:str,data:SessionUpdate,request:Request,workspace_id:str):
        user=owner(request);store=p(request).store;store.session(user,workspace_id,key)
        fields={}
        if data.unified is not None:
            if not data.unified:raise AppError('统一对话不能降级')
            fields['unified']=1
        if data.title is not None:
            if not data.title.strip():raise AppError('标题不能为空')
            fields.update(title=data.title.strip(),manual_title=1)
        if data.archived is not None:fields['archived']=int(data.archived)
        if 'repository_id' in data.model_fields_set:
            if data.repository_id:store.repository(user,workspace_id,data.repository_id)
            fields['repository']=data.repository_id or ''
        if fields:
            with store.state.transaction() as db:db.execute('UPDATE cs_sessions SET '+','.join(k+'=?' for k in fields)+',updated_at=? WHERE id=? AND owner=? AND deleted=0',list(fields.values())+[now(),key,user])
        return {**store.session(user,workspace_id,key),'repository_id':fields.get('repository',store.session(user,workspace_id,key)['repository'])}

    @app.delete('/api/sessions/{key}')
    def delete_session(key:str,request:Request,workspace_id:str,confirm:bool=False):
        if not confirm:raise AppError('删除会话须显式确认',409)
        user=owner(request);store=p(request).store;store.session(user,workspace_id,key)
        with store.state.transaction() as db:
            db.execute('UPDATE cs_sessions SET deleted=1,updated_at=? WHERE id=? AND owner=?',(now(),key,user))
            db.execute("UPDATE cs_jobs SET status='cancelled',error_code='conversation_deleted',updated_at=? WHERE owner=? AND ingestion IN (SELECT id FROM cs_ingestions WHERE owner=? AND session=?) AND status IN ('pending','blocked','retry_wait')",(now(),user,user,key))
        return {'deleted':True,'retention':'soft_deleted_for_recovery','session_id':key}

    @app.get('/api/memory-types')
    def memory_types(request:Request,workspace_id:str,family:str='conversation'):
        user=owner(request);p(request).store.workspace(user,workspace_id);mode=selected_mode(family=family)
        return {'family':family,'items':list(TYPE_MAP[mode].values())}

    @app.get('/api/memory-catalog')
    def catalog(request:Request,workspace_id:str,session_id:str|None=None,state_view:str='current'):
        from .visualization import MemoryViews
        records,_=MemoryViews(p(request)).select(owner(request),workspace_id,session_id,limit=500,state_view=state_view)
        return {'items':sorted({p(request).serialize(s,m)['memory_type'] for s,m,_ in records}),'sample_budget':500}

    @app.get('/api/memory-schema')
    def memory_schema(request:Request,workspace_id:str,session_id:str|None=None,retrieval_scope:str='conversation',repository_id:str|None=None,state_view:str='current'):
        from .semantic_schema import overview
        if retrieval_scope not in ('conversation','conversations') or retrieval_scope=='conversation' and not session_id:raise AppError('请选择有效会话范围')
        return overview(p(request),owner(request,workspace_id),workspace_id,session_id if retrieval_scope=='conversation' else None,repository_id,state_view=state_view)

    @app.get('/api/repositories')
    def repositories(request:Request,workspace_id:str):
        user=owner(request);p(request).store.workspace(user,workspace_id)
        return {'items':[p(request).repositories.public(r) for r in p(request).store.rows('SELECT * FROM cs_repositories WHERE owner=? AND workspace=? AND deleted=0 ORDER BY created_at',(user,workspace_id))],
                'allowed_roots':settings.public()['repository_roots'] if principal(request)['role']=='admin' else [],
                'remote_enabled':settings.remote_git_enabled,'remote_platforms':['github.com','gitlab.com'],
                'remote_limits':{'files':settings.remote_git_max_files,'file_bytes':settings.repository_max_file_bytes,'total_bytes':settings.remote_git_max_bytes,'timeout_seconds':settings.remote_git_timeout}}

    @app.post('/api/repositories/remote',status_code=202)
    def import_remote(data:RemoteRepositoryRegistration,request:Request):
        user=owner(request,data.workspace_id);p(request).accounts.limit('remote:'+principal(request)['id'],15)
        return p(request).repositories.remote.register(user,data.workspace_id,data.url,data.ref,data.name,data.request_id)

    @app.post('/api/repositories',status_code=202)
    def register_repository(data:RepositoryRegistration,request:Request):
        local_repository_access(request)
        return p(request).repositories.register(owner(request,data.workspace_id),data.workspace_id,data.root,data.name,data.request_id)

    @app.get('/api/repositories/{key}')
    def repository(key:str,request:Request,workspace_id:str):
        return p(request).repositories.current(owner(request),workspace_id,key)

    @app.post('/api/repositories/{key}/scan',status_code=202)
    def scan_repository(key:str,data:RepositoryScan,request:Request,workspace_id:str):
        user=owner(request,workspace_id);repo=p(request).store.repository(user,workspace_id,key)
        if repo['source_kind']=='local':local_repository_access(request)
        p(request).accounts.limit('remote:'+principal(request)['id'],15)
        return p(request).repositories.rescan(user,workspace_id,key,data.request_id)

    @app.delete('/api/repositories/{key}')
    def remove_repository(key:str,request:Request,workspace_id:str,confirm:bool=False):
        user=owner(request,workspace_id);platform=p(request);platform.store.repository(user,workspace_id,key)
        if not confirm:raise AppError('请确认移除仓库；历史源码证据保留，退出当前问答',409)
        with platform.state.transaction() as db:
            db.execute('UPDATE cs_repositories SET deleted=1,status=\'removed\' WHERE id=?',(key,))
            db.execute("UPDATE cs_jobs SET error_code='cancel_requested',status=CASE WHEN status='running' THEN status ELSE 'cancelled' END,updated_at=? WHERE owner=? AND workspace=? AND json_extract(payload,'$.repository_id')=? AND status IN ('pending','running','retry_wait','blocked')",(now(),user,workspace_id,key))
            db.execute("UPDATE cs_sessions SET repository='' WHERE owner=? AND workspace=? AND repository=?",(user,workspace_id,key))
        return {'id':key,'deleted':True,'historical_evidence_retained':True}

    @app.get('/api/repositories/{key}/files')
    def repository_files(key:str,request:Request,workspace_id:str,q:str=Query(default='',max_length=200),offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=30,ge=1,le=50)):
        return p(request).repositories.files(owner(request),workspace_id,key,offset,limit,q)

    @app.get('/api/repositories/{key}/symbols')
    def repository_symbols(key:str,request:Request,workspace_id:str,q:str=Query(default='',max_length=200),path:str|None=None,offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=30,ge=1,le=50)):
        return p(request).repositories.symbols(owner(request),workspace_id,key,offset,limit,q,path)

    @app.get('/api/repositories/{key}/map')
    def repository_map(key:str,request:Request,workspace_id:str):
        user=owner(request);p(request).store.repository(user,workspace_id,key)
        return p(request).explorer(user,workspace_id,'code',repository_id=key,memory_type='repository_map',limit=1)

    @app.get('/api/repositories/{key}/history')
    def repository_history(key:str,request:Request,workspace_id:str,path:str=Query(max_length=1000),offset:int=Query(default=0,ge=0),limit:int=Query(default=24,ge=1,le=50)):
        return p(request).repositories.history(owner(request),workspace_id,key,path,offset,limit)

    @app.post('/api/memories',status_code=202)
    def add(data:AddMemory,request:Request):
        start=time.monotonic();result=p(request).store.accept(owner(request,data.workspace_id),data);p(request).store.metric('add_accept',(time.monotonic()-start)*1000);return result

    @app.get('/api/ingestions/{key}')
    def ingestion(key:str,request:Request,workspace_id:str):
        user=owner(request);p(request).store.workspace(user,workspace_id)
        ing=p(request).store.ingestion(user,key)
        if ing['workspace']!=workspace_id:raise AppError('提交记录不属于当前工作区',404)
        ing.pop('payload');ing.pop('digest');return ing

    @app.get('/api/memories/search')
    def search(request:Request,workspace_id:str,q:str=Query(min_length=1,max_length=1000),mode:str|None=None,family:str|None=None,repository_id:str|None=None,
               retrieval_scope:str='current',session_id:str|None=None,memory_type:str|None=None,source_kind:str|None=None,limit:int=Query(default=8,ge=1,le=12)):
        user=owner(request,workspace_id)
        with p(request).state.transaction() as db:p(request).store.access.consume(db,principal(request)['id'],'retrieval_per_minute',60000)
        selected=selected_mode(mode,family)
        if retrieval_scope in ('conversation','conversations'):
            if retrieval_scope=='conversation' and not session_id:raise AppError('本对话检索需要会话标识')
            from .routing import query_mode
            return p(request).search(owner(request),workspace_id,q,query_mode(q,repository_id),limit,repository_id=repository_id,retrieval_scope='workspace',memory_type=memory_type,source_kind=source_kind,session_id=session_id if retrieval_scope=='conversation' else None,conversation_first=True)
        return p(request).search(owner(request),workspace_id,q,selected,limit,repository_id=repository_id,retrieval_scope=retrieval_scope,memory_type=memory_type,source_kind=source_kind)

    @app.get('/api/memories')
    def memories(request:Request,workspace_id:str,mode:str|None=None,level:str|None=None,session_id:str|None=None,
                 q:str=Query(default='',max_length=300),offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=24,ge=1,le=50),family:str|None=None,
                 repository_id:str|None=None,memory_type:str|None=None,source_kind:str|None=None,retrieval_scope:str='current',state_view:str='current'):
        selected=selected_mode(mode,family)
        if retrieval_scope in ('conversation','conversations'):
            from .visualization import MemoryViews
            if retrieval_scope=='conversation' and not session_id:raise AppError('请选择会话')
            views=MemoryViews(p(request));records,total=views.select(owner(request),workspace_id,session_id if retrieval_scope=='conversation' else None,repository_id,source_kind,memory_type,offset,limit,keyword=q,state_view=state_view)
            items=[p(request).serialize(s,m) for s,m,_ in records]
            if state_view=='current':items=[m for m in items if m['eligible']]
            return {'items':items,'total':total,'offset':offset,'limit':limit,'selection':views.selection_info,'state_view':state_view}
        if level not in (None,'L0','L1','L2'):raise AppError('无效筛选条件')
        return p(request).explorer(owner(request),workspace_id,selected,level,session_id,q,offset,limit,repository_id=repository_id,memory_type=memory_type,source_kind=source_kind,retrieval_scope=retrieval_scope,state_view=state_view)

    @app.get('/api/memories/{key}')
    def memory(key:str,request:Request,workspace_id:str,family:str|None=None,repository_id:str|None=None,audit:bool=False):
        if audit:
            from .visualization import MemoryViews
            scope,m,_=MemoryViews(p(request)).locate(owner(request),workspace_id,key,repository_id)
            return p(request).serialize(scope,m,detail=True)
        scope,m,_=p(request).locate(owner(request),workspace_id,key,selected_mode(family=family) if family else None,repository_id);return p(request).serialize(scope,m,detail=True)

    @app.get('/api/memories/{key}/relations')
    def relations(key:str,request:Request,workspace_id:str,state_view:str='current'):
        from .visualization import MemoryViews
        return MemoryViews(p(request)).relations(owner(request),workspace_id,key,state_view=state_view)

    @app.get('/api/memories/{key}/history')
    def memory_history(key:str,request:Request,workspace_id:str):
        from .visualization import MemoryViews
        return MemoryViews(p(request)).history(owner(request),workspace_id,key)

    @app.get('/api/memory-groups/{key}')
    def group_members(key:str,request:Request,workspace_id:str,session_id:str|None=None,retrieval_scope:str='conversation',repository_id:str|None=None,state_view:str='current',offset:int=Query(default=0,ge=0),limit:int=Query(default=24,ge=1,le=50)):
        from .visualization import MemoryViews
        if retrieval_scope not in ('conversation','conversations') or retrieval_scope=='conversation' and not session_id:raise AppError('请选择有效范围')
        return MemoryViews(p(request)).members(owner(request),workspace_id,key,session_id if retrieval_scope=='conversation' else None,repository_id,state_view,offset,limit)

    @app.put('/api/memories/{key}',status_code=202)
    def revise(key:str,data:Revision,request:Request,workspace_id:str):
        user=owner(request);scope,m,info=p(request).locate(user,workspace_id,key)
        if m.memory_level!='L0_Observation':raise AppError('仅可修改基础记忆，高阶记忆由系统构建')
        if m.memory_type=='repository_chunk':raise AppError('源码只读；请在服务器修改授权仓库后重新扫描',409)
        if m.metadata.get('attachment_id'):raise AppError('文件内容不可在记忆编辑器中覆盖；请上传同名文件的新版本',409)
        if m.version!=data.expected_version:raise AppError('记忆已更新，请刷新后重试',409)
        return p(request).store.accept(user,{'workspace_id':workspace_id,'session_id':m.session_id,'request_id':data.request_id,'mode':info['mode'],'messages':[],
                                            'revision':{'uid':key,'repository_id':info['repository'],**data.model_dump()}})

    @app.post('/api/workspaces/{workspace_id}/memories/clear')
    def clear_memories(workspace_id:str,data:ClearMemories,request:Request):
        user=owner(request,workspace_id)
        from .memory_clear import CONFIRMATION
        if data.confirmation!=CONFIRMATION:
            raise AppError('请输入完全一致的“清除所有记忆”以确认',400,code='confirmation_required')
        return p(request).memory_clearing.clear(user,workspace_id,data.request_id)

    @app.delete('/api/memories/{key}',status_code=202)
    def delete(key:str,request:Request,workspace_id:str,expected_version:int,request_id:str):
        user=owner(request);scope,m,info=p(request).locate(user,workspace_id,key)
        if m.memory_level!='L0_Observation':raise AppError('请从基础来源管理高阶记忆的有效性')
        if m.memory_type=='repository_chunk':raise AppError('仓库索引由重新扫描维护，不能直接删除源码证据',409)
        if m.version!=expected_version:raise AppError('记忆版本冲突',409)
        receipt=p(request).store.accept(user,{'workspace_id':workspace_id,'session_id':m.session_id,'request_id':request_id,'mode':info['mode'],'messages':[],
                                            'revision':{'uid':key,'expected_version':expected_version,'delete':True,'repository_id':info['repository']}})
        if m.metadata.get('attachment_id'):
            with p(request).state.transaction() as db:
                db.execute('UPDATE cs_attachments SET active=0 WHERE id=? AND owner=? AND workspace=?',(m.metadata['attachment_id'],user,workspace_id))
        return receipt

    @app.post('/api/media',status_code=201)
    async def upload(request:Request,workspace_id:str=Form(),session_id:str=Form(),file:UploadFile=File()):
        user=owner(request,workspace_id);chunks=[];size=0
        while True:
            chunk=await file.read(256*1024)
            if not chunk:break
            size+=len(chunk)
            if size>settings.max_media_bytes:raise AppError('图片最大 8 MiB',413)
            chunks.append(chunk)
        media=await asyncio.to_thread(p(request).save_media,user,workspace_id,session_id,b''.join(chunks))
        return {k:media[k] for k in ('id','mime','width','height','bytes','session','workspace')}

    @app.get('/api/media/{key}')
    def media(key:str,request:Request,workspace_id:str,thumb:bool=False):
        user=owner(request);p(request).store.workspace(user,workspace_id)
        record=p(request).store.media(user,key,workspace_id)
        p(request).store.session(user,workspace_id,record['session'])
        return FileResponse(settings.data_dir/(record['thumb'] if thumb else record['path']),media_type='image/jpeg' if thumb else record['mime'])

    @app.post('/api/attachments',status_code=201)
    async def upload_attachment(request:Request,workspace_id:str=Form(),session_id:str=Form(),file:UploadFile=File()):
        from .attachments import save_attachment
        user=owner(request,workspace_id);content=await file.read(settings.attachment_max_bytes+1)
        return await asyncio.to_thread(save_attachment,p(request),user,workspace_id,session_id,file.filename,content,file.content_type)

    @app.get('/api/attachments/{key}')
    def attachment(key:str,request:Request,workspace_id:str):
        a=p(request).store.attachment(owner(request),key,workspace_id)
        return {k:v for k,v in a.items() if k not in ('owner','hash')}

    @app.delete('/api/attachments/{key}')
    def delete_attachment(key:str,request:Request,workspace_id:str):
        user=owner(request);platform=p(request);platform.store.attachment(user,key,workspace_id)
        with platform.state.transaction() as db:
            db.execute('UPDATE cs_attachments SET active=0 WHERE id=? AND owner=? AND workspace=?',(key,user,workspace_id))
        return {'id':key,'active':False,'historical_evidence_retained':True}

    @app.get('/api/memory-graph')
    def graph(request:Request,workspace_id:str,session_id:str|None=None,retrieval_scope:str='conversation',repository_id:str|None=None,source_kind:str|None=None,memory_type:str|None=None,
              offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=100,ge=1,le=500),center:str|None=None,relationship_view:str='all',q:str=Query(default='',max_length=200),relation:str|None=Query(default=None,max_length=64),state_view:str='current',aggregate:bool=False,expand_group:str|None=Query(default=None,max_length=100)):
        from .visualization import MemoryViews
        user=owner(request,workspace_id)
        if retrieval_scope not in ('conversation','conversations') or (retrieval_scope=='conversation' and not session_id):raise AppError('无效会话范围')
        return MemoryViews(p(request)).graph(user,workspace_id,session_id if retrieval_scope=='conversation' else None,repository_id,source_kind,memory_type,offset,limit,center,relationship_view,q,relation,state_view,aggregate,expand_group)

    @app.get('/api/sessions/{key}/timeline')
    def timeline(key:str,request:Request,workspace_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=30,ge=1,le=50)):
        from .visualization import MemoryViews
        return MemoryViews(p(request)).timeline(owner(request),workspace_id,key,offset,limit)

    @app.get('/api/jobs')
    def jobs(request:Request,workspace_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=30,ge=1,le=50)):
        return p(request).store.jobs(owner(request),workspace_id,offset,limit)

    @app.post('/api/jobs/{key}/retry',status_code=202)
    def retry(key:str,data:Retry,request:Request,workspace_id:str):
        user=owner(request);platform=p(request);versions=None
        platform.store.workspace(user,workspace_id)
        jobs=platform.store.rows('SELECT * FROM cs_jobs WHERE id=? AND owner=?',(key,user))
        if not jobs or jobs[0]['workspace']!=workspace_id:raise AppError('任务不属于当前工作区',404)
        if jobs[0]['kind']=='repository':
            repo=platform.store.repository(user,workspace_id,json.loads(jobs[0]['payload'])['repository_id'])
            if repo['source_kind']=='local':local_repository_access(request)
            with platform.state.transaction() as db:platform.store.access.check(db,user,'repository_imports')
        if jobs and jobs[0]['kind']=='derive' and jobs[0]['ingestion']:
            ing=platform.store.ingestion(user,jobs[0]['ingestion'])
            if ing['mode']=='code' or json.loads(jobs[0]['payload']).get('route')=='engineering':
                sources,_=platform.session_sources(platform.raw_scope(ing),ing['session'])
                if json.loads(jobs[0]['payload']).get('route')=='engineering':sources=[m for m in sources if 'engineering' in m.metadata.get('routes',[])]
                versions={m.logical_uid:m.version for m in sources}
        return platform.store.retry(user,key,data.allow_unknown,versions)

    @app.post('/api/jobs/{key}/cancel',status_code=202)
    def cancel_repository_job(key:str,request:Request,workspace_id:str):
        user=owner(request,workspace_id);platform=p(request)
        jobs=platform.store.rows('SELECT * FROM cs_jobs WHERE id=? AND owner=? AND workspace=?',(key,user,workspace_id))
        if not jobs:raise AppError('任务不存在',404)
        job=jobs[0]
        if job['kind']!='repository':raise AppError('此入口只允许安全停止仓库扫描',409)
        if job['status'] not in ('pending','running','retry_wait'):return {'id':key,'status':job['status']}
        with platform.state.transaction() as db:
            if job['status']=='running':
                db.execute("UPDATE cs_jobs SET error_code='cancel_requested',updated_at=? WHERE id=? AND status='running'",(now(),key))
            else:
                db.execute("UPDATE cs_jobs SET status='cancelled',error_code='repository_cancelled',lease=NULL,updated_at=? WHERE id=? AND status IN ('pending','retry_wait')",(now(),key))
                db.execute("UPDATE cs_repositories SET status='cancelled' WHERE id=? AND owner=?",(json.loads(job['payload'])['repository_id'],user))
        return {'id':key,'status':'stopping' if job['status']=='running' else 'cancelled','partial_evidence_retained':True}

    @app.get('/api/events')
    async def events(request:Request,workspace_id:str):
        user=owner(request);p(request).store.workspace(user,workspace_id)
        async def changes():
            previous=None;beats=0
            while not await request.is_disconnected():
                if not p(request).accounts.authenticate(request.cookies.get(COOKIE)):
                    yield 'event: auth_expired\ndata: {}\n\n';return
                rows=p(request).store.rows('SELECT status,COUNT(*) n,MAX(updated_at) changed FROM cs_jobs WHERE owner=? AND workspace=? GROUP BY status',(user,workspace_id))
                snapshot=canonical_json(rows)
                if snapshot!=previous:
                    previous=snapshot;yield 'event: jobs\ndata: '+snapshot+'\n\n'
                beats+=1
                if beats%5==0:yield ': heartbeat\n\n'
                await asyncio.sleep(2)
        return StreamingResponse(changes(),media_type='text/event-stream',headers={'X-Accel-Buffering':'no'})

    @app.get('/api/status')
    def status(request:Request,workspace_id:str):
        result=p(request).status(owner(request),workspace_id)
        # Filesystem allowlists are operational information for trusted administrators.
        if principal(request)['role']!='admin':
            result['metrics']=[]
            if isinstance(result.get('config'),dict):result['config'].pop('repository_roots',None)
            result['retrieval']={k:v for k,v in result['retrieval'].items() if k not in ('collection',)}
            for key in ('sparse',):
                result['retrieval'][key]={k:v for k,v in result['retrieval'].get(key,{}).items() if k in ('ready','reason','model','device','checked_at')}
            result['retrieval']['models']={name:{k:v for k,v in value.items() if k in ('ready','reason','model','device','checked_at')} for name,value in result['retrieval'].get('models',{}).items()}
        return result

    @app.get('/api/chat/history')
    def history(request:Request,workspace_id:str,session_id:str,offset:int=Query(default=0,ge=0),limit:int=Query(default=40,ge=1,le=60)):
        user=owner(request);store=p(request).store;store.session(user,workspace_id,session_id)
        rows=store.rows('SELECT id,request_id,question,answer,media,citations,status,model,created_at,error,route,attachments FROM cs_chat WHERE owner=? AND workspace=? AND session=? ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?', (user,workspace_id,session_id,limit,offset))
        for row in rows:
            row['attachment_ids']=json.loads(row.pop('attachments'))
            row['route']=json.loads(row['route'])
            row['attachment_files']=[]
            for key in row['attachment_ids']:
                files=store.rows('SELECT id,name,version,active FROM cs_attachments WHERE id=? AND owner=? AND workspace=? AND session=?',(key,user,workspace_id,session_id))
                row['attachment_files'].extend(files)
            row['media']=json.loads(row['media']);row['citations']=json.loads(row['citations'])
            if row['status']=='completed' and (not citations_current(p(request),user,workspace_id,row['citations']) or not files_current(p(request),user,workspace_id,row['route'])):
                row['status']='stale';row['error']='历史回答的来源已更新，请基于当前版本重新提问'
        return {'items':list(reversed(rows))}

    @app.get('/api/chat/capabilities')
    def chat_capabilities(request:Request):
        principal(request)
        return {'generation':settings.reasoning_capabilities(),'vision_generation':settings.reasoning_capabilities(True),'web':p(request).web.public()}

    @app.post('/api/chat')
    async def chat(data:Chat,request:Request):
        user=owner(request,data.workspace_id);platform=p(request);store=platform.store;session=store.session(user,data.workspace_id,data.session_id)
        from .attachments import SENSITIVE
        if SENSITIVE.search(data.content):raise AppError('消息疑似包含凭据，请脱敏后再发给模型',422,code='sensitive_input')
        if data.reasoning_mode not in settings.reasoning_capabilities(bool(data.media_ids))['modes']:raise AppError('此端点未配置该推理模式，请使用快速模式',422,code='reasoning_unsupported')
        from .web_tools import search_query
        # Legacy explicit queries remain optional. Automatic planning runs only
        # after the original message is reliably saved and context is authorized.
        public_query=search_query(data.content,data.web_query) if data.web_search and data.web_query else ''
        route_mode=session['mode'] if data.mode=='auto' else data.mode
        if not session['unified'] and route_mode!=session['mode']:raise AppError('提问场景与会话归属不一致',409)
        repository_id=session['repository']
        if session['unified']:
            from .routing import query_mode
            route_mode=query_mode(data.content,repository_id)
        if data.repository_id is not None and data.repository_id!=repository_id:raise AppError('提问仓库与会话归属不一致',409)
        if repository_id:store.repository(user,data.workspace_id,repository_id)
        if data.media_ids and not session['unified'] and session['mode']!='multimodal':raise AppError('请在统一会话或多模态会话上传图片')
        for mid in data.media_ids:store.media(user,mid,data.workspace_id,data.session_id)
        for aid in data.attachment_ids:store.attachment(user,aid,data.workspace_id,data.session_id)
        request_data=data.model_dump()
        if not data.web_search and not data.web_query:request_data.pop('web_search');request_data.pop('web_query')
        if data.reasoning_mode=='fast':request_data.pop('reasoning_mode')
        fingerprint=digest(request_data);chat_id='chat_'+digest([user,data.workspace_id,data.request_id]);prior_route={}
        initial_route={'options':{'reasoning_mode':data.reasoning_mode,'web_search':data.web_search,'web_query':data.web_query,'retrieval_scope':data.retrieval_scope},
                       'generation':{'mode':data.reasoning_mode,'parameters':settings.reasoning_capabilities(bool(data.media_ids))['mapping'][data.reasoning_mode]}}
        with platform.state.transaction() as db:
            store.assert_memory_writable(db,user,data.workspace_id)
            old=db.execute('SELECT * FROM cs_chat WHERE id=?',(chat_id,)).fetchone()
            if old:prior_route=json.loads(old['route'])
            if not old or old['status'] not in ('processing','completed'):
                account=principal(request)['id']
                store.access.consume(db,account,'chat_per_minute',60000)
                store.access.check(db,user,'model_per_day')
                active=db.execute("SELECT COUNT(*) FROM cs_chat c JOIN cs_account_owners o ON o.owner=c.owner WHERE o.account=? AND c.status='processing'",(account,)).fetchone()[0]
                if active>=2:raise AppError('当前账号已有回答正在生成，请等待或中断后重试',429,code='chat_busy')
            if old:
                if old['digest']!=fingerprint:raise Conflict('相同聊天请求对应不同内容')
                if old['status']=='processing':raise AppError('该问题正在处理',409)
                if old['status']!='completed':
                    db.execute("UPDATE cs_chat SET status='processing',updated_at=?,error=NULL WHERE id=?",(now(),chat_id));old=None
            else:
                db.execute('INSERT INTO cs_chat(id,owner,workspace,session,request_id,digest,question,media,status,created_at,updated_at,attachments) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', (chat_id,user,data.workspace_id,data.session_id,data.request_id,fingerprint,data.content,canonical_json(data.media_ids),'processing',now(),now(),canonical_json(data.attachment_ids)))
            if not old:db.execute('UPDATE cs_chat SET route=? WHERE id=?',(canonical_json(dict(prior_route,**initial_route)),chat_id))
        try:
            receipt=await asyncio.to_thread(store.accept,user,AddMemory(workspace_id=data.workspace_id,session_id=data.session_id,request_id='chat-memory-'+data.request_id,
                mode=session['mode'],repository_id=repository_id or None,attachment_ids=data.attachment_ids,messages=[{'role':'user','content':data.content,'media_ids':data.media_ids}]))
        except Exception:
            with platform.state.transaction() as db:db.execute("UPDATE cs_chat SET status='failed',error='问题未完成可靠入队，请重试' WHERE id=?",(chat_id,))
            raise

        def event(name,body):return 'event: '+name+'\ndata: '+canonical_json(body)+'\n\n'

        async def answer():
            text='';citations=[];start=time.monotonic();last_save=start;first=True
            web_cancel=None
            route=dict(prior_route,**initial_route)
            generation_meta={}
            account_token=platform.generation.account_context.set(principal(request)['id'])
            platform.generation.stream_metadata.set(None)
            try:
                yield event('accepted',{'chat_id':chat_id,'receipt':receipt})
                if old:
                    cached=dict(old)
                    if cached['status']=='completed' and (not citations_current(platform,user,data.workspace_id,json.loads(cached['citations'])) or not files_current(platform,user,data.workspace_id,json.loads(cached['route']))):
                        cached['status']='stale'
                    yield event('route',json.loads(cached['route']));yield event('sources',json.loads(cached['citations']));yield event('token',{'text':cached['answer']})
                    yield event('done',{'chat_id':chat_id,'status':cached['status'],'cached':True});return
                from .conversation import recent_turns, attachment_context, plan_turn, generation_messages, citation_numbers, select_evidence,web_citation_numbers
                created = store.rows('SELECT created_at FROM cs_chat WHERE id=?',(chat_id,))[0]['created_at']
                turns = await asyncio.to_thread(recent_turns,platform,user,data.workspace_id,data.session_id,chat_id,created)
                files, file_sources = await asyncio.to_thread(attachment_context,platform,user,data.workspace_id,data.session_id,data.attachment_ids,turns,data.content)
                # No wait for ingestion, vector publication, or high-order memory.
                yield event('status',{'message':'正在准备回答'})
                planning_start=time.monotonic()
                code_hint=None
                if settings.memory_query_strategy=='adaptive_v33':
                    from .typed_retrieval import qualified_code_hint
                    code_hint=await asyncio.to_thread(qualified_code_hint,platform,user,data.workspace_id,data.content,data.session_id if data.retrieval_scope in ('current','conversation') else None,repository_id)
                plan=await asyncio.to_thread(plan_turn,platform,data.content,turns,files,repository_id,bool(data.attachment_ids),bool(data.media_ids),**({'code_hint':code_hint} if code_hint else {}))
                web_files=files if plan.get('use_attachments',False) else []
                if not plan.get('use_attachments',False):files=[];file_sources=[]
                if not platform.accounts.authenticate(request.cookies.get(COOKIE)):raise asyncio.CancelledError()
                store.session(user,data.workspace_id,data.session_id)
                store.metric('chat_planning',(time.monotonic()-planning_start)*1000)
                unified=bool(session['unified']) or data.retrieval_scope in ('conversation','conversations')
                scope_name='conversation' if unified and data.retrieval_scope in ('conversation','current') else 'conversations' if unified else data.retrieval_scope
                route={'retrieval_scope':scope_name,'conversation_id':data.session_id,'planner':plan,
                       'retrieval_used':False,'recent_turns':len(turns),'attachment_sources':file_sources,'searched_scopes':0,'searched_families':[],'searched_repositories':[],
                       'options':{'reasoning_mode':data.reasoning_mode,'web_search':data.web_search,'web_query':data.web_query,'retrieval_scope':data.retrieval_scope},
                       'generation':{'mode':data.reasoning_mode,'parameters':settings.reasoning_capabilities(bool(data.media_ids))['mapping'][data.reasoning_mode]}}
                result={'items':[]}
                if plan['retrieve']:
                    yield event('status',{'message':'正在查找相关历史依据'})
                    retrieval_started=time.monotonic()
                    result=await asyncio.to_thread(platform.search,user,data.workspace_id,plan['query'] or data.content,route_mode,8,repository_id=repository_id,
                        retrieval_scope='workspace' if unified else data.retrieval_scope,session_id=data.session_id if scope_name=='conversation' else None,conversation_first=unified,exclude_sources=receipt['memory_ids'])
                    route.update(result['route']);route.update(retrieval_scope=scope_name,retrieval_used=True,conversation_id=data.session_id)
                    store.metric('chat_retrieval_'+route_mode,(time.monotonic()-retrieval_started)*1000)
                    selection_started=time.monotonic()
                    if route.get('trace',{}).get('strategy')=='adaptive_v33' and route.get('trace',{}).get('fast_exact') and len(result['items'])==1:
                        route['evidence_selection']={'status':'unique_verified_symbol','candidates':1,'selected':1,'extra_llm_calls':0}
                    else:result['items'],route['evidence_selection']=await asyncio.to_thread(select_evidence,platform,data.content,result['items'])
                    store.metric('chat_evidence_selection',(time.monotonic()-selection_started)*1000)
                from .query_planner import pack_context
                expand=bool(route.get('memory_query_plan',{}).get('expansion_policy',{}).get('raw_sources'))
                context,citations,packing=pack_context(result['items'],budget=12000,expand=expand,dedup_mode=route.get('context_dedup','quotes-v2'),with_basis=route.get('context_semantics',False))
                route['context_packing']=packing
                web=None
                if data.web_search:
                    from .web_query_planner import plan_web_query
                    prior_web=prior_route.get('web',{})
                    if prior_web.get('status') in ('completed','empty','partial'):
                        # Request fingerprint already binds question, session,
                        # attachments and options. Failed generation retries must
                        # not rerun paid planning/search that already completed.
                        web=dict(prior_web,reused_for_same_request=True)
                    elif not public_query and plan['intent'] in ('social','identity'):
                        web={'requested':True,'status':'skipped','reason':'current_context_sufficient','results':[],'calls':{'search':0,'fetch':0},
                             'plan':{'search_required':False,'planner_method':'conversation_fast_path','reason_codes':['current_context_sufficient'],'model_calls':0}}
                    else:
                        yield event('status',{'message':'正在组织搜索查询','stage':'web_planning'})
                        web_plan=prior_web.get('plan')
                        if not web_plan or not web_plan.get('primary_query'):
                            web_plan=await asyncio.to_thread(plan_web_query,platform,data.content,turns,web_files,explicit=public_query)
                        else:web_plan=dict(web_plan,reused_for_same_request=True)
                        route['web']={'requested':True,'status':'processing','query':web_plan['primary_query'],'plan':web_plan,'results':[],
                                      'calls':{'search':None,'fetch':None},'calls_known':False}
                        with platform.state.transaction() as db:db.execute('UPDATE cs_chat SET route=? WHERE id=?',(canonical_json(route),chat_id))
                        yield event('route',route)
                        import threading
                        web_cancel=threading.Event();stages=asyncio.Queue(maxsize=8);loop=asyncio.get_running_loop()
                        def progress(stage,values):
                            if web_cancel.is_set():raise ModelError('web_cancelled','本轮联网已中断')
                            loop.call_soon_threadsafe(stages.put_nowait,{'stage':'web_'+stage,'message':'正在搜索网页' if stage=='search' else '正在读取网页'})
                        web_task=asyncio.create_task(asyncio.to_thread(platform.web.search,user,data.workspace_id,web_plan['primary_query'],
                            additional_queries=web_plan['additional_queries'],progress=progress))
                        while not web_task.done():
                            if await request.is_disconnected() or not platform.accounts.authenticate(request.cookies.get(COOKIE)):raise asyncio.CancelledError()
                            await asyncio.wait({web_task},timeout=.1)
                            while not stages.empty():yield event('status',stages.get_nowait())
                        while not stages.empty():yield event('status',stages.get_nowait())
                        web=await web_task;web['plan']=web_plan
                    route['web']=web
                    if web['status']=='failed':yield event('status',{'message':'联网未成功，本轮没有网页依据：'+web.get('message','服务不可用'),'stage':'web_failed'})
                with platform.state.transaction() as db:db.execute('UPDATE cs_chat SET route=? WHERE id=?',(canonical_json(route),chat_id))
                if not platform.accounts.authenticate(request.cookies.get(COOKIE)):raise asyncio.CancelledError()
                yield event('route',route)
                media=[store.media(user,mid,data.workspace_id,data.session_id) for mid in data.media_ids]
                messages=generation_messages(platform,data.content,media,turns,files,context,web)
                yield event('status',{'message':'正在生成回答','stage':'generation'})
                generation_options={'reasoning_mode':data.reasoning_mode} if data.reasoning_mode!='fast' else {}
                async for token in platform.generation.stream(messages,vision=bool(data.media_ids),**generation_options):
                    generation_meta=platform.generation.stream_metadata.get() or generation_meta
                    if await request.is_disconnected():raise asyncio.CancelledError()
                    if not platform.accounts.authenticate(request.cookies.get(COOKIE)):raise asyncio.CancelledError()
                    if first:platform.store.metric('chat_first_token',(time.monotonic()-start)*1000);first=False
                    text+=token
                    if len(text)>40000:raise ModelError('answer_budget','回答超过最大长度')
                    yield event('token',{'text':token})
                    if time.monotonic()-last_save>0.8:
                        with platform.state.transaction() as db:db.execute('UPDATE cs_chat SET answer=?,updated_at=? WHERE id=?',(text,now(),chat_id))
                        last_save=time.monotonic()
                if not text.strip():raise ModelError('empty_response','模型返回空回答')
                used=citation_numbers(text)
                web_used=web_citation_numbers(text);web_records={r['id']:r for r in (web or {}).get('results',[])}
                if not web_used<=set(web_records):raise ModelError('invalid_web_citation','回答包含不存在的网页来源编号，未标记为完成')
                route['web_sources']=[r for key,r in web_records.items() if key in web_used]
                route['generation'].update(platform.generation.stream_metadata.get() or {})
                if any(n>len(context) or n<1 for n in used):raise ModelError('invalid_citation','回答包含不存在的来源编号，未标记为完成')
                citations=[c for c in citations if c['citation_index'] in used]
                # A file changed/deleted during generation cannot leave a completed stale answer.
                for source in file_sources:
                    current=store.attachment(user,source['attachment_id'],data.workspace_id,data.session_id)
                    if not current.get('active',1) or current['hash']!=source['file_hash'] or current.get('version',1)!=source['version']:
                        raise ModelError('source_changed','附件版本在回答期间发生变化，请重试')
                for citation in citations:
                    scope,current,_ = platform.locate(user,data.workspace_id,citation['id'])
                    if current.version != citation['version'] or not platform.eligible(scope,current):
                        raise ModelError('source_changed','来源在回答期间发生变化，请基于最新来源重新提问')
                if not platform.accounts.authenticate(request.cookies.get(COOKIE)):raise asyncio.CancelledError()
                store.session(user,data.workspace_id,data.session_id)
                yield event('sources',citations)
                yield event('route',route)
                with platform.state.transaction() as db:
                    db.execute("UPDATE cs_chat SET answer=?,citations=?,route=?,status='completed',model=?,updated_at=? WHERE id=?",(text,canonical_json(citations),canonical_json(route),platform.generation.stream_model.get() or (settings.vlm_request_model if data.media_ids else settings.llm_request_model),now(),chat_id))
                    db.execute('UPDATE cs_sessions SET updated_at=? WHERE id=? AND owner=?',(now(),data.session_id,user))
                store.metric('chat_complete',(time.monotonic()-start)*1000)
                yield event('done',{'chat_id':chat_id,'status':'completed','elapsed_ms':round((time.monotonic()-start)*1000),'request_model':settings.vlm_request_model if data.media_ids else settings.llm_request_model,'response_model':platform.generation.stream_model.get()})
            except (asyncio.CancelledError,GeneratorExit):
                if old:raise # A cancelled replay must never erase a completed answer.
                # Only the local snapshot is safe during cross-context iterator
                # finalization; an unrelated ContextVar must never supply it.
                route['generation'].update(generation_meta)
                if route.get('web',{}).get('status')=='processing':route['web'].update(status='interrupted',message='联网请求已中断，远程结果和费用未确认；手动重试可能再次调用')
                route['web_sources']=[r for r in route.get('web',{}).get('results',[]) if r['id'] in web_citation_numbers(text)] if text else []
                with platform.state.transaction() as db:db.execute("UPDATE cs_chat SET answer=?,citations=?,route=?,status='interrupted',updated_at=?,error='用户中断或网络断开' WHERE id=?",(text,canonical_json(citations),canonical_json(route),now(),chat_id))
                raise
            except Exception as exc:
                message=getattr(exc,'message','回答未完成，请检查系统状态后重试');code=getattr(exc,'code','generation_failed')
                if old:
                    yield event('error',{'code':code,'message':message,'chat_id':chat_id});return
                route['generation'].update(platform.generation.stream_metadata.get() or {})
                if route.get('web',{}).get('status')=='processing':route['web'].update(status='interrupted',message='联网结果未确认，未作为网页依据使用')
                route['web_sources']=[r for r in route.get('web',{}).get('results',[]) if r['id'] in web_citation_numbers(text)] if text else []
                with platform.state.transaction() as db:db.execute("UPDATE cs_chat SET answer=?,citations=?,route=?,status='failed',error=?,updated_at=? WHERE id=?",(text,canonical_json(citations),canonical_json(route),message,now(),chat_id))
                yield event('error',{'code':code,'message':message,'chat_id':chat_id})
            finally:
                if web_cancel is not None:web_cancel.set()
                try:platform.generation.account_context.reset(account_token)
                except ValueError:
                    # ASGI can finalize a suspended async iterator in a new
                    # context. The original request task is gone; never erase
                    # its durable state or leak a cleanup traceback to callers.
                    pass
        return StreamingResponse(answer(),media_type='text/event-stream',headers={'X-Accel-Buffering':'no'})

    @app.post('/api/demo/seed',status_code=202)
    def seed(request:Request):
        if not settings.demo_enabled:raise AppError('演示导入未启用',404)
        from .demo import seed_demo
        return seed_demo(p(request),owner(request))

    @app.delete('/api/demo/{workspace_id}',status_code=200)
    def reset_demo(workspace_id:str,request:Request):
        user=owner(request,workspace_id);w=p(request).store.workspace(user,workspace_id)
        if not w['demo']:raise AppError('仅可重置当前用户的独立演示工作区',403)
        scopes=p(request).store.scopes(user,workspace_id)
        with p(request).state.transaction() as db:
            active=db.execute("SELECT 1 FROM cs_jobs WHERE owner=? AND workspace=? AND status='running' LIMIT 1",(user,workspace_id)).fetchone()
            if active:raise AppError('演示后台任务尚在运行，完成后再重置',409)
            db.execute('UPDATE cs_workspaces SET demo=-1 WHERE owner=? AND id=?',(user,workspace_id))
            db.execute("UPDATE cs_jobs SET status='cancelled' WHERE owner=? AND workspace=? AND status IN ('pending','blocked','retry_wait')",(user,workspace_id))
        # Scope reset uses existing canonical mutations and owns no AML namespace.
        for scope,_ in scopes:
            rows=p(request).state._rows('SELECT logical_uid FROM memories WHERE scope=? AND active=1',(scope.key,))
            from mandol_substrate.state import Mutation
            mutation=Mutation(delete_uids=tuple(r['logical_uid'] for r in rows))
            p(request).coordinator.apply(scope,uid('demo-reset-'),digest([workspace_id,[r['logical_uid'] for r in rows]]),mutation,{})
        with p(request).state.transaction() as db:
            media=db.execute('SELECT path,thumb FROM cs_media WHERE owner=? AND workspace=?',(user,workspace_id)).fetchall()
            db.execute('DELETE FROM cs_media WHERE owner=? AND workspace=?',(user,workspace_id))
            db.execute('DELETE FROM cs_ingestions WHERE owner=? AND workspace=?',(user,workspace_id))
            db.execute('DELETE FROM cs_chat WHERE owner=? AND workspace=?',(user,workspace_id))
            db.execute('DELETE FROM cs_sessions WHERE owner=? AND workspace=?',(user,workspace_id))
            db.execute('DELETE FROM cs_workspaces WHERE owner=? AND id=?',(user,workspace_id))
        for record in media:
            for field in ('path','thumb'):(settings.data_dir/record[field]).unlink(missing_ok=True)
        return {'reset':True,'workspace_id':workspace_id}

    dist=Path(os.getenv('CHINASOFT_STUDIO_DIST',str(ROOT/'studio/dist'))).resolve()
    if (dist/'assets').is_dir():app.mount('/assets',StaticFiles(directory=dist/'assets'),name='studio-assets')

    @app.get('/{path:path}',include_in_schema=False)
    def studio(path:str):
        if path.startswith(('api/','healthz')):raise HTTPException(404)
        if not (dist/'index.html').is_file():return JSONResponse({'message':'Studio 尚未构建，请运行 npm --prefix studio run build'},status_code=503)
        return FileResponse(dist/'index.html',media_type='text/html')

    return app
